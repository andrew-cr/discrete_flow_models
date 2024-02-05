import os
import time
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
import uuid

# -----------------------------------------------------------------------------
# These configs will be overridden by the config file and so their values here do not matter.
out_dir = 'out'

run_name = 'gpt2' # 'run' + str(time.time())

# data
dataset = 'text8'
batch_size = 64
block_size = 256

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
qk_layernorm = True
do_x1_sc = False

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
data_dir = '/path/to/datasets/text8' #  directory should contain meta.pkl

# sampling
total_samples = 128
dt = 0.001
max_t = 0.98
argmax_final = True
noise = 0.0
x1_temp = 1.0
use_different_x1_sc_temp = False
x1_sc_temp = 1.0
ignore_x1_sc = False # If true, even if the model is self conditioned, we just put in the mask condition every iteration anyway

model_type = 'flow' # flow, d3pm

do_purity_sampling = False
purity_temp = 1.0

ckpt_path = 'out/ckpt.pt'

# d3pm settings
timesteps = 1000


# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert model_type in ['flow', 'd3pm']

hash = str(uuid.uuid1()).split("-")[0]
samples_dir = os.path.join(out_dir, 'samples_' + time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + hash)
os.mkdir(samples_dir)
with open(os.path.join(samples_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f, sort_keys=False)

with open(os.path.join(samples_dir, f'run_name_{run_name}.txt'), 'w') as f:
    f.write(f'{run_name}')


from flow_model import GPT, GPTConfig

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

stoi = meta['stoi']
itos = meta['itos']

if dataset == 'text8':
    # increase vocab size by 1 to include a mask token
    meta_vocab_size += 1
    mask_token_id = meta_vocab_size - 1
    stoi['X'] = mask_token_id
    itos[mask_token_id] = 'X'
else:
    raise NotImplementedError

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


device_type = 'cuda'
device = 'cuda:0'


def load_model(ckpt_path):
    # resume training from a checkpoint.
    print(f"Loading network from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model, checkpoint

model, checkpoint = load_model(ckpt_path)

# save the model information to the sample directory
model_information = {
    'model_args': checkpoint['model_args'],
    'iter_num': checkpoint['iter_num'],
    'best_val_loss': checkpoint['best_val_loss'],
    'config': checkpoint['config'],
}
torch.save(model_information, os.path.join(samples_dir, 'model_information.pt'))
checkpoint = None
model.eval()
model.to(device)

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
print(torch.__version__)
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------- SAMPLING CODE --------------=-

S = meta_vocab_size

B = batch_size
D = block_size

# write an empty file to store the samples eventually
with open(os.path.join(samples_dir, 'samples.txt'), 'w') as f:
    pass

assert total_samples % B == 0


with torch.no_grad():
    with ctx:

        mask_one_hot = torch.zeros((S,), device=device)
        mask_one_hot[mask_token_id] = 1.0


        for _ in range(total_samples // B):

            if model_type == 'flow':
                samples = mask_token_id * torch.ones((B, D), device=device, dtype=torch.long)

                if model.config.do_x1_sc:
                    x1_sc = model.config.mask_token_id * torch.ones_like(samples)

                t = 0.0
                while True:

                    model_input_samples = samples

                    if not model.config.do_x1_sc:
                        logits, _ = model(model_input_samples, t * torch.ones((B,), device=device)) # (B, T, V)
                    else:
                        logits = model._run_net(model_input_samples, t * torch.ones((B,), device=device), x1=x1_sc)

                    masked_logits = logits * (samples == mask_token_id).view(B, D, 1).float() + \
                        -1e9 * (samples != mask_token_id).view(B, D, 1).float()
                    max_masked_logits = torch.max(masked_logits, dim=-1)[0] # (B, T)
                    purity_weights = torch.softmax(max_masked_logits/purity_temp, dim=-1) # (B, T)

                    pt_x1_probs = F.softmax(logits / x1_temp, dim=-1) # (B, D, S)

                    if use_different_x1_sc_temp:
                        pt_sc_x1_probs = F.softmax(logits / x1_sc_temp, dim=-1) # (B, D, S)
                    else:
                        pt_sc_x1_probs = pt_x1_probs


                    if model.config.do_x1_sc:
                        x1_sc = torch.multinomial(pt_sc_x1_probs.view(B*D, S), num_samples=1).view(B, D).long()
                    if ignore_x1_sc:
                        x1_sc = model.config.mask_token_id * torch.ones_like(samples)

                    sample_is_mask = (samples == mask_token_id).view(B, D, 1).float()

                    # for when the current sample is a mask
                    step_probs = dt * pt_x1_probs * ((1+ noise*t) / ((1 - t))) # (B, D, S)
                    if do_purity_sampling:
                        step_probs = step_probs * sample_is_mask * purity_weights.view(B, D, 1) * torch.sum(sample_is_mask, dim=(1,2)).view(B, 1, 1)
                    else:
                        step_probs = step_probs * sample_is_mask

                    # when the current sample is not a mask
                    step_probs += dt * (1 - sample_is_mask) * mask_one_hot.view(1, 1, -1) * noise


                    step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
                    step_probs[
                        torch.arange(B, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(B),
                        samples.flatten()
                    ] = 0.0
                    step_probs[
                        torch.arange(B, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(B),
                        samples.flatten()
                    ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
                    step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
                    samples = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(B, D)

                    t += dt
                    if t > max_t:
                        break

                if argmax_final:
                    sample_is_mask = (samples == mask_token_id).view(B, D).float()
                    with torch.no_grad():
                        logits, _ = model(samples, t * torch.ones((B,), device=device)) # (B, T, V)
                    samples = torch.argmax(logits, dim=-1) * sample_is_mask + samples * (1 - sample_is_mask)

                samples_np = samples.cpu().detach().numpy() # (B, D)
            elif model_type == 'd3pm':

                samples = mask_token_id * torch.ones((batch_size, block_size), dtype=torch.int64, device=device)
                ts = np.arange(timesteps, 0, -1)
                B = batch_size
                D = block_size
                S = meta_vocab_size
                mask_one_hot = torch.zeros((S,), device=device)
                mask_one_hot[mask_token_id] = 1.0

                if model.config.do_x1_sc:
                    x1_sc = model.config.mask_token_id * torch.ones_like(samples)

                for t in ts:

                    if not model.config.do_x1_sc:
                        logits, _ = model(samples, t/timesteps * torch.ones((B,), device=device)) # (B, D, S)
                    else:
                        logits = model._run_net(samples, t/timesteps * torch.ones((B,), device=device), x1=x1_sc)


                    logits[:, :, mask_token_id] = -1e4
                    x0_probs = F.softmax(logits/x1_temp, dim=-1) # (B, D, S)

                    if model.config.do_x1_sc:
                        x1_sc = torch.multinomial(x0_probs.view(B*D, S), num_samples=1).view(B, D).long()

                    sample_is_mask = (samples == mask_token_id).view(B, D, 1).float()

                    step_probs = (1 / t) * x0_probs + (1 - 1/t) * mask_one_hot.view(1, 1, -1) # (B, D, S)
                    new_samples = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(B, D)

                    samples = samples * (1 - sample_is_mask[:, :, 0].long()) + new_samples * sample_is_mask[:, :, 0].long()
                samples_np = samples.cpu().detach().numpy()


            else:
                raise ValueError(f"unknown model type {model_type}")


            for sample_idx in range(samples_np.shape[0]):
                with open(os.path.join(samples_dir, 'samples.txt'), 'a') as f:
                    f.write(decode(samples_np[sample_idx]) + '\n')

with open(os.path.join(samples_dir, 'finished_sampling.txt'), 'w') as f:
    f.write('finished sampling\n')