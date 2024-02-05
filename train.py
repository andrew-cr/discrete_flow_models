"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from pathlib import Path

from flow_model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# these values will be overridden by the config file so their values here don't matter.
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

wandb_id = 'blank'
is_repeat = False

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
overfit_batch = False
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
qk_layernorm = False
proper_timestep_emb = False
do_x1_sc = False
x1_sc_prob = 0.5

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

data_dir = 'data/text8' #  directory should contain train.bin, val.bin, meta.pkl
warm_start_ckpt = None
resume_dir = None

model_type = 'flow' # flow, d3pm

d3pm_loss_weighting = False
d3pm_loss_weighting_maxT = 1000
timesteps = 1000

min_t = 0.0

bonus_seed_offset = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and (isinstance(v, (int, float, bool, str)) or v is None) ]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

assert model_type in ['flow', 'd3pm']

if resume_dir is None:
    if wandb_id == 'blank':
        out_dir = os.path.join(out_dir, time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + wandb_run_name)
    else:
        out_dir = os.path.join(out_dir, str(wandb_id) + '_' + wandb_run_name)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

else:
    out_dir = resume_dir

assert (resume_dir is not None) == is_repeat


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    print("ddp run")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    print("not ddp run")
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

shared_generator = torch.Generator(device).manual_seed(42) # for use when we want the random numbers to be the same across processes

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process and resume_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

torch.manual_seed(1337 + seed_offset + bonus_seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# attempt to derive vocab_size from the dataset
# data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
assert os.path.exists(meta_path)
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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, qk_layernorm=qk_layernorm,
                  do_x1_sc=do_x1_sc, mask_token_id=mask_token_id, proper_timestep_emb=proper_timestep_emb,
                  d3pm_loss_weighting=d3pm_loss_weighting, d3pm_loss_weighting_maxT=d3pm_loss_weighting_maxT)
if init_from == 'scratch' and not is_repeat:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume' or is_repeat:

    # print(f"Resuming training from {out_dir}")
    assert wandb_id != 'blank'
    init_from = 'resume'
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'current_ckpt.pt')
    print(f"resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    print(f"loaded checkpoint model args {checkpoint_model_args}")

    # override some values
    checkpoint_model_args['vocab_size'] = meta_vocab_size

    print(f"overrided checkpoint model args {checkpoint_model_args}")

    # create the model
    gptconf = GPTConfig(**checkpoint_model_args)
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
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
elif init_from == 'warm_start':
    print(f"warm starting from checkpoint {warm_start_ckpt}")
    checkpoint = torch.load(warm_start_ckpt, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # if warm starting from a non-SC model, will need to init the SC params
    if 'xt_x1_proj.weight' in model.state_dict():
        if 'xt_x1_proj.weight' not in state_dict.keys():
            print("SC params not in warmstart ckpt so initializing them from scratch")
            state_dict['xt_x1_proj.weight'] = model.state_dict()['xt_x1_proj.weight']
            if 'xt_x1_proj.bias' in model.state_dict():
                state_dict['xt_x1_proj.bias'] = model.state_dict()['xt_x1_proj.bias']

    model.load_state_dict(state_dict)
else:
    raise ValueError(f"Unknown init_from value {init_from}")

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    # don't do this if warmstart
    print("loading optimizer state from checkpoint")
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    if do_x1_sc:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    else:
        model = DDP(model, device_ids=[ddp_local_rank])


def corrupt_data(data, times):
    b = times.shape[0]
    t = data.shape[1]
    assert times.shape == (b,)
    assert data.shape == (b, t)
    if model_type == 'flow':
        u = torch.rand((batch_size, block_size), device=times.device)
        target_mask = u < (1.0 - times.view(batch_size, 1))
        data[target_mask] = mask_token_id

        return data, target_mask
    elif model_type == 'd3pm':
        u = torch.rand((batch_size, block_size), device=times.device)
        target_mask = u < (times.view(batch_size, 1).float() / timesteps)
        data[target_mask] = mask_token_id

        return data, target_mask
    else:
        raise ValueError(f'Unknown model type {model_type}')

# poor man's data loader
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split, times=None):
    data = train_data if split == 'train' else val_data
    if not overfit_batch:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        ix = torch.zeros((batch_size,), dtype=torch.int64)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if times is None:
        if model_type == 'flow':
            times = torch.rand((batch_size,)) * (1.0 - min_t) + min_t
        elif model_type == 'd3pm':
            times = torch.randint(low=1, high=timesteps+1, size=(batch_size,)).float()
        else:
            raise ValueError(f'Unknown model type {model_type}')
    else:
        assert times.shape == (batch_size,)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, times = x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True), \
            times.pin_memory().to(device, non_blocking=True)
    else:
        x, y, times = x.to(device), y.to(device), times.to(device)
    return x, y, times


def calc_loss(X, Y, times, target_mask, infill_probs, num_ones_in_mask):
    if model_type == 'flow':
        logits, loss = model(X, times, Y, target_mask)
    elif model_type == 'd3pm':
        logits, loss = model(X, times.float()/timesteps, Y, target_mask)
    else:
        raise ValueError(f'Unknown model type {model_type}')

    return loss






# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, times = get_batch(split)
            if model_type == 'flow':
                X, target_mask = corrupt_data(X, times)
                with ctx:
                    loss = calc_loss(X, Y, times, target_mask, None, None)
            elif model_type == 'd3pm':
                X, target_mask = corrupt_data(X, times)
                with ctx:
                    loss = calc_loss(X, Y, times, target_mask, None, None)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    if wandb_id == 'blank' :
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=None,
            resume=is_repeat)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id=wandb_id,
            resume=is_repeat)


# training loop
X, Y, times = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:


            try:
                if model_type == 'flow':
                    times = 0.85 * torch.ones((batch_size,))
                    X, Y, times = get_batch('train', times)
                    X, target_mask = corrupt_data(X, times)
                    with torch.no_grad():
                        logits, _ = model(X, times) # (B, T, V)
                elif model_type == 'd3pm':
                    times = (0.15 * torch.ones((batch_size,)) * timesteps).long().float()
                    X, Y, times = get_batch('train', times)
                    X, target_mask = corrupt_data(X, times)
                    with torch.no_grad():
                        logits, _ = model(X, times.float()/timesteps) # (B, T, V)
                else:
                    raise ValueError(f'Unknown model type {model_type}')

                predictions = torch.argmax(logits, dim=-1)
                samples = torch.multinomial(torch.softmax(logits, dim=-1).view(-1, meta_vocab_size), num_samples=1)[:, 0].view(batch_size, -1)

                # calculate accuracy
                matches = (samples == Y) # (B, T)
                acc = (matches * target_mask).sum().float() / target_mask.sum()

                first_pred_idx = torch.argmax(target_mask[0].float())
                zero_logit = logits[0, first_pred_idx, 0]
                one_logit = logits[0, first_pred_idx, 1]
                two_logit = logits[0, first_pred_idx, 2]
                predictions[~target_mask] = X[~target_mask]
                samples[~target_mask] = X[~target_mask]
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "clean" : decode(Y[0].cpu().numpy()),
                    "corrupted" : decode(X[0].cpu().numpy()),
                    "argmax_recon" : decode(predictions[0].cpu().numpy()),
                    "sample_recon" : decode(samples[0].cpu().numpy()),
                    "zero_logit": zero_logit,
                    "one_logit": one_logit,
                    "two_logit": two_logit,
                    "acc": acc,
                }, step=iter_num)
            except Exception as e:
                print(f"logging failed: {e}")

        def save_checkpoint(file_path):
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                # print(f"saving checkpoint to {out_dir}")
                # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"saving checkpoint to {file_path}")
                torch.save(checkpoint, file_path)

        save_checkpoint(os.path.join(out_dir, 'current_ckpt.pt'))

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            save_checkpoint(os.path.join(out_dir, 'best_ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    if do_x1_sc and torch.rand(1, generator=shared_generator, device=device) < x1_sc_prob:
        do_self_conf_loop = True
    else:
        do_self_conf_loop = False

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if model_type == 'flow':
            X, target_mask = corrupt_data(X, times)
        elif model_type == 'd3pm':
            X, target_mask = corrupt_data(X, times)
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if model_type == 'flow':
                logits, loss = model(X, times, Y, target_mask, do_self_conf_loop=do_self_conf_loop)
            elif model_type == 'd3pm':
                logits, loss = model(X, times.float()/timesteps, Y, target_mask, do_self_conf_loop=do_self_conf_loop)
            else:
                raise ValueError(f'Unknown model type {model_type}')

            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y, times = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        try:
            wandb.log({"train/iter_loss": lossf}, step=iter_num)
        except Exception as e:
            print(e)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
