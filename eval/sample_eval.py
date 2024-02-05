from transformers import GPTJForCausalLM, AutoTokenizer
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import yaml

import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path to network folder that contains the sample folders")
parser.add_argument("--cache_dir", help="path to cache directory for GPTJ")
parser.add_argument("--data_dir", help="path to data directory containing meta.pkl")

# Parse the command-line arguments
args = parser.parse_args()

device = "cuda"

print("loading GPTJ model")

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6b",
    revision="float16",
    torch_dtype=torch.float16,
    cache_dir=args.cache_dir
).to(device)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")

data_dir = args.data_dir

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# increase vocab size by 1 to include a mask token
meta_vocab_size += 1
mask_token_id = meta_vocab_size - 1

stoi = meta['stoi']
itos = meta['itos']
stoi['X'] = mask_token_id
itos[mask_token_id] = 'X'

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

block_size = 256
batch_size = 1

network_path = args.path
sample_folders = glob(os.path.join(network_path, 'samples_*'))
for sample_folder in sample_folders:
    # check if status.txt exists
    if os.path.exists(os.path.join(sample_folder, 'status.txt')):
        # read status
        # with open(os.path.join(sample_folder, 'status.txt'), 'r') as f:
        #     status = f.read()
        # if status == 'done' or status == 'in progress':
        #     continue
        print("skipping samples at ", sample_folder, " because status.txt exists")
        continue
    elif not os.path.exists(os.path.join(sample_folder, 'finished_sampling.txt')):
        print("skiping samples at ", sample_folder, " because finished_sampling.txt does not exist")
        continue
    else:
        # create status.txt
        with open(os.path.join(sample_folder, 'status.txt'), 'w') as f:
            f.write('in progress')
        print("evaluating samples at ", sample_folder)

    # load the config.yaml file
    with open(os.path.join(sample_folder, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)


    # read samples.txt
    with open(os.path.join(sample_folder, 'samples.txt'), 'r') as f:
        lines = f.readlines()
        samples = [line.strip() for line in lines]

    # ------------ NLL -----------------
    losses = []
    counter = 0

    for sample in samples:

        input_ids = tokenizer(sample, return_tensors="pt").input_ids.to(device)
        output = model.forward(
            input_ids,
            return_dict=True,
            labels=input_ids,
        )
        losses.append(output.loss.item())

    # save np array losses in the sample folder
    np.savetxt(os.path.join(sample_folder, 'losses.txt'), np.array(losses))
    np.savetxt(os.path.join(sample_folder, 'loss_mean.txt'), np.array([np.mean(losses)]))

    print("nll", np.mean(losses))

    # -------------- Unigram ---------------
    sample_stats = {}
    for sample in samples:
        tokens = tokenizer(sample, return_tensors="pt").input_ids[0]
        for token in tokens:
            if token.item() not in sample_stats:
                sample_stats[token.item()] = 0
            sample_stats[token.item()] += 1

    with open(os.path.join(sample_folder, 'unigram_stats.pkl'), 'wb') as f:
        pickle.dump(sample_stats, f)

    sample_probs = {}
    total_sample_stats = sum(sample_stats.values())
    sample_entropy = 0
    for k, v in sample_stats.items():
        p = v / total_sample_stats
        sample_probs[k] = p
        if p > 0:
            sample_entropy += -p * np.log(p)
    print("sample entropy: ", sample_entropy)

    # save the dataset entropy and sample entropy in a text file
    with open(os.path.join(sample_folder, 'entropy.txt'), 'w') as f:
        f.write("sample entropy: " + str(sample_entropy) + "\n")

    # write done in status.txt
    with open(os.path.join(sample_folder, 'status.txt'), 'w') as f:
        f.write('done')


