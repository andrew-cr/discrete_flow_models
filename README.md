
# Discrete Flow Models

Repository for notebooks and the text experiment from the paper https://arxiv.org/abs/2402.04997

Code for the protein co-design experiments can be found at https://github.com/jasonkyuyim/multiflow

This repository is built on top of https://github.com/karpathy/nanoGPT


## Install

Package requirements are listed in `environment.yml` and a conda environment can
be installed from this file e.g.

```bash
conda env create --file environment.yml
```

## Notebooks

To get started playing around with toy discrete flow models, we have included
some notebooks that contain masking, uniform and the general forms as described
in the implementation details section of the paper.

## Sampling a Pre-Trained Model

We provide our pre-trained text8 model at `https://www.dropbox.com/scl/fi/rno9fq8mpjs2bdctz7o53/dfm.pt?rlkey=1ge1wxv14b4a46b730hbltwkg&dl=0`

To generate samples with this model first update the config file `config/sample_text8.py`.
- Set the `out_dir` to a directory where samples will be saved.
- Set `ckpt_path` to point to where the pre-trained `.pt` model is.
- All other settings can be left at their default values.

Then run the following command to generate samples:

```bash
python sample.py config/sample_text8.py
```

We have provided a script that can re-create the logit temperature sweep from the paper.
Within the `scripts/generate_samples.sh`, the arguments to the `sample_eval.py` script should be modified.
- Set the `--path` argument to where the same directory as `out_dir` in `config/sample_text8.py`.
- Set `--cache_dir` to a path to where you would like the GPT-J-6B model to be downloaded to.

Then run the following command to generate samples:

```bash
bash scripts/generate_samples.sh
```
This will save an NLL file and entropy file in each sample folder which can then be coallated to
form the logit temperature sweep.


## Training

### Downloading the text8 dataset
First we download the text8 data. Set the `DATA_DIR` variable within the `data/text8/download.sh` script
to the location of this repository's `data/text8` directory. Then run

```bash
bash data/text8/download.sh
```

Then we pre-process the downloaded data.
```bash
python data/text8/prepare.py
```

### Running training
First update the config file `config/train_text8.py`. Set the `out_dir` to a directory where a folder can be created 
to store the model checkpoints.
Then to run on 4GPUs run the following command:

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_text8.py
```
