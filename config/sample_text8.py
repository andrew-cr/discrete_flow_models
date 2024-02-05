
out_dir = 'path/to/outdir'
ckpt_path = 'path/to/ckpt_pt'
data_dir = 'data/text8'

run_name = 'base'

dataset = 'text8'
batch_size = 256
block_size = 256 # context of up to 256 previous characters

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
qk_layernorm = True
do_x1_sc = True

total_samples = 512
dt = 0.001
max_t = 0.98
argmax_final = True
noise = 15.0
x1_temp = 1.0
use_different_x1_sc_temp = False
x1_sc_temp = 1.0
ignore_x1_sc = False

do_purity_sampling = False
purity_temp = 1.0

model_type = 'flow'