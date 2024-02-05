out_dir = 'path/to/out_dir'
eval_interval = 2000
eval_iters = 200
log_interval = 1

data_dir = 'data/text8'

warm_start_ckpt = None
init_from = 'scratch'
resume_dir = None

always_save_checkpoint = False

wandb_log = True
wandb_project = 'text8'
wandb_run_name = 'D3PM'
wandb_id = 'blank'
is_repeat = False

dataset = 'text8'
gradient_accumulation_steps = 8
batch_size = 256
block_size = 256
overfit_batch = False

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
qk_layernorm = True
do_x1_sc = True
x1_sc_prob = 0.5
do_d3pm_loss_weighting = False
proper_timestep_emb = False

model_type = 'd3pm'

learning_rate = 1e-4
max_iters = 750000
lr_decay_iters = 1000000
min_lr = 1e-5
beta2 = 0.99

warmup_iters = 1000

timesteps = 1000
