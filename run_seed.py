import os

config_path = "configs\\transfer.yaml"
seed = 20

epochs = 5500
log_interval = 35
early_stop = 0
start_by = 14
session = 1

batch_size = 96

tmp_saved_path = "D:\\Users\\86189\\Desktop\\0528-MaxUpLLLearning\\DG"
command = (
    f"python meta_main.py "
    f"--seed {seed} "
    f"--config {config_path} "
    f"--n_epochs {epochs} "
    f"--log_interval {log_interval} "
    f"--early_stop {early_stop} "
    f"--tmp_saved_path {tmp_saved_path} "
    f"--session {session} "
    f"--saved_model False "
    f"--n_inner_steps 1 "
    f"--inner_lr 1e-3 "
)
os.system(command)

