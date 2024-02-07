export CUDA_VISIBLE_DEVICES=0,1,2,3

# EDIT loss_type, output_dir in config.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml --num_processes=8 --main_process_port 29500 scripts/run_spin.py configs/config.yaml