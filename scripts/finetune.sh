export CUDA_VISIBLE_DEVICES="1,2,3,4"
export HF_HOME="/data/jkx/.cache"
# EDIT loss_type, output_dir in config.yaml
# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml --num_processes=4 --main_process_port 29500 spin/run_spin.py configs/config.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml --num_processes=4 --main_process_port 29500 spin/run_spin.py configs/config_lora.yaml
