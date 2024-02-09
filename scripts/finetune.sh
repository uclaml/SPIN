# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Set the home directory for Hugging Face transformers library cache.
#export HF_HOME="${your_hf_home}"

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

# Launch a distributed training job using the `accelerate` CLI tool.
# --config_file: Specifies the configuration file for DeepSpeed, which includes distributed training options and optimizations.
# --num_processes: The number of separate processes to launch, typically matches the number of available GPUs for parallel training.
# Other zones to specify that can override settings in config.yaml:
# --num_train_epochs=6 --loss_type="corr_clip" --output_dir="${path_to_save_checkpoint}" #\
# The script to run (spin/run_spin.py) along with its configuration file (configs/config.yaml).
accelerate launch --config_file configs/deepspeed_zero3.yaml --num_processes=8 --main_process_port 2950 spin/run_spin.py configs/config.yaml
