set -x
# Set the home directory for Hugging Face transformers library cache.
#export HF_HOME="${your_hf_home}"

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

TRAIN_EPOCH=${1:-5}

WORKER_NUM=${2:-1}
WORKER_RANK=${3:-0}
WORKER_NUM_GPU=${4:-8}
WORKER_0_PORT=${5:-2950}
WORKER_0_HOST=${6:-"127.0.0.1"}

TRAIN_BATCH_SIZE=${7:-2}
EVAL_BATCH_SIZE=${8:-1}

# Launches a distributed training job with the `accelerate` CLI tool. Key parameters include:
# --config_file: Path to the DeepSpeed configuration file. This file defines distributed training options and optimizations.
# --num_processes: Sets the number of processes to launch, typically equal to the number of GPUs available for parallel training.
# Additional override options (specified at command line) that can alter settings defined in config.yaml:
# --num_train_epochs=6: Specifies the total number of training epochs.
# --learning_rate=1e-7: Sets the learning rate for the training process.
# --beta=0.1: Custom beta parameter value.
# --warmup_ratio=0.1: Defines the warmup ratio for learning rate scheduling.
# --output_dir="${path_to_save_checkpoint}": Directory where training checkpoints will be saved.
# Execution command: Runs 'spin/run_spin.py' with 'configs/config.yaml' as its configuration.

accelerate launch \
    --config_file configs/tulu-70B/deepspeed_zero3.yaml \
    --num_machines $WORKER_NUM\
    --machine_rank $WORKER_RANK\
    --num_processes $WORKER_NUM_GPU \
    --main_process_port $WORKER_0_PORT \
    --main_process_ip $WORKER_0_HOST\
    spin/run_spin.py configs/tulu-70B/config.yaml \
    --num_train_epochs=$TRAIN_EPOCH \
    --per_device_train_batch_size=$TRAIN_BATCH_SIZE \
