
INPUT_DIR=${1:-"UCLA-AGI/SPIN_iter0"}
OUTPUT_DIR=${2:-"generated/iter0"}
MODEL_NAME_OR_PATH=${3:-"allenai/tulu-2-70b"}
TP_PER_WORKER=${4:-8}


python3 spin/batched_generate_vllm.py \
    --model $MODEL_NAME_OR_PATH \
    --input_dir $INPUT_DIR \
    --frac_len 5000 \
    --num_data_frac 11 \
    --tp_per_worker $TP_PER_WORKER \
    --output_dir $OUTPUT_DIR

# Generate for the test split as well
python3 spin/batched_generate_vllm.py \
    --model $MODEL_NAME_OR_PATH \
    --input_dir $INPUT_DIR \
    --frac_len 5000 \
    --num_data_frac 1 \
    --tp_per_worker $TP_PER_WORKER \
    --split test \
    --output_dir $OUTPUT_DIR
