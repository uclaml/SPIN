
set -x

input_dataset_path_or_name=${1:-"allenai/tulu-v2-sft-mixture"}
output_dir=${2:-"data"}
iter=${3:-0}
model_name_or_path=${4:-"allenai/tulu-2-70b"}

reformated_dataset_output=${output_dir}/SPIN_iter${iter}
generated_dataset_output=${output_dir}/generated/iter${iter}
training_dataset_output=${output_dir}/new_data/iter${iter}

# 0. reformat huggingface data 
python spin/reformat.py \
    --data $input_dataset_path_or_name \
    --output_dir $reformated_dataset_output

# 1. generate training data
bash scripts/tulu-70B/generate_vllm_batched.sh \
    $reformated_dataset_output \
    $generated_dataset_output \
    $model_name_or_path \
    8
    
# 2. gathering dataset
python spin/convert_data.py \
    --output_dir $training_dataset_output \
    --input_dir $generated_dataset_output \
    --num_fracs 11
