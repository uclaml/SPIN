python3 spin/batched_generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --frac_len 5000 --num_data_frac 11 --tp_per_worker 1 --output_dir generated/iter0

# Generate for the test split as well
python3 spin/batched_generate_vllm.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --frac_len 5000 --num_data_frac 1 --tp_per_worker 1 --split test --output_dir generated/iter0
