python3 spin/batched_generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --num_data_frac 11 --tp_per_worker 1 --output_dir generated/iter1

# Generate for the test split as well
python3 spin/batched_generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --num_data_frac 1 --tp_per_worker 1 --split test --output_dir generated/iter1
