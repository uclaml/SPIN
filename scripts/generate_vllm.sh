export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 0 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 1 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 2 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 3 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 4 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 5 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 6 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 7 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 8 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 9 --world_size 8 --output_dir generated/iter1
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 10 --world_size 8 --output_dir generated/iter1

# Generate for the test split as well
python3 spin/generate_vllm.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 0 --world_size 8 --output_dir generated/iter1
