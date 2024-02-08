export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 0
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 1
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 2
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 3
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 4
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 5
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 6
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 7
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 8
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 9
accelerate launch spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir data/iter0 --frac_len 800 --data_frac 10

