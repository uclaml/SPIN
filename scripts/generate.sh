export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 0
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 1
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 2
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 3
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 4
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 5
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 6
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 7
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 8
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 9
accelerate launch --main_process_port=2950 spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 10

