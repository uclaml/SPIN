export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 0
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 1
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 2
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 3
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 4
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 5
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 6
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 7
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 8
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 9
accelerate launch generate.py --model alignment-handbook/zephyr-7b-sft-full --frac_len 800 --data_frac 10
