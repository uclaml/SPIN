export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 0 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 1 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 2 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 3 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 4 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 5 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 6 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 7 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 8 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 9 --batch_size 8 --output_dir generated/iter1
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 10 --batch_size 8 --output_dir generated/iter1

# Generate for the test split as well
python3 ../spin/generate.py --model UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0 --input_dir UCLA-AGI/SPIN_iter0 --frac_len 800 --data_frac 0 --batch_size 8 --output_dir generated/iter1