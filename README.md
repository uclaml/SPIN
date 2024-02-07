<p align="center">
    <img src="images/spin_dalle.png" width="30%"> <br>
</p>
<p align="center">
    🤗 <a href="https://huggingface.co/collections/UCLA-AGI/zephyr-7b-sft-full-spin-65c361dfca65637272a02c40" target="_blank">Models</a> | 🤗 <a href="https://huggingface.co/collections/UCLA-AGI/datasets-spin-65c3624e98d4b589bbc76f3a" target="_blank">Datasets</a>
</p>

# Self-Play Fine-Tuning (SPIN)

This official repo holds code of the paper "[Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)".

Authors: [Zixiang Chen](https://sites.google.com/view/zxchen)\*, [Yihe Deng](https://sites.google.com/g.ucla.edu/yihedeng/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

<p align="center">
    <img src="images/iter_openllm.png" width="50%"> <br>
  Average score of <b>SPIN</b> at different iterations on the HuggingFace Open LLM leaderboard. 
</p>

## 🔔 News 
- **[01/02/2024]** Our paper is released on arXiv: https://arxiv.org/abs/2401.01335.


## Table of Contents
- [Setup](#Setup)
    - [Data](#Data)
- [Usage](#Usage)
    - [Step 1: Generation](#step-1-generation)
    - [Step 1.5: Gather generations and convert data type](#step-15-gather-generations-and-convert-data-type)
    - [Step 2: Fine-tuning](#step-2-fine-tuning)
- [Citation](#Citation)

## Setup
The following steps provide the necessary setup to run our codes.
1. Create a Python virtual environment with Conda:
```
conda create -n myenv python=3.10
conda activate myenv
```
2. Install the following Python dependencies to run the codes.
```
python -m pip install .
python -m pip install flash-attn --no-build-isolation
```

### Data 
We provide the data used in our experiments along with the synthetic data we generated in this repo as well as on HuggingFace. These data is converted to .parquet format for fine-tuning (e.g. [iter0](data/iter0), [iter1](data/iter1), [iter2](data/iter2), [iter3](data/iter3)). 

🔍Note: With the provided data, one can directly jump to [Step 2: Fine-tuning](#step-2-fine-tuning) without doing generation on their own. You may also start from any iteration to reproduce our results using our open-sourced checkpoints.


## Usage
### Step 1: Generation
```
accelerate launch spin/generate.py [options]
```
Options
- `--model`: load model checkpoint for generation.
    - default: `alignment-handbook/zephyr-7b-sft-full`
- `--input_dir`: directory to the data files with prompts for generation
    - The code is for generation based on old data. 
    - default: `synthetic_ultra_14k`
- `--output_dir`: directory to save the output data. 
- `--batch_size`: per device batch size
    - default: 16
- `--data_frac`: break data into fractions for generations across server.
    - `--frac_len`: length of the data fraction. Default is 0 which uses the entire dataset for generation. Set `frac_len` to a positive number to generate only for a fraction of data.  
    - Setting `data_frac` to be 0, 1, 2... to generate for different fractions of length `frac_len`.
    - Note: maintain the same frac length when doing generation using data_frac. It's recommended to set a smaller `frac_len` to 800.

The generated data is in json format where each data contains the following attributes:
```
{
    "chosen": [{"role": "user", "content": <prompt>}, 
               {"role": "assistant", "content": <ground truth>}],
    "rejected": [{"role": "user", "content": <prompt>}, 
                 {"role": "assistant", "content": <generation>}]
}
```

#### Example
The following code generates 8k synthetic data for iteration 1.
```
bash scripts/generate.sh
``` 

### Step 1.5: Gather generations and convert data type
```
python spin/convert_data.py [options]
```
Options
- `--num_fracs`: number of files to load in.
- `--output_dir`: directory to the data files

### Step 2: Fine-tuning
```
accelerate launch --config_file configs/multi_gpu.yaml --num_processes=8 --main_process_port 29500 spin/run_spin.py configs/config.yaml
```
<!-- **[TODO]**: wrap up necessary codes into the folder spin. Add explainations/instructions here.  -->

You might need to change the configuration in `configs/config.yaml`. Here are some key configs you might need to customize:

- `--model_name_or_path`: load model checkpoint for finetuning.
    - default: `alignment-handbook/zephyr-7b-sft-full`
- `--output_dir`: the output directory of finetuned model and checkpoints 
    - default: `outputs`
- `--output_dir`: directory to save the output data. 
- `per_device_train_batch_size`: batch size on one GPU
    - default: 16
- `gradient_accumulation_steps`: make sure that per_device_train_batch_size\*num_processes\*gradient_accumulation_steps equals to your true batch size.
- `num_train_epochs`: the training epochs of this iteration
    - default: 3
- `beta`: beta in SPIN
    - default: 0.1


#### Examples
```
bash scripts/finetune.sh
```

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{chen2024selfplay,
      title={Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models}, 
      author={Zixiang Chen and Yihe Deng and Huizhuo Yuan and Kaixuan Ji and Quanquan Gu},
      year={2024},
      eprint={2401.01335},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgement
This repo is built upon [The Alignment Handbook](https://github.com/huggingface/alignment-handbook). We thank the authors for their great work.