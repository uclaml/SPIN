# Self-Play Fine-Tuning (SPIN)

This official repo holds code of the paper "[Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)".

Authors: [Zixiang Chen](https://sites.google.com/view/zxchen)\*, [Yihe Deng](https://sites.google.com/g.ucla.edu/yihedeng/)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ)\*, [Kaixuan Ji](https://scholar.google.com/citations?user=FOoKDukAAAAJ), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

<p align="center">
    <img src="images/iter_openllm.png" width="50%"> <br>
  Average score of <b>SPIN</b> at different iterations on the HuggingFace Open LLM leaderboard. 
</p>

## 🔔 News 
- **[01/02/2024]** Our paper is released on arXiv: https://arxiv.org/abs/2401.01335.

## Setup
Install the following Python dependencies to reproduce our results.

**[TODO]** Instructions for set-up.

### Data 
We provide the data used in our experiments along with the synthetic data we generated in this repo as well as on HuggingFace. These data is converted to .parquet format for fine-tuning (e.g. [iter0](data/iter0/train_prefs-00000-of-00001.parquet)).


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

#### Examples
```
bash scripts/generate.sh
``` 

### Step 2: Fine-tuning
```
accelerate launch --config_file configs/multi_gpu.yaml --num_processes=8 --main_process_port 29500 spin/run_spin.py configs/config.yaml
```
**[TODO]**: wrap up necessary codes into the folder spin. Add explainations/instructions here. 

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