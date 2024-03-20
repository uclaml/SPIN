# Tulu-70B Scripts Guide

This document provides an overview and usage guide for the scripts located in the `tulu-70B` folder within the SPIN project. These scripts are designed to facilitate fine-tuning the Tulu-70B model by SPIN.

## Scripts Overview

- `finetune.sh`: This script is used to launch a distributed training job for SPIN the Tulu-70B model. It utilizes the `accelerate` CLI tool for managing distributed training across multiple GPUs.

- `generate_training_data.sh.sh`: This script generates training data from the Tulu-70B model in a batched manner for SPIN.


## Usage

### Generate SPIN training data
To generate SPIN training data using the `generate_training_data.sh` script, follow the example below with detailed parameter explanations:
```bash
# from the root directory of SPIN

bash SPIN/scripts/tulu-70B/generate_training_data.sh.sh <DATA_DIR> <OUTPUT_DIR> <SPIN_ITER> <MODEL_PATH>
```

The `generate_training_data.sh` script requires the following parameters:
- `<DATA_DIR>`: The directory where your raw data is stored. This should be the SFT dataset you want to train with SPIN.
- `<OUTPUT_DIR>`: The directory where the generated training data will be saved. 
- `<SPIN_ITER>`: The iteration mark of SPIN algorithm 
- `<MODEL_PATH>`: The model name or path

### Fine-tuning

```bash
# from the root directory of SPIN

bash SPIN/scripts/tulu-70B/finetune.sh <TRAIN_EPOCH> <WORKER_NUM> <WORKER_RANK> <WORKER_NUM_GPU> <WORKER_0_PORT> <WORKER_0_HOST> <TRAIN_BATCH_SIZE> <EVAL_BATCH_SIZE>
```

To fine-tune the Tulu-70B model, use the `finetune.sh` script with the appropriate parameters. For example:
- `TRAIN_EPOCH`: Number of training epochs. Default is 5.
- `WORKER_NUM`: Number of worker machines. Default is 1.
- `WORKER_RANK`: Rank of the current worker. Default is 0.
- `WORKER_NUM_GPU`: Number of GPUs per worker. Default is 8.
- `WORKER_0_PORT`: Port of the main worker (rank 0). Default is 2950.
- `WORKER_0_HOST`: Host IP of the main worker (rank 0). Default is "127.0.0.1".
- `TRAIN_BATCH_SIZE`: Training batch size per device. Default is 1.
- `EVAL_BATCH_SIZE`: Evaluation batch size per device. Default is 1.
















