from datasets import load_dataset
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='reformatted')
    parser.add_argument('--data', type=str, default='HuggingFaceH4/ultrachat_200k')
    return parser.parse_args()


def load_and_process_data_ultrachat(dataset_name, split):
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def load_and_process_data_tulu(dataset_name, input_split, test_split: float=0.1):
    try:
        dataset = load_dataset(dataset_name, split=input_split)
        dataset = dataset.train_test_split(test_size=test_split)
        reformatted_train_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["train"]]
        reformatted_test_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["test"]]
        return reformatted_train_data, reformatted_test_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []

def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")

def save_to_parquet(dataset, path):
    try:
        pq.write_table(dataset.data.table, path)
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")

def main():
    args = setup_arg_parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data == 'HuggingFaceH4/ultrachat_200k':
        train_data = load_and_process_data_ultrachat(args.data, 'train_sft')
        test_data = load_and_process_data_ultrachat(args.data, 'test_sft')
    elif "tulu-v2-sft-mixture" in args.data:
        train_data, test_data = load_and_process_data_tulu(args.data, 'train', test_split=0.1)
    else:
        raise ValueError(f"current {args.data} dataset is not supported")

    train_json_path = output_dir / 'train.json'
    test_json_path = output_dir / 'test.json'

    save_to_json(train_data, train_json_path)
    save_to_json(test_data, test_json_path)

    dataset = load_dataset('json', data_files=str(train_json_path), split='train')
    dataset_test = load_dataset('json', data_files=str(test_json_path), split='train')

    save_to_parquet(dataset, output_dir / 'train_prefs-00000-of-00001.parquet')
    save_to_parquet(dataset_test, output_dir / 'test_prefs-00000-of-00001.parquet')

    os.remove(train_json_path)
    os.remove(test_json_path)

if __name__ == "__main__":
    main()