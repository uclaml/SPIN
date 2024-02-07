import json
from datasets import load_dataset
import pandas as pd
import pyarrow.parquet as pq
import random
random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_fracs', type=int, default=125)
parser.add_argument('--output_dir', type=str, default='generated_iter0')

args = parser.parse_args()
num_fracs = args.num_fracs
output_dir = args.output_dir

data = []
for i in range(num_fracs):
    with open(f'data/{output_dir}/loser_{i}.jsonl', 'r') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        result['rejected'][1]['content'] = result['rejected'][1]['content'].lstrip()
        data.append(result)

print(len(data))
test_data = []

with open(f'data/{output_dir}/loser_0_test.jsonl', 'r') as f:
    json_list = list(f)
    print(len(json_list))

for json_str in json_list:
    result = json.loads(json_str)
    result['rejected'][1]['content'] = result['rejected'][1]['content'].lstrip()
    test_data.append(result)

with open('data/synthetic_train.json', 'w') as f:
    json.dump(data, f, indent=4)
with open('data/synthetic_test.json', 'w') as f:
    json.dump(test_data, f, indent=4)

dataset = load_dataset('json', data_files='data/synthetic_train.json',split='train')
dataset_test = load_dataset('json', data_files='data/synthetic_test.json',split='train')

print(len(dataset))
print(len(dataset_test))

# dataset.to_csv('train_prefs-00000-of-00000.csv')
pq.write_table(dataset.data.table, 'synthetic_ultra_50k_3112_iter2/train_prefs-00000-of-00001.parquet')
pq.write_table(dataset_test.data.table, 'synthetic_ultra_50k_3112_iter2/test_prefs-00000-of-00001.parquet')