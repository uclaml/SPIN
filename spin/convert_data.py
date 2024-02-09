import json
import os
from datasets import load_dataset
import pyarrow.parquet as pq
import random
random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_fracs', type=int, default=125)
parser.add_argument('--input_dir', type=str, default='generated/iter0')
parser.add_argument('--output_dir', type=str, default='synthetic')

args = parser.parse_args()
num_fracs = args.num_fracs
input_dir = args.input_dir
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data = []
for i in range(num_fracs):
    with open(f'{input_dir}/loser_{i}.jsonl', 'r') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        result['generated'][1]['content'] = result['generated'][1]['content'].lstrip()
        data.append(result)

print(len(data))
test_data = []

with open(f'{input_dir}/loser_0_test.jsonl', 'r') as f:
    json_list = list(f)
    print(len(json_list))

for json_str in json_list:
    result = json.loads(json_str)
    result['generated'][1]['content'] = result['generated'][1]['content'].lstrip()
    test_data.append(result)

with open(f'{input_dir}/synthetic_train.json', 'w') as f:
    json.dump(data, f, indent=4)
with open(f'{input_dir}/synthetic_test.json', 'w') as f:
    json.dump(test_data, f, indent=4)

dataset = load_dataset('json', data_files=f'{input_dir}/synthetic_train.json',split='train')
dataset_test = load_dataset('json', data_files=f'{input_dir}/synthetic_test.json',split='train')

print(len(dataset))
print(len(dataset_test))

pq.write_table(dataset.data.table, f'{output_dir}/train_prefs-00000-of-00001.parquet')
pq.write_table(dataset_test.data.table, f'{output_dir}/test_prefs-00000-of-00001.parquet')