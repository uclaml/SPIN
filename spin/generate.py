from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json
import argparse
from datasets import load_dataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0')
parser.add_argument('--data_frac', type=int, default=0)
parser.add_argument('--frac_len', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='generated/iter1')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--input_dir', type=str, default='data/iter0')

args = parser.parse_args()
model_path = args.model
data_frac = args.data_frac
batch_size = args.batch_size

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# load a base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   
tokenizer.pad_token = tokenizer.eos_token

# load data
data = load_dataset(args.input_dir, split='train')
data = data.shuffle(seed=42)
if args.frac_len > 0:
    sub_len = args.frac_len 
    if sub_len*(data_frac+1) > len(data):
        data = data[sub_len*data_frac:]['chosen']
    else:
        data = data[sub_len*data_frac:sub_len*(data_frac+1)]['chosen']

prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
prompts_old = [data[idx][0]['content'] for idx in range(len(data))]
corrects_all = [data[idx][1]['content'] for idx in range(len(data))]

# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(prompts_all) as prompts:
    results = []

    # have each GPU do inference in batches
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=args.batch_size)

    for prompts_tokenized in tqdm(prompt_batches):
        # set max_new_tokens smaller for faster inference
        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

        # remove prompt from gen. tokens
        outputs_tokenized=[ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
        # decode gen. tokens 
        outputs=tokenizer.batch_decode(outputs_tokenized)
        results.extend(outputs)

# collect results from all the GPUs and remove paddings
results_gathered=gather_object(results)
results = [r.replace("</s>","").lstrip() for r in results_gathered]

if accelerator.is_local_main_process:
    timediff=time.time()-start
    print(f"time elapsed: {timediff}")

    # collecting data
    for idx in range(len(corrects_all)):
        d = {"chosen": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": corrects_all[idx]}], "rejected": [{"role": "user", "content": prompts_old[idx]}, {"role": "assistant", "content": results[idx]}]}
        with open(f"{args.output_dir}/loser_{data_frac}.jsonl", 'a') as f:
            json.dump(d, f)
            f.write('\n')