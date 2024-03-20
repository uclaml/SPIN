import argparse
import json
import multiprocessing as mp
import os
import time
import ray
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to generate text using vLLM")

    parser.add_argument(
        "--model", type=str, default="UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0"
    )
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument(
        "--num_data_frac", type=int, default=0, help="Number of Data fraction"
    )
    parser.add_argument(
        "--tp_per_worker",
        type=int,
        default=1,
        help="Number of GPUs to be used per Worker",
    )
    parser.add_argument("--input_dir", type=str, default="UCLA-AGI/SPIN_iter0")
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


# NOTE: `gpu_queue, task_queue` needs to be at the end (others handled by partial)
def run_process_on_gpu(
    model_path, input_dir, frac_len, world_size, output_dir, split, gpu_queue, task_queue
):
    gpu_ids = [gpu_queue.get() for _ in range(world_size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu_id) for gpu_id in gpu_ids])
    print(f"Running on GPU: {gpu_ids}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
    )
    # Assuming the existence of a function that handles the generation process for a single GPU
    while not task_queue.empty():
        data_frac = task_queue.get()
        generate_on_single_gpu(
            model_path, input_dir, frac_len, data_frac, world_size, output_dir, split, llm
        )
        task_queue.task_done()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)
    ray.shutdown()


def generate_on_single_gpu(
    model_path, input_dir, frac_len, data_frac, world_size, output_dir, split, llm
):
    # TODO: the generation can be decoupled to use async engine and multiple clients
    # to accelerate, which will amortize the loading time
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating on GPU with data fraction {data_frac}...")
    # load a base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256)

    # load data
    data = load_dataset(input_dir, split=split)
    data = data.shuffle(seed=42)
    if frac_len > 0:
        sub_len = frac_len
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len * data_frac :]["real"]
        else:
            data = data[sub_len * data_frac : sub_len * (data_frac + 1)]["real"]

    prompts_all = [
        "### Instruction: " + data[idx][0]["content"] + "\n\n### Response: "
        for idx in range(len(data))
    ]
    prompts_old = [data[idx][0]["content"] for idx in range(len(data))]
    corrects_all = [data[idx][1]["content"] for idx in range(len(data))]

    start = time.time()

    # run vllm
    results_gathered = list(
        map(lambda x: x.outputs[0].text, llm.generate(prompts_all, sampling_params))
    )

    results = [r.replace("</s>", "").lstrip() for r in results_gathered]

    timediff = time.time() - start
    print(f"time elapsed: {timediff:.2f}s")

    # collecting data
    for idx in range(len(corrects_all)):
        d = {
            "real": [
                {"role": "user", "content": prompts_old[idx]},
                {"role": "assistant", "content": corrects_all[idx]},
            ],
            "generated": [
                {"role": "user", "content": prompts_old[idx]},
                {"role": "assistant", "content": results[idx]},
            ],
        }
        if split == "test":
            filename = f"{output_dir}/loser_{data_frac}_test.jsonl"
        else:
            filename = f"{output_dir}/loser_{data_frac}.jsonl"
        with open(filename, "a") as f:
            json.dump(d, f)
            f.write("\n")


def main():
    start = time.time()
    mp.set_start_method("spawn", force=True)
    args = parse_arguments()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Check if the model is already downloaded
    model_path = args.model

    if not model_path.startswith("/"):  # hub path
        filepath = try_to_load_from_cache(model_path, "config.json")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_directory = cache_dir / f"models--{model_path.replace('/', '--')}"

        print(f"checking cache results: {filepath}")
        if isinstance(filepath, str):
            print(f"Model {model_path} is already downloaded.")
        else:
            print(f"Model {model_path} is not downloaded, will be downloaded now")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"Model {model_path} downloaded.")
            del tokenizer
            del model

    else:  # local path
        model_directory = model_path
    print(f"model directory: {model_directory}")

    # Create a pool of processes. Each process will run on a separate GPU.
    with mp.Manager() as manager:
        gpu_queue = manager.Queue()  # Create a Manager Queue
        task_queue = manager.Queue()  # Create a Task Queue
        # Add gpu_id to the queue
        for i in range(num_gpus):
            gpu_queue.put(i)
        
        # add fraction task into task queue
        for i in range(args.num_data_frac):
            task_queue.put(i)

        num_workers = num_gpus//args.tp_per_worker
        with mp.Pool(processes=num_workers) as pool:
            # Partial function with all arguments except the one that changes per process (data_frac)
            func = partial(
                run_process_on_gpu,
                args.model,
                args.input_dir,
                args.frac_len,
                args.tp_per_worker,
                args.output_dir,
                args.split,
            )

            # for each data_frac, scheduling one task
            res_futs = []
            for _ in range(num_workers):
                res_futs.append(
                    pool.apply_async(
                        func,
                        (
                            gpu_queue,
                            task_queue,
                        ),
                    )
                )

            for res in res_futs:
                res.get()
    print(f"finished generating in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
