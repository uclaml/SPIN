# Adapted from https://github.com/huggingface/alignment-handbook 

import os
import re
from pathlib import Path
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def apply_chat_template(
    example, tokenizer, assistant_prefix="<|assistant|>\n", require_revised=False, include_revised=True
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if not all(k in example for k in ("real", "generated")):
        raise ValueError(f"Require [real, generated] keys but found {list(example.keys())}")
    if require_revised and not example.get("revised"):
        raise ValueError("Training with alpha < 1 requires a revised conversation for every example.")
    responses = ["real", "generated"]
    if include_revised and example.get("revised"):
        responses.append("revised")
    for name in responses:
        messages = example[name]
        if not messages or len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"{name} must contain a prompt followed by an assistant response.")
        if messages[:-1] != example["real"][:-1]:
            raise ValueError(f"{name} and real must share the same prompt messages.")
        example[f"text_{name}"] = _strip_prefix(
            tokenizer.apply_chat_template(messages[-1:], tokenize=False), assistant_prefix
        )
    prompt_messages = list(example["real"][:-1])
    if prompt_messages[0]["role"] != "system":
        prompt_messages.insert(0, {"role": "system", "content": ""})
    example["text_prompt"] = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    return example


def prepare_datasets(raw_datasets, tokenizer, num_proc=None, alpha=0.5):
    """Format each split independently; test data retains the original two responses."""
    formatted = DatasetDict()
    for split, dataset in raw_datasets.items():
        use_revised = split == "train" and alpha < 1.0
        if use_revised and "revised" not in dataset.column_names:
            raise ValueError("Training with alpha < 1 requires the revised column.")
        formatted[split] = dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer, "require_revised": use_revised, "include_revised": use_revised},
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Formatting {split} comparisons with prompt template",
        )
        columns = {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
        if use_revised:
            columns["text_revised"] = "revised"
        formatted[split] = formatted[split].rename_columns(columns)
    return formatted


def _load_local_split(path, split):
    """Load local JSON/JSONL/Parquet splits separately so their schemas may differ."""
    root = Path(path)
    if not root.is_dir():
        return None
    if (root / split / "dataset_info.json").is_file():
        return load_from_disk(str(root / split))
    files_by_format = {}
    for extension, builder in [("json", "json"), ("jsonl", "json"), ("parquet", "parquet")]:
        files = list(root.glob(f"{split}*.{extension}"))
        files += list((root / split).rglob(f"*.{extension}"))
        if files:
            files_by_format.setdefault(builder, []).extend(str(file) for file in files)
    if len(files_by_format) > 1:
        raise ValueError(f"Split {split} in {path} mixes JSON and Parquet files; use one format per split.")
    if files_by_format:
        builder, files = next(iter(files_by_format.items()))
        return load_dataset(builder, data_files={split: sorted(files)}, split=split)
    return None


def get_datasets(
    data_config: DataArguments | dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif type(data_config) is dict:
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            dataset = _load_local_split(ds, split)
            if dataset is None:
                try:
                    dataset = load_dataset(ds, split=split)
                except DatasetGenerationError:
                    dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
