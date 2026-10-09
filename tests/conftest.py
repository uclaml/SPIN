import os
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spin"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
torch.set_num_threads(2)


@pytest.fixture
def tokenizer():
    words = ["[PAD]", "[UNK]", "</s>", "hello", "good", "bad", "answer", "better", "clear",
             "<|user|>", "<|assistant|>", "<|system|>"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="</s>", model_max_length=64
    )
    tokenizer.add_special_tokens({"additional_special_tokens": words[-3:]})
    from alignment.data import DEFAULT_CHAT_TEMPLATE
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(42)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=2, eos_token_id=2, pad_token_id=0,
    ))


@pytest.fixture
def example():
    return {
        name: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": answer}]
        for name, answer in [("real", "good answer"), ("generated", "bad"), ("revised", "better clear answer")]
    }
