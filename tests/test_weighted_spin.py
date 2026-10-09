import copy
import json
import math
from pathlib import Path
import subprocess
import sys

from datasets import Dataset, DatasetDict
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, T5Config, T5ForConditionalGeneration
import yaml

from alignment import SPINConfig, SPINTrainer
from alignment.data import apply_chat_template, get_datasets, prepare_datasets
from alignment.utils import DataCollatorWithPadding


def write_data(root, example, file_format):
    root.mkdir()
    rows = {
        "train": [example, example, example, example],
        "test": [{k: v for k, v in example.items() if k != "revised"}] * 2,
    }
    for split, records in rows.items():
        path = root / f"{split}.{file_format}"
        if file_format == "parquet":
            Dataset.from_list(records).to_parquet(path)
        elif file_format == "jsonl":
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
        else:
            path.write_text(json.dumps(records))


@pytest.mark.parametrize("file_format", ["json", "jsonl", "parquet"])
def test_different_train_test_schemas(tmp_path, tokenizer, example, file_format):
    data = tmp_path / "data"
    write_data(data, example, file_format)
    raw = get_datasets({str(data): 1.0})
    assert len(raw["train"]) == 4 and len(raw["test"]) == 2
    assert "revised" in raw["train"].column_names
    assert "revised" not in raw["test"].column_names
    formatted = prepare_datasets(raw, tokenizer)
    assert set(formatted["train"].column_names) == {"prompt", "real", "generated", "revised"}
    assert set(formatted["test"].column_names) == {"prompt", "real", "generated"}
    assert "better clear answer" in formatted["train"][0]["revised"]
    assert "hello" not in formatted["train"][0]["revised"]


def test_missing_revised_and_legacy_alpha_one(tokenizer, example):
    old = {k: v for k, v in example.items() if k != "revised"}
    raw = DatasetDict(train=Dataset.from_list([old]), test=Dataset.from_list([old]))
    with pytest.raises(ValueError, match="revised column"):
        prepare_datasets(raw, tokenizer)
    legacy = prepare_datasets(raw, tokenizer, alpha=1.0)
    assert "revised" not in legacy["train"].column_names
    raw["train"] = Dataset.from_list([example, {**old, "revised": None}])
    with pytest.raises(ValueError, match="every example"):
        prepare_datasets(raw, tokenizer)


def test_revised_prompt_must_match(tokenizer, example):
    example["revised"][0]["content"] = "different question"
    with pytest.raises(ValueError, match="same prompt"):
        apply_chat_template(example, tokenizer, require_revised=True)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_alpha_validation(tmp_path, alpha):
    with pytest.raises(ValueError, match="alpha"):
        SPINConfig(output_dir=str(tmp_path), alpha=alpha, use_cpu=True)
    with pytest.raises(ValueError, match="alpha"):
        SPINTrainer(alpha=alpha)


@pytest.mark.parametrize("truncation_mode", ["keep_start", "keep_end"])
def test_collator_shared_prompt_masking_and_padding(tokenizer, truncation_mode):
    collator = DataCollatorWithPadding(
        tokenizer, max_length=10, max_prompt_length=4, truncation_mode=truncation_mode, padding_value=7
    )
    feature = {"prompt": "hello good bad answer hello good", "real": "good", "generated": "bad answer",
               "revised": "better " * 12}
    short = {"prompt": "hello", "real": "good", "generated": "bad", "revised": "clear"}
    batch = collator([feature, short])
    prompt = tokenizer(feature["prompt"], add_special_tokens=False)["input_ids"]
    expected_prompt = prompt[:4] if truncation_mode == "keep_start" else prompt[-4:]
    for response in ("real", "generated", "revised"):
        assert batch[f"{response}_input_ids"].shape[1] <= 10
        assert batch[f"{response}_input_ids"][0, :4].tolist() == expected_prompt
        assert batch[f"{response}_labels"][0, :4].tolist() == [-100] * 4
        assert batch[f"{response}_labels"][1, 0].item() == -100
        assert set(batch[f"{response}_attention_mask"].flatten().tolist()) <= {0, 1}
        assert batch[f"{response}_labels"][1, -1].item() == -100
    with pytest.raises(ValueError, match="every example"):
        collator([feature, {k: v for k, v in short.items() if k != "revised"}])


@pytest.mark.parametrize("loss_type", ["sigmoid", "hinge"])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
def test_weighted_loss_and_gradient_match_formula(alpha, loss_type):
    trainer = object.__new__(SPINTrainer)
    trainer.alpha, trainer.beta, trainer.loss_type = alpha, 0.3, loss_type
    trainer.model, trainer.ref_model = object(), object()
    values = torch.tensor([[-1.0, -3.0], [-4.0, -1.0], [-0.5, -6.0]], requires_grad=True)
    reference = torch.tensor([[-2.0, -2.0], [-3.0, -2.0], [-2.5, -4.0]])
    calls = []

    def forward(model, batch, include_revised=True):
        calls.append((model, include_revised))
        v = values if model is trainer.model else reference
        names = ["real", "generated", "revised"] if include_revised else ["real", "generated"]
        return {name: (v[i], torch.zeros(2, 1, 1)) for i, name in enumerate(names)}

    trainer.concatenated_forward = forward
    loss, metrics = trainer.get_batch_metrics(trainer.model, {"revised_labels": torch.ones(2, 1)})
    real_margin = (values[0] - values[1]) - (reference[0] - reference[1])
    revised_margin = (values[2] - values[1]) - (reference[2] - reference[1])
    objective = (lambda m: F.softplus(-0.3 * m)) if loss_type == "sigmoid" else (lambda m: (1 - 0.3 * m).clamp_min(0))
    expected = (alpha * objective(real_margin) + (1 - alpha) * objective(revised_margin)).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(torch.autograd.grad(loss, values, retain_graph=True)[0],
                               torch.autograd.grad(expected, values)[0])
    assert len(calls) == 2  # One policy and one frozen-reference forward for all responses.
    if alpha < 1:
        torch.testing.assert_close(metrics["loss/weighted"], expected.detach())
        with pytest.raises(ValueError, match="requires revised"):
            trainer.get_batch_metrics(trainer.model, {})
    else:
        legacy_loss, _ = trainer.get_batch_metrics(trainer.model, {})
        torch.testing.assert_close(legacy_loss, expected)
    eval_loss, eval_metrics = trainer.get_batch_metrics(trainer.model, {}, train_eval="eval")
    torch.testing.assert_close(eval_loss, objective(real_margin).mean())
    assert not any("revised" in key for key in eval_metrics)
    assert calls[-1][1] is False and calls[-2][1] is False


@pytest.mark.parametrize("encoder_decoder", [False, True])
def test_model_forward_training_and_original_evaluation(tmp_path, tokenizer, model, example, encoder_decoder):
    if encoder_decoder:
        model = T5ForConditionalGeneration(T5Config(
            vocab_size=len(tokenizer), d_model=16, d_ff=32, d_kv=8, num_layers=1, num_decoder_layers=1,
            num_heads=2, dropout_rate=0.0, pad_token_id=0, decoder_start_token_id=0, eos_token_id=2,
        ))
    old = {k: v for k, v in example.items() if k != "revised"}
    data = prepare_datasets(DatasetDict(train=Dataset.from_list([example] * 2),
                                       test=Dataset.from_list([old] * 2)), tokenizer)
    args = SPINConfig(output_dir=str(tmp_path), use_cpu=True, report_to=[], max_steps=1, alpha=0.5,
                      per_device_train_batch_size=2, per_device_eval_batch_size=2, learning_rate=0.001)
    trainer = SPINTrainer(model=model, args=args, tokenizer=tokenizer, max_length=32, max_prompt_length=16,
                          max_target_length=16, train_dataset=data["train"], eval_dataset=data["test"])
    batch = trainer.data_collator([data["train"][0], data["train"][1]])
    combined = trainer.concatenated_forward(model, batch)
    for response in ("real", "generated", "revised"):
        if encoder_decoder:
            logits = model(input_ids=batch["prompt_input_ids"], attention_mask=batch["prompt_attention_mask"],
                           labels=batch[f"{response}_labels"]).logits
            labels = batch[f"{response}_labels"]
        else:
            logits = model(input_ids=batch[f"{response}_input_ids"],
                           attention_mask=batch[f"{response}_attention_mask"]).logits[:, :-1]
            labels = batch[f"{response}_labels"][:, 1:]
        mask = labels != -100
        logps = logits.log_softmax(-1).gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(combined[response][0], (logps * mask).sum(-1))
    before = copy.deepcopy(model.state_dict())
    result = trainer.train()
    assert trainer.state.global_step == 1 and math.isfinite(result.training_loss)
    assert any(not torch.equal(before[k], value) for k, value in model.state_dict().items())
    assert all(p.grad is None for p in trainer.ref_model.parameters())
    evaluation = trainer.evaluate()
    assert math.isfinite(evaluation["eval_loss"])
    assert evaluation["eval_loss"] == pytest.approx(evaluation["eval_loss/real"])
    assert not any("revised" in key for key in evaluation)


def test_training_entrypoint_with_alpha_override(tmp_path, model, tokenizer, example):
    repo = Path(__file__).resolve().parents[1]
    data = tmp_path / "data"
    write_data(data, example, "json")
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    output = tmp_path / "output"
    config = {
        "model_name_or_path": str(model_dir), "torch_dtype": "float32", "use_flash_attention_2": False,
        "dataset_mixer": {str(data): 1.0}, "dataset_splits": ["train", "test"], "preprocessing_num_workers": 1,
        "output_dir": str(output), "use_cpu": True, "max_steps": 2, "learning_rate": 0.001,
        "max_length": 32, "max_prompt_length": 16, "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2, "logging_steps": 1, "do_eval": True,
        "evaluation_strategy": "steps", "eval_steps": 1, "save_strategy": "no", "report_to": [],
        "push_to_hub": False, "alpha": 0.5, "disable_tqdm": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, "spin/run_spin.py", str(config_path), "--alpha=0.25"],
                            cwd=repo, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 2
    records = [row for row in state["log_history"] if "loss/weighted" in row]
    assert len(records) == 2
    assert records[-1]["loss/real"] != pytest.approx(records[-1]["loss/revised"])
    for row in records:
        assert row["loss/weighted"] == pytest.approx(0.25 * row["loss/real"] + 0.75 * row["loss/revised"])
    evaluation = [row for row in state["log_history"] if "eval_loss" in row]
    assert len(evaluation) == 2
    assert all(row["eval_loss"] == pytest.approx(row["eval_loss/real"]) for row in evaluation)
    trained = AutoModelForCausalLM.from_pretrained(output)
    assert not torch.equal(model.model.embed_tokens.weight, trained.model.embed_tokens.weight)
    saved_args = torch.load(output / "training_args.bin", map_location="cpu")
    assert saved_args.alpha == 0.25
