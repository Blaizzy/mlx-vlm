"""The CLI and notebook API share preparation, configuration, and training."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.train import _load_dataset, build_parser, parse_args, run
from mlx_vlm.trainer import DPOTrainingArgs, Trainer, VLMTrainingArgs
from mlx_vlm.trainer.peft import DoRALinear


def test_config_cli_precedence_and_algorithm_selection(tmp_path):
    config = tmp_path / "train.json"
    config.write_text(
        json.dumps(
            {
                "algorithm": "dpo",
                "model": "unused",
                "batch_size": 4,
                "compile": True,
                "beta": 0.3,
            }
        )
    )
    args, settings = parse_args(
        [
            "-c",
            str(config),
            "--batch-size",
            "2",
            "--no-compile",
        ]
    )
    assert isinstance(settings, DPOTrainingArgs)
    assert settings.batch_size == 2 and settings.beta == 0.3
    assert not settings.compile
    assert args.algorithm == "dpo"
    _, settings = parse_args(
        ["--model", "unused", "--train-mode", "orpo", "--beta", "0.4"]
    )
    assert settings.beta == 0.4


@pytest.mark.parametrize(
    "flags",
    [
        ["--iters", "1", "--epochs", "2"],
        ["--task", "asr"],
        ["--algorithm", "infonce"],
        ["--dataset-config", "[]"],
        ["--reference-model", "unused"],
        ["--batch-size", "0"],
    ],
)
def test_invalid_cli_options_fail_before_loading(flags):
    with pytest.raises(SystemExit):
        parse_args(["--model", "unused", *flags])


def test_legacy_cli_flags_and_optional_none_values(tmp_path):
    args, settings = parse_args(
        [
            "--model-path",
            "unused",
            "--dataset",
            "owner/data",
            "--split",
            "train",
            "--full-finetune",
            "--adapter-path",
            "saved",
            "--output-path",
            str(tmp_path),
            "--steps-per-eval",
            "none",
            "--assistant-id",
            "none",
            "--cache-size",
            "none",
        ]
    )
    assert args.model == "unused" and args.dataset == "owner/data"
    assert args.train_type == "full" and args.resume_adapter_file == "saved"
    assert settings.adapter_file == str(tmp_path / "adapters.safetensors")
    assert (
        settings.steps_per_eval is settings.assistant_id is settings.cache_size is None
    )
    assert "--guide-model" not in build_parser().format_help()


def test_huggingface_dataset_loading_uses_named_configuration():
    args, _ = parse_args(
        [
            "--model",
            "unused",
            "--dataset",
            "owner/data",
            "--hf-dataset-config",
            "images",
        ]
    )
    with patch("datasets.load_dataset", return_value={"train": []}) as load:
        assert _load_dataset(args) == {"train": []}
    load.assert_called_once_with("owner/data", "images")


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny", vocab_size=16)
        self.embedding = nn.Embedding(16, 64)
        self.projection = nn.Linear(64, 16)

    def __call__(self, ids, pixels=None, mask=None, **kwargs):
        return self.projection(self.embedding(ids))


def test_cli_raw_rows_prepare_quantize_train_evaluate_and_save(tmp_path):
    args, settings = parse_args(
        [
            "--model",
            "unused",
            "--train-type",
            "dora",
            "--quantization-bits",
            "4",
            "--iters",
            "3",
            "--gradient-accumulation-steps",
            "2",
            "--train-on-completions",
            "--max-seq-length",
            "8",
            "--pad-to-multiple",
            "1",
            "--adapter-file",
            str(tmp_path / "weights.safetensors"),
        ]
    )
    model = TinyVLM()
    processor = SimpleNamespace(pad_token_id=0)
    rows = [{"question": "Which color?", "answer": "red"}]

    def template(processor, config, messages, *, add_generation_prompt, **kwargs):
        return "prefix" if add_generation_prompt else "full"

    def prepare(**kwargs):
        ids = [1, 2] if kwargs["prompts"] == ["prefix"] else [1, 2, 3, 4]
        return {"input_ids": np.array([ids])}

    def build_trainer(prepared, *positional, **kwargs):
        assert isinstance(prepared.projection, DoRALinear)
        assert isinstance(prepared.projection.linear, nn.QuantizedLinear)
        return Trainer(prepared, *positional, **kwargs)

    with (
        patch("mlx_vlm.train._load_model", return_value=(model, processor)),
        patch(
            "mlx_vlm.train._load_dataset", return_value={"train": rows, "valid": rows}
        ),
        patch("mlx_vlm.train.Trainer", side_effect=build_trainer),
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
            return_value=({}, template, prepare, None),
        ),
    ):
        metrics = run(args, settings)
    assert metrics["optimizer_step"] == 2
    assert mx.isfinite(mx.array(metrics["train_loss"])).item()
    assert Path(settings.adapter_file).is_file()
    metadata = json.loads((tmp_path / "adapter_config.json").read_text())
    assert metadata["fine_tune_type"] == "dora"
    assert set(metadata["base_quantization"]) == {"embedding", "projection"}


def test_trainer_requires_one_split_and_processor_for_raw_rows():
    with pytest.raises(ValueError, match="processing_class"):
        Trainer(TinyVLM(), VLMTrainingArgs(), train_dataset=[])
    with pytest.raises(TypeError, match="one dataset split"):
        Trainer(
            TinyVLM(), VLMTrainingArgs(), train_dataset={"train": []}, prepared=True
        )
