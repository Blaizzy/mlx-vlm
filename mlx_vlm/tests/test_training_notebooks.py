"""Run the standalone notebook code with tiny real MLX models and fixture processors."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from mlx_vlm.tests.test_training_cli import TinyVLM
from mlx_vlm.trainer import prepare_model_for_training

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples" / "training"


class NotebookVLM(TinyVLM):
    def __init__(self):
        super().__init__()
        self.config.image_token_index = 9
        self.layers = [nn.Linear(64, 64)]

    def __call__(self, ids, pixels=None, mask=None, **kwargs):
        hidden = self.embedding(ids)
        for layer in self.layers:
            hidden = nn.relu(layer(hidden))
        return self.projection(hidden)


@pytest.mark.parametrize(
    "notebook", ["sft_text_only", "sft_vison_text", "orpo", "dpo", "custom_loss"]
)
@pytest.mark.parametrize("train_type", ["lora", "dora", "full"])
def test_notebook_cells_train_evaluate_and_reload(notebook, train_type, tmp_path):
    document = json.loads((NOTEBOOKS / f"{notebook}.ipynb").read_text())
    assert document["nbformat"] == 4
    processor = type("Processor", (), {"pad_token_id": 0})()
    latex_targets = [rf" \frac{{x_{i}}}{{2}} + \alpha " for i in range(10)]

    text_conversations = [
        [
            {"role": "user", "content": f"Explain example {index}."},
            {
                "role": "assistant",
                "content": f"Response {index}.\nKeep this text unchanged.",
            },
        ]
        for index in range(10)
    ]

    invalid_text_conversations = [
        [{"role": "user", "content": "A prompt without an answer."}],
        text_conversations[0]
        + [{"role": "user", "content": "An unanswered follow-up."}],
        [
            {"role": "user", "content": "A prompt with an empty answer."},
            {"role": "assistant", "content": " \n "},
        ],
        [],
    ]

    def load_dataset_fixture(dataset_id, *, revision=None, split=None):
        if dataset_id == "mlx-community/JOSIE-v2-Instruct-5K":
            assert split == "train"
            return Dataset.from_dict(
                {"messages": text_conversations + invalid_text_conversations}
            )
        assert dataset_id == "mlx-community/LaTeX_OCR"
        features = Features({"image": Image(), "text": Value("string")})
        rows = Dataset.from_dict(
            {
                "image": [PILImage.new("RGB", (32, 16), "white")] * 10,
                "text": latex_targets,
            },
            features=features,
        )
        return DatasetDict({"train": rows, "test": rows.select([0, 1])})

    def load(path):
        mx.random.seed(123)
        return NotebookVLM(), processor

    template_calls = []

    def template(processor, config, messages, *, add_generation_prompt, **kwargs):
        template_calls.append(kwargs)
        return "prefix" if add_generation_prompt else "full"

    def prepare(**kwargs):
        ids = [1, 2] if kwargs["prompts"] == ["prefix"] else [1, 2, 3, 4]
        if kwargs.get("images"):
            ids[1] = 9  # Processor-expanded image slot in the prompt.
        result = {"input_ids": np.array([ids], dtype=np.int32)}
        if kwargs.get("images"):
            result["pixel_values"] = np.ones((1, 2, 4), dtype=np.float32)
        return result

    namespace = {"__name__": "__notebook__"}
    with (
        patch("mlx_vlm.load", side_effect=load),
        patch("datasets.load_dataset", side_effect=load_dataset_fixture),
        patch("mlx_vlm.prompt_utils.apply_chat_template", side_effect=template),
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
            return_value=({}, template, prepare, None),
        ),
    ):
        for index, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source)
            exec(compile(source, f"{notebook}:cell:{index}", "exec"), namespace)
            if "MODEL_ID =" in source:
                namespace["TRAIN_TYPE"] = train_type
                namespace["args"] = replace(
                    namespace["args"],
                    iters=2,
                    max_seq_length=16,
                    pad_to_multiple=1,
                    steps_per_eval=2,
                    steps_per_report=2,
                    steps_per_save=2,
                    adapter_file=str(tmp_path / "weights.safetensors"),
                )
    if notebook in {"sft_text_only", "custom_loss"}:
        train_rows = namespace["train_rows"]
        assert namespace["dropped_rows"] == len(invalid_text_conversations)
        assert train_rows.column_names == ["messages"]
        assert len(train_rows) == 9 and len(namespace["eval_rows"]) == 1
        for rows in (train_rows, namespace["eval_rows"]):
            assert all(row["messages"] in text_conversations for row in rows)
        prepared = namespace["trainer"].train_dataset[0]
        assert prepared["completion_mask"].tolist() == [0, 0, 1, 1]
        assert "pixel_values" not in prepared
        assert all(call.get("num_images", 0) == 0 for call in template_calls)
        assert namespace["metrics"]["total_vision_tokens"] == 0
    if notebook == "sft_vison_text":
        train_rows = namespace["train_rows"]
        assert set(train_rows.column_names) == {"image", "messages"}
        assert isinstance(train_rows.features["image"], Image)
        assert isinstance(train_rows[0]["image"], PILImage.Image)
        assert len(train_rows) == 9 and len(namespace["eval_rows"]) == 1
        for row in train_rows:
            assert row["messages"][0]["content"] == namespace["OCR_PROMPT"]
            assert row["messages"][1]["content"] in latex_targets
        assert (
            namespace["latex_to_messages"](latex_targets[0])["messages"][1]["content"]
            == latex_targets[0]
        )
        with pytest.raises(ValueError, match="non-empty"):
            namespace["latex_to_messages"]("")
        assert namespace["formatted_sample"] == "full"
        prepared = namespace["trainer"].train_dataset[0]
        assert prepared["completion_mask"].tolist() == [0, 0, 1, 1]
        assert "pixel_values" in prepared
    assert mx.isfinite(mx.array(namespace["metrics"]["train_loss"])).item()
    if notebook not in {"sft_text_only", "custom_loss"}:
        expected = 4 if notebook in {"dpo", "orpo"} else 2
        assert namespace["metrics"]["total_vision_tokens"] == expected
        assert namespace["validation"]["val_num_vision_tokens"] > 0
    assert mx.isfinite(mx.array(namespace["validation"]["val_loss"])).item()
    assert (tmp_path / "weights.safetensors").is_file()
    metadata = json.loads((tmp_path / "adapter_config.json").read_text())
    assert metadata["fine_tune_type"] == train_type
    original = namespace["model"]
    original.eval()
    restored, _ = load("unused")
    restored = prepare_model_for_training(
        restored,
        train_type=train_type,
        checkpoint_path=tmp_path,
        verbose=False,
    )
    restored.eval()
    ids = mx.array([[1, 2, 3]])
    assert mx.allclose(original(ids), restored(ids)).item()
