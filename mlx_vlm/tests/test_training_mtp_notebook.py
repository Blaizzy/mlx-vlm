"""Execute native MTP notebook cells with a frozen tiny MLX target."""

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from datasets import Dataset
from mlx.utils import tree_flatten

from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.tests.test_training_mtp import tiny_model

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples" / "training"
STANDARD_NOTEBOOKS = (
    "sft_text_only",
    "sft_vison_text",
    "dpo",
    "orpo",
    "grpo_vision",
    "grpo_text_only",
)


def test_mtp_notebook_trains_only_head_reloads_and_exports(tmp_path):
    document = json.loads((NOTEBOOKS / "sft_mtp.ipynb").read_text())
    namespace = {"__name__": "__notebook__"}
    rows = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Response {i}"},
                ]
                for i in range(20)
            ]
        }
    )
    target = tiny_model()
    del target["mtp_head"]
    base_before = {
        key: mx.array(value) for key, value in tree_flatten(target.parameters())
    }
    mx.eval(base_before)
    processor = SimpleNamespace(pad_token_id=0)
    initial_head = tiny_model().mtp_head
    before = mx.array(initial_head.fc.weight)
    mx.eval(before)

    def drafter(path, **kwargs):
        if path == "Qwen/Qwen3.5-2B":
            return tiny_model().mtp_head, "mtp"
        # Exercise the real local export loader, not a replacement artifact.
        return load_drafter(path, **kwargs)

    def template(*args, add_generation_prompt=False, **kwargs):
        return "prefix" if add_generation_prompt else "full"

    def prepare(**kwargs):
        ids = [1, 2] if kwargs["prompts"] == ["prefix"] else [1, 2, 3, 4]
        return {"input_ids": np.array([ids], dtype=np.int32)}

    with (
        patch("mlx_vlm.load", return_value=(target, processor)),
        patch("datasets.load_dataset", return_value=rows),
        patch("mlx_vlm.speculative.drafters.load_drafter", side_effect=drafter),
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
            exec(compile(source, f"sft_mtp:cell:{index}", "exec"), namespace)
            if "MODEL_ID =" in source:
                namespace["OUTPUT_DIR"] = tmp_path
                namespace["EXPORT_DIR"] = tmp_path / "mtp_drafter"
                namespace["args"] = replace(
                    namespace["args"],
                    iters=2,
                    gradient_accumulation_steps=1,
                    max_seq_length=16,
                    pad_to_multiple=1,
                    steps_per_report=1,
                    steps_per_eval=1,
                    steps_per_save=2,
                    adapter_file=str(tmp_path / "weights.safetensors"),
                )
    after = dict(tree_flatten(target.parameters()))
    for key, value in base_before.items():
        np.testing.assert_array_equal(np.array(value), np.array(after[key]))
    assert mx.abs(target.mtp_head.fc.weight - before).sum().item() > 0
    assert namespace["validation"]["val_num_mtp_tokens"] > 0
    assert namespace["exported_kind"] == "mtp"
    config = json.loads((tmp_path / "mtp_drafter/config.json").read_text())
    assert config["model_type"] == "qwen3_5_mtp"
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert json.loads((tmp_path / "mtp_drafter/training_info.json").read_text())[
        "head_only"
    ]


@pytest.mark.parametrize("notebook", (*STANDARD_NOTEBOOKS, "custom_loss"))
def test_existing_notebooks_do_not_use_mtp(notebook):
    document = json.loads((NOTEBOOKS / f"{notebook}.ipynb").read_text())
    for cell in document["cells"]:
        assert "mtp" not in "".join(cell["source"]).lower()
