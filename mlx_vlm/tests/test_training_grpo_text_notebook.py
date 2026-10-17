"""Execute text GRPO notebook cells and verify rule-based rewards and prompts."""

import ast
import hashlib
import json
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest
from datasets import Dataset, DatasetDict
from mlx.utils import tree_flatten

from mlx_vlm.tests.test_training_notebooks import NotebookVLM
from mlx_vlm.trainer import prepare_model_for_training

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "examples/training/grpo_text_only.ipynb"
)


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def reward_namespace():
    namespace = {"re": re, "Decimal": Decimal, "InvalidOperation": InvalidOperation}
    source = next(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell["cell_type"] == "code" and "NUMBER_PATTERN =" in "".join(cell["source"])
    )
    exec(source, namespace)
    return namespace


@pytest.mark.parametrize(
    "completion, target, accuracy, format_score",
    [
        ("<think>56 + 83 = 139.</think><answer>139</answer>", "139", 1, 1),
        (" <think>Calculation.</think>\n<answer>139.0</answer> ", "139", 1, 1),
        ("<think>Calculation.</think><answer>1.39e2</answer>", "139", 1, 1),
        ("139", "139", 1, 0),
        ("<think>Calculation.</think><answer>140</answer>", "139", 0, 1),
        ("<think></think><answer>139</answer>", "139", 1, 0),
        ("<think>Calculation.</think><answer>139</answer> extra text", "139", 1, 0),
        ("<answer>139</answer><answer>140</answer>", "139", 0, 0),
        ("<answer>139</answer><answer>140", "139", 0, 0),
        ("<think>Calculation.</think><answer>139", "139", 0, 0),
        ("<answer>139 or 140</answer>", "139", 0, 0),
        ("The answer is 139.", "139", 0, 0),
        ("<answer>NaN</answer>", "139", 0, 0),
        ("<answer>Infinity</answer>", "139", 0, 0),
        ("<think>Calculation.</think><answer>-0</answer>", "0", 1, 1),
    ],
)
def test_notebook_rewards(completion, target, accuracy, format_score):
    namespace = reward_namespace()
    assert namespace["correctness_reward"](
        completions=[completion], answer=[target]
    ) == [accuracy]
    assert namespace["reasoning_format_reward"](completions=[completion]) == [
        format_score
    ]


class ArithmeticTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [6]

    def decode(self, tokens):
        if tokens == [3]:
            return "<think>56 + 83 = 139.</think><answer>139</answer>"
        if tokens == [5]:
            return "<think>56 + 83 = 140.</think><answer>140</answer>"
        return "prompt"


@pytest.mark.parametrize("train_type", ["lora", "dora", "full"])
@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_text_grpo_notebook_trains_evaluates_saves_and_reloads(
    tmp_path, train_type, beta
):
    train = Dataset.from_dict(
        {"prompt": ["What is 56 + 83?"] * 6, "answer": ["139"] * 6}
    )
    valid = Dataset.from_dict(
        {"prompt": ["What is 100 + 39?"] * 2, "answer": ["139"] * 2}
    )
    test = Dataset.from_dict(
        {"prompt": ["What is 70 + 69?"] * 2, "answer": ["139"] * 2}
    )
    namespace = {"__name__": "__notebook__"}
    template_calls, generation_calls, models = [], [], []

    def load(path):
        assert path == "mlx-community/Qwen3.5-2B-4bit"
        mx.random.seed(123)
        model, processor = NotebookVLM(), ArithmeticTokenizer()
        models.append(model)
        return model, processor

    def template(*args, **kwargs):
        raise AssertionError("Preformatted prompts must bypass chat templates.")

    def prepare(**kwargs):
        assert kwargs["images"] is None
        prompt = kwargs["prompts"][0]
        template_calls.append(prompt)
        assert "139" not in prompt
        assert prompt.endswith("\nAssistant:")
        return {"input_ids": mx.array([[1, 4]])}

    def generate(ids, model, pixels, mask, **kwargs):
        assert pixels is None
        generation_calls.append(kwargs)
        yield 3 if len(generation_calls) % 2 else 5, None
        yield 2, None
        yield 6, None  # Never consumed after EOS.

    with (
        patch("mlx_vlm.convert") as conversion,
        patch("mlx_vlm.load", side_effect=load),
        patch("mlx_vlm.trainer.WandBCallback") as wandb_callback,
        patch(
            "datasets.load_dataset",
            return_value=DatasetDict(train=train, valid=valid, test=test),
        ) as data_loading,
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
            return_value=({}, template, prepare, None),
        ),
        patch.object(
            import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
        ),
    ):
        for i, cell in enumerate(notebook_cells()):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source)
            exec(compile(source, f"grpo_text_only:cell:{i}", "exec"), namespace)
            if "MODEL_ID =" in source:
                namespace["TRAIN_TYPE"] = train_type
                namespace["OUTPUT_DIR"] = tmp_path / "adapters"
                namespace["args"] = replace(
                    namespace["args"],
                    iters=2,
                    group_size=2,
                    max_seq_length=16,
                    max_completion_length=4,
                    beta=beta,
                    grad_checkpoint=False,
                    val_batches=-1,
                    steps_per_report=2,
                    steps_per_eval=2,
                    steps_per_save=2,
                    adapter_file=str(tmp_path / "adapters/weights.safetensors"),
                )
            elif "model = prepare_model_for_training(" in source:
                policy_before = [
                    mx.array(value)
                    for _, value in tree_flatten(
                        namespace["model"].trainable_parameters()
                    )
                ]
                reference_before = (
                    [
                        mx.array(value)
                        for _, value in tree_flatten(models[1].parameters())
                    ]
                    if beta
                    else []
                )

    conversion.assert_not_called()
    wandb_callback.assert_called_once()
    wandb_callback.return_value.finish.assert_called_once_with()
    assert wandb_callback.call_args.kwargs["project_name"] == "mlx-vlm-grpo-text"
    data_loading.assert_called_once_with("mlx-community/simple-grpo")
    assert namespace["MODEL_ID"] == "mlx-community/Qwen3.5-2B-4bit"
    assert namespace["train_rows"]["question"] == train["prompt"]
    assert namespace["eval_rows"]["question"] == valid["prompt"]
    assert namespace["test_rows"]["question"] == test["prompt"]
    assert set(namespace["train_rows"].column_names) == {"prompt", "answer", "question"}
    assert len(models) == (2 if beta else 1)
    assert template_calls and generation_calls

    metrics = namespace["metrics"]
    assert mx.isfinite(mx.array(metrics["train_loss"])).item()
    assert metrics["train_reward_0_correctness_reward"] == 0.5
    assert metrics["train_reward_1_reasoning_format_reward"] == 1
    assert metrics["total_vision_tokens"] == 0
    assert namespace["test_metrics"]["val_num_sequences"] == len(test) * 2
    assert namespace["validation"]["val_num_vision_tokens"] == 0
    assert any(
        not mx.array_equal(before, after).item()
        for before, (_, after) in zip(
            policy_before, tree_flatten(namespace["model"].trainable_parameters())
        )
    )
    if beta:
        assert not tree_flatten(models[1].trainable_parameters())
        assert all(
            mx.array_equal(before, after).item()
            for before, (_, after) in zip(
                reference_before, tree_flatten(models[1].parameters())
            )
        )

    output = tmp_path / "adapters"
    assert (output / "weights.safetensors").is_file()
    assert (
        json.loads((output / "adapter_config.json").read_text())["fine_tune_type"]
        == train_type
    )
    assert not (output / "chat_template.jinja").exists()
    assert (output / "prompt_template.txt").read_text().format(
        question=train[0]["prompt"]
    ) == namespace["train_rows"][0]["prompt"]
    assert "chat_template" not in vars(namespace["training_tokenizer"])
    run_config = json.loads((output / "run_config.json").read_text())
    assert run_config["dataset_id"] == "mlx-community/simple-grpo"
    assert run_config["model_id"] == "mlx-community/Qwen3.5-2B-4bit"
    assert run_config["base_model_id"] == run_config["model_id"]
    assert run_config["reward_weights"] == [1.0, 0.1]
    assert run_config["dataset_config"] == {"apply_chat_template": False}
    assert run_config["prompt_source"] == "https://arxiv.org/html/2501.12948v1#S2.T1"

    # A saved adapter reload reproduces the trained policy on the original base.
    restored, _ = load(namespace["MODEL_ID"])
    restored = prepare_model_for_training(
        restored, train_type=train_type, checkpoint_path=output, verbose=False
    )
    original = namespace["model"]
    original.eval()
    restored.eval()
    ids = mx.array([[1, 4, 3]])
    assert mx.allclose(original(ids), restored(ids)).item()

    assert all("BASE_PATH" not in "".join(cell["source"]) for cell in notebook_cells())


def test_mapping_uses_published_instruction_without_answer_leakage():
    namespace = {"Decimal": Decimal, "InvalidOperation": InvalidOperation}
    settings_source = next(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell["cell_type"] == "code"
        and "FORMAT_INSTRUCTION =" in "".join(cell["source"])
    )
    settings = ast.parse(settings_source)
    instruction = next(
        node
        for node in settings.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FORMAT_INSTRUCTION"
            for target in node.targets
        )
    )
    exec(
        compile(ast.Module(body=[instruction], type_ignores=[]), "instruction", "exec"),
        namespace,
    )
    prefix = namespace["FORMAT_INSTRUCTION"]
    assert hashlib.sha256(" ".join(prefix.split()).encode()).hexdigest() == (
        "e3c33b9fab873701ecc25c8aeed6e2676cc74f3a926ba33bbd1a8e829bf1e042"
    )
    mapping_source = next(
        "".join(cell["source"])
        for cell in notebook_cells()
        if cell["cell_type"] == "code" and "def mapp_func(" in "".join(cell["source"])
    )
    mapping = ast.parse(mapping_source)
    function = next(
        node
        for node in mapping.body
        if isinstance(node, ast.FunctionDef) and node.name == "mapp_func"
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "mapping", "exec"),
        namespace,
    )
    row = namespace["mapp_func"]({"prompt": "What is 56 + 83?", "answer": "139"})
    assert row == {
        "prompt": prefix + "\nUser: What is 56 + 83?\nAssistant:",
        "question": "What is 56 + 83?",
        "answer": "139",
    }
    assert "139" not in row["prompt"]
    assert all(
        "CHAT_TEMPLATE" not in "".join(cell["source"]) for cell in notebook_cells()
    )
