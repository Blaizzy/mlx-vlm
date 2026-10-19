"""GRPO numerics, memory chunk equivalence, multimodal rollouts, and lifecycle."""

import ast
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.train import parse_args, run
from mlx_vlm.trainer import GRPOTrainingArgs, Trainer
from mlx_vlm.trainer.vlm.grpo.dataset import GRPOVisionDataset
from mlx_vlm.trainer.vlm.grpo.loss import completion_logps, grpo_objective
from mlx_vlm.trainer.vlm.grpo.rewards import calculate_rewards_and_advantages
from mlx_vlm.trainer.vlm.grpo.trainer import generate_grpo


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="tiny", vocab_size=8, image_token_index=7
        )
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8)

    def __call__(self, ids, pixels=None, mask=None, **kwargs):
        hidden = self.embedding(ids)
        if pixels is not None:
            hidden = hidden + pixels.mean()
        return self.projection(hidden)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [5, 6]

    def decode(self, tokens):
        return "yes" if tokens == [3] else "no"


def settings(tmp_path, **kwargs):
    return GRPOTrainingArgs(
        max_seq_length=16,
        max_completion_length=4,
        group_size=2,
        iters=3,
        gradient_accumulation_steps=2,
        steps_per_report=3,
        steps_per_save=3,
        steps_per_eval=3,
        val_batches=-1,
        adapter_file=str(tmp_path / "weights.safetensors"),
        **kwargs,
    )


def row(vision=False):
    inputs = {
        "input_ids": mx.array([1, 7] if vision else [1]),
        "attention_mask": mx.ones((2 if vision else 1,), dtype=mx.int32),
    }
    if vision:
        inputs.update(
            pixel_values=mx.ones((1, 2, 4)), image_grid_thw=mx.array([[1, 1, 1]])
        )
    return {"inputs": inputs, "prompt": "question", "answer": "yes", "metadata": {}}


def sampled_stream():
    calls = []

    def generate(ids, model, pixels, mask, **kwargs):
        calls.append((ids.tolist(), pixels, kwargs))
        yield 3 if len(calls) % 2 else 4, None
        yield 2, None
        yield 6, None  # Must never be consumed after EOS.

    return generate, calls


@pytest.mark.parametrize("level", ["token", "sequence"])
def test_on_policy_gradient_and_reference_is_not_old_policy(level):
    logps = mx.array([[-1.0, -2.0]])
    ref = mx.array([[-3.0, -4.0]])
    mask = mx.array([[True, True]])
    grad = mx.grad(
        lambda x: grpo_objective(
            x, ref, mask, mx.array([1.0]), importance_sampling_level=level
        )[0]
    )(logps)
    assert mx.allclose(grad, mx.array([[-0.5, -0.5]])).item()
    loss, info = grpo_objective(
        logps, ref, mask, mx.array([1.0]), importance_sampling_level=level
    )
    assert loss.item() == -1.0
    assert info["clip_ratio"].item() == 0


@pytest.mark.parametrize(
    "variant, expected", [("grpo", -1.5), ("bnpo", -5 / 3), ("dr_grpo", -5 / 8)]
)
def test_loss_denominators_and_masked_nonfinite_values(variant, expected):
    logps = mx.array([[-1, float("nan")], [-1, -1]])
    mask = mx.array([[True, False], [True, True]])
    loss, info = grpo_objective(
        logps, logps, mask, mx.array([1.0, 2.0]), grpo_loss_type=variant, max_tokens=4
    )
    assert loss.item() == pytest.approx(expected)
    assert info["kl"].item() == 0
    _, info = grpo_objective(
        mx.array([[-1000.0]]),
        mx.array([[-1.0]]),
        mx.array([[True]]),
        mx.array([0.0]),
        beta=0.1,
    )
    assert mx.isfinite(info["kl"]).item() and info["kl_clip_ratio"].item() == 1


def test_cpu_rewards_weighting_missing_coverage_and_zero_variance():
    calls = []

    def a(**kwargs):
        calls.append(kwargs)
        return [1, 3, None, 2]

    def b(**kwargs):
        return [None, 1, 2, 2]

    advantages, metrics = calculate_rewards_and_advantages(
        [a, b],
        ["p"] * 4,
        ["c"] * 4,
        ["a"] * 4,
        [0, 0, 1, 1],
        reward_weights=[0, 1],
    )
    assert len(calls) == 1
    assert advantages.tolist() == pytest.approx([-1 / 1.0002, 1 / 1.0002, 0, 0])
    assert metrics["reward_0_coverage"].item() == 0.75
    for callback, match in [
        (lambda **kw: [None] * 4, "All reward"),
        (lambda **kw: [float("inf")] * 4, "infinite"),
        (lambda **kw: [1], "one reward"),
    ]:
        with pytest.raises(ValueError, match=match):
            calculate_rewards_and_advantages(
                [callback], ["p"] * 4, ["c"] * 4, ["a"] * 4, [0] * 4
            )


@pytest.mark.parametrize("vision", [False, True])
def test_rollouts_keep_exact_ids_eos_media_and_restore_mode(tmp_path, vision):
    args, model = settings(tmp_path), TinyVLM()
    model.train()
    generate, calls = sampled_stream()
    with patch.object(
        import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
    ):
        completions, texts, indices = generate_grpo(
            model, Tokenizer(), [row(vision)], args
        )
    assert [x.tolist() for x in completions] == [[3, 2], [4, 2]]
    assert texts == ["yes", "no"] and indices == [0, 0]
    assert model.training
    assert all((pixels is not None) == vision for _, pixels, _ in calls)
    if vision:
        assert all("image_grid_thw" in kw for _, _, kw in calls)
    with patch.object(
        import_module("mlx_vlm.generate"),
        "generate_step",
        side_effect=RuntimeError("failure"),
    ):
        with pytest.raises(RuntimeError):
            generate_grpo(model, Tokenizer(), [row(vision)], args)
    assert model.training
    with pytest.raises(ValueError, match="never truncated"):
        generate_grpo(model, Tokenizer(), [row(True)], replace(args, max_seq_length=5))


@pytest.mark.parametrize("variant", ["grpo", "bnpo", "dr_grpo"])
@pytest.mark.parametrize("compiled", [False, True])
def test_chunk_gradients_match_full_objective_for_unequal_lengths(
    tmp_path, variant, compiled
):
    model = TinyVLM()
    args = settings(tmp_path, grpo_loss_type=variant, compile=compiled)
    trainer = Trainer(model, args, processing_class=Tokenizer(), prepared=True)
    samples = []
    for ids, advantage in [([1, 3, 2], 1), ([1, 4, 4, 2], -1)]:
        batch = {
            "input_ids": mx.array([ids]),
            "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32),
        }
        samples.append(
            {
                "batch": batch,
                "start": 0,
                "advantage": mx.array([advantage], dtype=mx.float32),
            }
        )
    rollout = {"samples": samples, "reward_metrics": {}}
    chunked = trainer.task.value_and_grad_factory(model, compiled)
    (loss, _), gradients = chunked(model, rollout)

    def full(m):
        logps = [completion_logps(m, s["batch"], 0) for s in samples]
        padded = mx.concatenate([mx.pad(logps[0], [(0, 0), (0, 1)]), logps[1]], axis=0)
        return grpo_objective(
            padded,
            mx.stop_gradient(padded),
            mx.array([[True, True, False], [True, True, True]]),
            mx.array([1.0, -1.0]),
            grpo_loss_type=variant,
            max_tokens=4,
        )[0]

    expected, full_grads = nn.value_and_grad(model, full)(model)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    for (_, a), (_, b) in zip(tree_flatten(gradients), tree_flatten(full_grads)):
        assert mx.allclose(a, b, atol=1e-6).item()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_trainer_updates_saves_and_evaluates(tmp_path, compiled, vision):
    model = TinyVLM()
    before = mx.array(model.projection.weight)
    generate, _ = sampled_stream()
    with patch.object(
        import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
    ):
        trainer = Trainer(
            model,
            settings(tmp_path, compile=compiled),
            optimizer=optim.AdamW(learning_rate=1e-5, weight_decay=0.0),
            processing_class=Tokenizer(),
            train_dataset=[row(vision)],
            eval_dataset=[row(vision)],
            prepared=True,
        )
        metrics = trainer.train()
        model.eval()
        validation = trainer.evaluate()
    assert mx.isfinite(mx.array(metrics["train_loss"])).item()
    assert metrics["optimizer_step"] == 2
    assert not mx.array_equal(before, model.projection.weight).item()
    assert (tmp_path / "weights.safetensors").is_file()
    assert validation["val_reward_mean"] == 0.5
    assert not model.training
    assert metrics["total_vision_tokens"] == (6 if vision else 0)


def test_kl_reference_frozen_and_beta_zero_skips_it(tmp_path):
    model, reference = TinyVLM(), TinyVLM()
    before = [mx.array(v) for _, v in tree_flatten(reference.parameters())]
    with pytest.raises(ValueError, match="reference_model"):
        Trainer(
            model,
            settings(tmp_path, beta=0.1),
            processing_class=Tokenizer(),
            prepared=True,
        )
    generate, _ = sampled_stream()
    with patch.object(
        import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
    ):
        trainer = Trainer(
            model,
            settings(tmp_path, beta=0.1),
            optimizer=optim.AdamW(learning_rate=1e-5, weight_decay=0.0),
            processing_class=Tokenizer(),
            train_dataset=[row(True)],
            prepared=True,
            reference_model=reference,
        )
        trainer.train()
    assert not tree_flatten(reference.trainable_parameters())
    assert all(
        mx.array_equal(a, b).item()
        for a, (_, b) in zip(before, tree_flatten(reference.parameters()))
    )
    trainer = Trainer(
        model,
        settings(tmp_path),
        processing_class=Tokenizer(),
        prepared=True,
        reference_model=reference,
    )
    with patch(
        "mlx_vlm.trainer.vlm.grpo.trainer.completion_logps", wraps=completion_logps
    ) as scoring:
        trainer.task.objective(
            model,
            {
                "samples": [
                    {
                        "batch": {
                            "input_ids": mx.array([[1, 3]]),
                            "attention_mask": mx.ones((1, 2)),
                        },
                        "start": 0,
                        "advantage": mx.array([1.0]),
                    }
                ],
                "reward_metrics": {},
            },
        )
    assert scoring.call_count == 1


def helper_fixtures():
    def template(processor, config, messages, **kwargs):
        assert kwargs["add_generation_prompt"]
        assert messages[-1]["role"] == "user"
        return "prompt"

    def prepare(**kwargs):
        output = {"input_ids": np.array([[1, 7]], dtype=np.int32)}
        if kwargs.get("images"):
            output["pixel_values"] = np.ones((1, 2, 4), dtype=np.float32)
        return output

    return {}, template, prepare, None


def test_prompt_preprocessing_custom_columns_and_no_answer_leakage():
    with patch(
        "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
        return_value=helper_fixtures(),
    ):
        dataset = GRPOVisionDataset(
            [{"question": "question", "target": "yes", "image": object()}],
            TinyVLM().config,
            Tokenizer(),
            config={"prompt_feature": "question", "answer_feature": "target"},
        )
        prepared = dataset[0]
        assert prepared["answer"] == "yes" and "pixel_values" in prepared["inputs"]
        with pytest.raises(ValueError, match="not an answer"):
            dataset.process({"messages": [{"role": "assistant", "content": "target"}]})


def test_grpo_cli_custom_rewards(tmp_path):
    reward_file = tmp_path / "rewards.py"
    reward_file.write_text(
        "def custom(*, completions, **kwargs):\n    return list(range(len(completions)))\n"
    )
    args, config = parse_args(
        [
            "--algorithm",
            "grpo",
            "--model",
            "tiny",
            "--iters",
            "1",
            "--max-seq-length",
            "16",
            "--max-completion-length",
            "4",
            "--group-size",
            "2",
            "--epsilon-high",
            "0.2",
            "--reward-functions-file",
            str(reward_file),
            "--reward-functions",
            "custom",
            "--reward-weights",
            "[1.0]",
            "--output-path",
            str(tmp_path),
        ]
    )
    assert config.epsilon_high == 0.2
    generate, _ = sampled_stream()
    with (
        patch("mlx_vlm.train._load_model", return_value=(TinyVLM(), Tokenizer())),
        patch("mlx_vlm.train._configure_model", side_effect=lambda model, args: model),
        patch(
            "mlx_vlm.train._load_dataset",
            return_value=[{"prompt": "q", "answer": "yes"}],
        ),
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
            return_value=helper_fixtures(),
        ),
        patch.object(
            import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
        ),
    ):
        metrics = run(args, config)
    assert metrics["train_reward_mean"] == 0.5


def test_notebook_runs_training_evaluation_and_saving(tmp_path):
    from datasets import Dataset, DatasetDict
    from PIL import Image

    from mlx_vlm.tests.test_training_notebooks import NotebookVLM

    document = json.loads(
        (
            Path(__file__).resolve().parents[2] / "examples/training/grpo_vision.ipynb"
        ).read_text()
    )
    rows = Dataset.from_dict(
        {"image": [Image.new("RGB", (32, 16), "white")] * 10, "text": ["yes"] * 10}
    )
    namespace = {"__name__": "__notebook__"}
    notebook_model = NotebookVLM()
    notebook_model.config.image_token_index = 7
    generate, _ = sampled_stream()
    with (
        patch("mlx_vlm.load", return_value=(notebook_model, Tokenizer())),
        patch("datasets.load_dataset", return_value=DatasetDict(train=rows)),
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
            return_value=helper_fixtures(),
        ),
        patch.object(
            import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
        ),
    ):
        for i, cell in enumerate(document["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source)
            exec(compile(source, f"grpo_vision:cell:{i}", "exec"), namespace)
            if "MODEL_ID =" in source:
                namespace["args"] = replace(
                    namespace["args"],
                    iters=2,
                    group_size=2,
                    max_seq_length=16,
                    max_completion_length=4,
                    grad_checkpoint=False,
                    steps_per_report=2,
                    steps_per_eval=2,
                    steps_per_save=2,
                    adapter_file=str(tmp_path / "weights.safetensors"),
                )
    assert namespace["MODEL_ID"] == "mlx-community/Qwen3.5-2B-4bit"
    assert namespace["metrics"]["train_reward_mean"] > 0
    assert namespace["validation"]["val_num_vision_tokens"] > 0
    assert (tmp_path / "adapter_config.json").is_file()


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_qwen35_chunked_projection_matches_dense_scores_and_gradients(vision, tied):
    from mlx_vlm.tests.test_models import _qwen_patch_model
    from mlx_vlm.trainer import prepare_model_for_training
    from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy
    from mlx_vlm.trainer.vlm.grpo.trainer import _cache_frozen_vision
    from mlx_vlm.trainer.vlm.sft.runtime import forward_vlm_logits

    model, _ = _qwen_patch_model()
    model.config.image_token_index = model.config.image_token_id = 7
    model.config.video_token_index = model.config.video_token_id = 8
    model.config.vision_start_token_id = 5
    model.config.vision_end_token_id = 6
    model.config.text_config.tie_word_embeddings = tied
    nn.quantize(model.language_model, group_size=32, bits=4)
    model = prepare_model_for_training(model, lora_rank=2, lora_alpha=4, verbose=False)
    ids = [1, 5, 7, 7, 7, 7, 6, 3, 2] if vision else [1, 4, 3, 2]
    batch = {
        "input_ids": mx.array([ids]),
        "attention_mask": mx.ones((1, len(ids)), dtype=mx.int32),
    }
    if vision:
        batch.update(
            pixel_values=mx.ones((4, 96)), image_grid_thw=mx.array([[1, 2, 2]])
        )
        prepared = dict(
            row(True),
            inputs={
                key: val[0] if key in {"input_ids", "attention_mask"} else val
                for key, val in batch.items()
            },
        )
        cached = _cache_frozen_vision(model, [prepared])[0]["inputs"]
        assert "cached_image_features" in cached
        batch["cached_image_features"] = cached["cached_image_features"]
    start = len(ids) - 3

    def dense(m):
        logits = forward_vlm_logits(
            m, batch["input_ids"][:, :-1], batch["attention_mask"][:, :-1], batch
        )
        return -token_cross_entropy(
            logits[:, start:], batch["input_ids"][:, start + 1 :]
        ).sum()

    def chunked(m):
        return completion_logps(m, batch, start, chunk_size=1).sum()

    dense_value, dense_grad = nn.value_and_grad(model, dense)(model)
    value, gradient = nn.value_and_grad(model, chunked)(model)
    assert mx.allclose(value, dense_value, atol=1e-5).item()
    for (_, a), (_, b) in zip(tree_flatten(gradient), tree_flatten(dense_grad)):
        assert mx.allclose(a, b, atol=1e-5).item()
    if vision:
        cached.update(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        assert mx.allclose(
            completion_logps(model, cached, start, 1).sum(), dense_value, atol=1e-5
        ).item()
        model.vision_tower.unfreeze()
        assert (
            "cached_image_features"
            not in _cache_frozen_vision(model, [prepared])[0]["inputs"]
        )


def test_real_qwen35_vision_generation_and_checkpointed_qlora_update(tmp_path):
    from mlx_vlm.tests.test_models import _qwen_patch_model
    from mlx_vlm.trainer import prepare_model_for_training

    model, _ = _qwen_patch_model()
    # The fused Metal recurrence requires head dimensions of at least 32.
    model.config.text_config.linear_key_head_dim = 32
    model.config.text_config.linear_value_head_dim = 32
    model = type(model)(model.config)
    model.config.image_token_index = model.config.image_token_id = 7
    model.config.video_token_index = model.config.video_token_id = 8
    model.config.vision_start_token_id = 5
    model.config.vision_end_token_id = 6
    model.config.text_config.eos_token_id = model.config.eos_token_id = 2
    nn.quantize(model.language_model, group_size=32, bits=4)
    model = prepare_model_for_training(model, lora_rank=2, lora_alpha=4, verbose=False)
    inputs = {
        "input_ids": mx.array([1, 5, 7, 7, 7, 7, 6]),
        "attention_mask": mx.ones((7,), dtype=mx.int32),
        "pixel_values": mx.ones((4, 96)),
        "image_grid_thw": mx.array([[1, 2, 2]]),
    }
    prepared = {"inputs": inputs, "prompt": "describe", "answer": "yes", "metadata": {}}
    before = [mx.array(v) for _, v in tree_flatten(model.trainable_parameters())]
    trainer = Trainer(
        model,
        settings(tmp_path, grad_checkpoint=True, top_k=0),
        optimizer=optim.AdamW(learning_rate=1e-5, weight_decay=0.0),
        processing_class=Tokenizer(),
        prepared=True,
        train_dataset=[prepared],
        reward_funcs=[lambda **kw: [float(i) for i in range(len(kw["completions"]))]],
    )
    result = trainer.train()
    assert mx.isfinite(mx.array(result["train_loss"])).item()
    assert result["total_vision_tokens"] == 24
    assert any(
        not mx.array_equal(a, b).item()
        for a, (_, b) in zip(before, tree_flatten(model.trainable_parameters()))
    )


@pytest.mark.parametrize(
    "option",
    [
        {"group_size": 1},
        {"max_completion_length": 16},
        {"temperature": float("nan")},
        {"epsilon_high": -0.1},
        {"advantage_epsilon": 0},
        {"top_p": 0},
        {"grpo_loss_type": "unknown"},
        {"importance_sampling_level": "unknown"},
    ],
)
def test_invalid_grpo_settings_fail_early(tmp_path, option):
    with pytest.raises(ValueError):
        replace(settings(tmp_path), **option).validate()


def test_prompt_length_sorting_reuses_preprocessing_cache(tmp_path):
    helpers = helper_fixtures()
    with (
        patch(
            "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers", return_value=helpers
        ),
        patch(
            "mlx_vlm.trainer.vlm.grpo.dataset.GRPOVisionDataset.process", wraps=None
        ) as processing,
    ):
        processing.return_value = row()
        trainer = Trainer(
            TinyVLM(),
            settings(tmp_path, cache_size=1),
            optimizer=optim.AdamW(learning_rate=1e-5, weight_decay=0.0),
            processing_class=Tokenizer(),
            train_dataset=[{"prompt": "question", "answer": "yes"}],
        )
        assert trainer.train_dataset.itemlen(0) == 1
        assert trainer.train_dataset[0]["prompt"] == "question"
        assert processing.call_count == 1


def test_sampling_suppresses_only_in_vocabulary_media_slots(tmp_path):
    model = TinyVLM()
    model.config.video_token_index = -200
    model.config.video_token_id = 999
    generate, calls = sampled_stream()
    with patch.object(
        import_module("mlx_vlm.generate"), "generate_step", side_effect=generate
    ):
        generate_grpo(model, Tokenizer(), [row(True)], settings(tmp_path))
    assert all(kw["logit_bias"] == {7: -float("inf")} for _, _, kw in calls)


def test_preformatted_grpo_prompts_bypass_template_and_preserve_text():
    prompt = "Instruction.\nUser: What is 2 + 2?\nAssistant:"
    seen = []

    def template(*args, **kwargs):
        raise AssertionError("Do not wrap a preformatted prompt.")

    def prepare(**kwargs):
        seen.append(kwargs["prompts"])
        return {"input_ids": np.array([[1, 3]], dtype=np.int32)}

    with patch(
        "mlx_vlm.trainer.vlm.sft.dataset._load_vlm_helpers",
        return_value=({}, template, prepare, None),
    ):
        dataset = GRPOVisionDataset(
            [{"prompt": prompt, "answer": "4"}],
            TinyVLM().config,
            Tokenizer(),
            config={"apply_chat_template": False},
        )
        sample = dataset[0]
        assert seen == [[prompt]]
        assert sample["prompt"] == prompt and sample["answer"] == "4"
        assert sample["inputs"]["input_ids"].tolist() == [1, 3]
        for invalid in (
            {"prompt": []},
            {"prompt": " "},
            {"prompt": prompt, "system": "An instruction that would be lost."},
        ):
            with pytest.raises(ValueError):
                dataset.process(invalid)
