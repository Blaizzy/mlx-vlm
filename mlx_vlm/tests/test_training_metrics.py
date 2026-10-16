"""Dynamic loss metrics, weighted reductions, timing, and notebook history."""

import math
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from mlx_vlm.tests.test_training_api import Recorder, TinyLM
from mlx_vlm.trainer import CoreTrainingArgs, Trainer, TrainingTask, VLMTrainingArgs
from mlx_vlm.trainer.common.metrics import (
    MetricAccumulator,
    format_metric_report,
    normalize_loss_output,
    report_loss_metrics,
)
from mlx_vlm.trainer.losses.cross_entropy import cross_entropy
from mlx_vlm.trainer.losses.preference import dpo_objective, orpo_objective
from mlx_vlm.trainer.vlm.orpo.trainer import ORPOTrainingArgs as LegacyORPOTrainingArgs
from mlx_vlm.trainer.vlm.orpo.trainer import orpo_loss as legacy_orpo_loss
from mlx_vlm.trainer.vlm.orpo.trainer import train_orpo
from mlx_vlm.trainer.vlm.sft.runtime import (
    vlm_batch_metrics,
    vlm_sft_loss,
    vlm_vision_mask,
)
from mlx_vlm.trainer.vlm.sft.trainer import vision_language_loss_fn


def test_dynamic_metrics_weighted_means_optional_keys_counts_and_distributed_sum():
    accumulator = MetricAccumulator()
    accumulator.add(mx.array(2.0), {"weight": 1, "quality": 0.0, "num_tokens": 1})
    accumulator.add(
        mx.array(4.0), {"weight": 3, "quality": 1.0, "num_tokens": 3, "extra": 7.0}
    )
    # Simulate identical counts on a second worker; means stay unchanged.
    with mock.patch("mlx.core.distributed.all_sum", side_effect=lambda x, **_: x * 2):
        values = accumulator.compute()
    assert values == {
        "weight": 8.0,
        "loss": 3.5,
        "quality": 0.75,
        "extra": 7.0,
        "num_tokens": 8,
    }
    report = report_loss_metrics(values, "val")
    output = format_metric_report({"iteration": 2, **report}, "Validation")
    for key in report:
        assert f"{key}=" in output
    assert "val_quality=0.75" in output


def test_empty_metrics_default_weight_validation_and_perplexity_after_reduction():
    accumulator = MetricAccumulator()
    accumulator.add(mx.array(1.0), {})
    accumulator.add(mx.array(3.0), {})
    assert accumulator.compute()["loss"] == 2.0
    with pytest.raises(ValueError, match="scalar"):
        normalize_loss_output((mx.array(1.0), {"vector": mx.ones((2,))}))
    with pytest.raises(ValueError, match="names"):
        normalize_loss_output((mx.array(1.0), {"loss": 1.0}))
    values = {"loss": 2.0, "nll": 2.0, "weight": 1}
    assert report_loss_metrics(values, "train")["train_perplexity"] == math.exp(2.0)
    assert "train_perplexity" not in report_loss_metrics({"loss": 2.0}, "train")


def test_supervised_accuracy_and_forward_token_counts_exclude_padding():
    logits = mx.array([[[3.0, 1.0], [3.0, 1.0], [1.0, 3.0]]])
    targets = mx.array([[0, 1, -100]])
    _, metrics = cross_entropy(logits, targets)
    assert metrics["token_accuracy"].item() == 0.5
    assert metrics["num_tokens"].item() == 2
    batch = {
        "input_ids": mx.array([[1, 2, 3, 4, 0], [1, 2, 3, 0, 0]]),
        "attention_mask": mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "completion_mask": mx.array([[0, 0, 1, 1, 0], [0, 0, 1, 0, 0]]),
    }
    counts = vlm_batch_metrics(batch)
    assert {k: v.item() for k, v in counts.items()} == {
        "num_processed_tokens": 7,
        "num_padded_tokens": 1,
        "num_sequences": 2,
    }
    _, metrics = vlm_sft_loss(TinyLM(), batch, train_on_completions=True)
    assert metrics["weight"].item() == 3
    for key, value in counts.items():
        assert metrics[key].item() == value.item()
    pair_counts = vlm_batch_metrics({"chosen": batch, "rejected": batch})
    assert pair_counts["num_processed_tokens"].item() == 14


@pytest.mark.parametrize("mapping", [False, True])
def test_vision_mask_uses_config_aliases_and_excludes_padding_and_final_target(mapping):
    config = {"image_token_index": 5, "image_token_id": 5, "video_token_id": 6}
    model = SimpleNamespace(config=config if mapping else SimpleNamespace(**config))
    batch = {
        "input_ids": mx.array([[1, 5, 5, 6, 5, 5]]),
        "attention_mask": mx.array([[1, 1, 1, 1, 0, 1]]),
    }
    assert vlm_vision_mask(model, batch).tolist() == [[False, True, True, True, False]]
    metrics = vlm_batch_metrics(batch, model)
    assert metrics["num_vision_tokens"].item() == 3
    assert (
        vlm_batch_metrics({"chosen": batch, "rejected": batch}, model)[
            "num_vision_tokens"
        ].item()
        == 6
    )
    model.config = {"image_token_index": 0}
    batch["input_ids"] = mx.array([[0, 0, 1, 2, 0, 0]])
    assert vlm_batch_metrics(batch, model)["num_vision_tokens"].item() == 2
    model.config = {}
    assert vlm_batch_metrics(batch, model)["num_vision_tokens"].item() == 0


@pytest.mark.parametrize("compiled", [False, True])
def test_vision_counts_flow_through_train_validation_and_callbacks(compiled, tmp_path):
    model = TinyLM()
    model.config.image_token_id = 5
    row = {
        "input_ids": mx.array([1, 5, 5, 2, 3, 0]),
        "attention_mask": mx.array([1, 1, 1, 1, 1, 0]),
        "completion_mask": mx.array([0, 0, 0, 0, 1, 0]),
    }
    callback = Recorder()
    trainer = Trainer(
        model,
        VLMTrainingArgs(
            iters=3,
            batch_size=1,
            gradient_accumulation_steps=2,
            max_seq_length=8,
            pad_to_multiple=1,
            train_on_completions=True,
            steps_per_report=2,
            steps_per_eval=2,
            steps_per_save=3,
            compile=compiled,
            val_batches=-1,
            adapter_file=str(tmp_path / "weights.safetensors"),
        ),
        train_dataset=[row],
        eval_dataset=[row],
        prepared=True,
        training_callback=callback,
    )
    metrics = trainer.train()
    assert metrics["trained_tokens"] == 3
    assert metrics["total_vision_tokens"] == 6
    assert metrics["train_num_vision_tokens"] == 2
    assert metrics["vision_tokens_per_second"] == pytest.approx(
        2 / metrics["step_time"]
    )
    assert callback.train[0]["train_num_vision_tokens"] == 4
    assert callback.train[0]["total_vision_tokens"] == 4
    assert callback.val[0]["val_num_vision_tokens"] == 2
    assert callback.val[0]["val_vision_tokens_per_second"] > 0
    validation = trainer.evaluate()
    assert validation["val_num_vision_tokens"] == 2
    assert validation["val_num_tokens"] == 1


def test_preference_losses_report_metrics_without_changing_objectives():
    chosen, rejected = mx.array([-1.0, -2.0]), mx.array([-2.0, -1.0])
    reference = mx.array([-3.0, -3.0])
    for kind in ("sigmoid", "hinge", "ipo", "dpop"):
        losses, metrics = dpo_objective(
            chosen, rejected, reference, reference, loss_type=kind
        )
        assert losses.shape == (2,)
        assert metrics["reward_accuracy"].item() == 0.5
        assert metrics["reward_margin"].item() == 0.0
        assert metrics["weight"].item() == 2
        logits = chosen - rejected
        if kind == "sigmoid":
            expected = -nn.log_sigmoid(0.1 * logits)
        elif kind == "hinge":
            expected = nn.relu(1 - 0.1 * logits)
        elif kind == "ipo":
            expected = (logits - 5) ** 2
        else:
            expected = -nn.log_sigmoid(0.1 * logits) + 50 * mx.maximum(
                reference - chosen, 0
            )
        assert mx.allclose(losses, expected).item()
    losses, metrics = orpo_objective(chosen, rejected)
    odds = chosen - mx.log1p(-mx.exp(chosen)) - rejected + mx.log1p(-mx.exp(rejected))
    assert mx.allclose(losses, -chosen - 0.1 * nn.log_sigmoid(odds)).item()
    assert metrics["preference_accuracy"].item() == 0.5
    assert metrics["chosen_nll"].item() == 1.5
    legacy_loss, legacy_metrics = legacy_orpo_loss(
        chosen, mx.array(0.0), rejected, mx.array(0.0), mx.ones((2, 3)), mx.ones((2, 3))
    )
    assert mx.allclose(legacy_loss, losses.mean()).item()
    assert isinstance(legacy_metrics, dict)


@pytest.mark.parametrize("compiled", [False, True])
def test_train_and_validation_print_arbitrary_loss_metrics_and_retain_history(
    compiled, tmp_path, capsys
):
    model = nn.Linear(1, 1, bias=False)
    model.weight = mx.array([[10.0]])
    callback = Recorder()

    def batches(rows, train=False, seed=None):
        while True:
            yield from rows
            if not train:
                return

    def loss(model, batch):
        loss = model(batch).sum()
        return loss, {
            "weight": mx.array(3),
            "num_tokens": mx.array(3),
            "num_processed_tokens": mx.array(5),
            "num_padded_tokens": mx.array(1),
            "num_sequences": mx.array(1),
            "my_quality_score": loss * 0.1,
        }

    task = TrainingTask(loss, batches)
    trainer = Trainer(
        model,
        CoreTrainingArgs(
            iters=3,
            steps_per_report=2,
            steps_per_eval=2,
            steps_per_save=3,
            gradient_accumulation_steps=2,
            compile=compiled,
            val_batches=-1,
            adapter_file=str(tmp_path / "weights.safetensors"),
        ),
        task=task,
        optimizer=optim.SGD(learning_rate=optim.linear_schedule(0.1, 0.01, 10)),
        train_dataset=[mx.ones((1, 1))],
        eval_dataset=[mx.ones((1, 1))],
        training_callback=callback,
    )
    metrics = trainer.train()
    assert metrics["optimizer_step"] == 2
    assert metrics["trained_tokens"] == 9
    assert metrics["processed_tokens"] == 15
    assert metrics["total_sequences"] == 3
    assert metrics["padding_fraction"] == pytest.approx(1 / 6)
    assert metrics["average_sequence_length"] == 5
    assert metrics["progress"] == 1 and metrics["remaining_time"] == 0
    assert metrics["processed_tokens_per_second"] == pytest.approx(
        5 / metrics["step_time"]
    )
    assert metrics["tokens_per_second"] == pytest.approx(3 / metrics["step_time"])
    assert (
        metrics["end_to_end_tokens_per_second"]
        <= metrics["processed_tokens_per_second"]
    )
    assert metrics["learning_rate"] == trainer.optimizer.learning_rate.item()
    assert callback.train[0]["learning_rate"] > callback.train[-1]["learning_rate"]
    assert [r["iteration"] for r in trainer.log_history] == [2, 2, 3, 3]
    assert trainer.log_history[0] == callback.train[0]
    assert trainer.log_history[1] == callback.val[0]
    assert trainer.log_history[0] is not callback.train[0]
    assert "train_my_quality_score" in metrics and "val_my_quality_score" in metrics
    assert "train_perplexity" not in metrics
    assert metrics["val_num_tokens"] == 3
    assert metrics["val_processed_tokens_per_second"] > 0
    output = capsys.readouterr().out
    assert "train_my_quality_score=" in output and "val_my_quality_score=" in output
    assert "learning_rate=" in output and "processed_tokens_per_second=" in output
    model.eval()
    validation = trainer.evaluate()
    assert "val_my_quality_score" in validation
    assert not model.training


def test_existing_sft_loss_returns_dictionary():
    class OutputModel(TinyLM):
        def __call__(self, *args, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(logits=super().__call__(*args, **kwargs))

    batch = {
        "input_ids": mx.array([[1, 2, 3]]),
        "attention_mask": mx.ones((1, 3)),
        "pixel_values": None,
    }
    loss, metrics = vision_language_loss_fn(OutputModel(), batch)
    assert loss.ndim == 0 and isinstance(metrics, dict)
    assert metrics["num_tokens"].item() == 2


def test_existing_orpo_pipeline_prints_train_and_validation_metrics(tmp_path, capsys):
    from types import SimpleNamespace

    class OutputModel(TinyLM):
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(logits=super().__call__(*args, **kwargs))

    sequence = {
        "input_ids": mx.array([[1, 2, 3]]),
        "attention_mask": mx.ones((1, 3)),
        "pixel_values": None,
    }
    rows = [{"chosen": sequence, "rejected": sequence}]

    def batches(dataset, batch_size, max_seq_length, train=False):
        while True:
            yield from dataset
            if not train:
                return

    model = OutputModel()
    optimizer = optim.Adam(learning_rate=0.01)
    optimizer.init(model.trainable_parameters())
    args = LegacyORPOTrainingArgs(
        iters=1,
        batch_size=1,
        steps_per_report=1,
        steps_per_save=1,
        steps_per_eval=1,
        val_batches=1,
        adapter_file=str(tmp_path / "adapters.safetensors"),
    )
    with mock.patch(
        "mlx_vlm.trainer.vlm.orpo.trainer.iterate_batches", side_effect=batches
    ):
        train_orpo(model, optimizer, rows, rows, args=args)
    output = capsys.readouterr().out
    assert "train_policy_chosen_logps=" in output
    assert "val_policy_chosen_logps=" in output
    assert (tmp_path / "adapters.safetensors").is_file()
