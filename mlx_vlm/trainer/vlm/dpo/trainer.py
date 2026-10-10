"""Direct Preference Optimization for vision-language models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import mlx.nn as nn

from mlx_vlm.trainer.vlm.dpo.config import DPOTrainingArgs
from mlx_vlm.trainer.vlm.preference.dataset import (
    PreferenceVisionDataset,
    iterate_vlm_preference_batches,
)
from mlx_vlm.trainer.vlm.preference.loss import dpo_objective, dpo_vlm_loss
from mlx_vlm.trainer.vlm.preference.trainer import evaluate_preference_vlm


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _nested_config_value(config: Any, parent: str, name: str) -> Any:
    nested = _config_value(config, parent)
    return _config_value(nested, name)


def _validate_reference_model(policy: nn.Module, reference: nn.Module) -> None:
    policy_config = getattr(policy, "config", None)
    reference_config = getattr(reference, "config", None)
    for name in ("model_type", "image_token_index", "image_token_id"):
        policy_value = _config_value(policy_config, name)
        reference_value = _config_value(reference_config, name)
        if (
            policy_value is not None
            and reference_value is not None
            and policy_value != reference_value
        ):
            raise ValueError(
                f"DPO reference model {name}={reference_value!r} does not match "
                f"policy model {name}={policy_value!r}."
            )
    policy_vocab = _nested_config_value(
        policy_config, "text_config", "vocab_size"
    ) or _config_value(policy_config, "vocab_size")
    reference_vocab = _nested_config_value(
        reference_config, "text_config", "vocab_size"
    ) or _config_value(reference_config, "vocab_size")
    if (
        policy_vocab is not None
        and reference_vocab is not None
        and policy_vocab != reference_vocab
    ):
        raise ValueError(
            "DPO policy and reference models use different vocabulary sizes."
        )


dpo_loss = dpo_vlm_loss


def _prepare_reference_model(model: nn.Module, reference_model: nn.Module) -> None:
    if reference_model is None or reference_model is model:
        raise ValueError("DPO requires a separate reference model.")
    _validate_reference_model(model, reference_model)
    reference_model.freeze()
    reference_model.eval()


def evaluate_dpo(
    model: nn.Module,
    reference_model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int,
    max_seq_length: int = 2048,
    *,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    delta: float = 50.0,
) -> float:
    _prepare_reference_model(model, reference_model)
    objective = partial(
        dpo_loss,
        reference_model=reference_model,
        beta=beta,
        loss_type=loss_type,
        delta=delta,
    )
    return evaluate_preference_vlm(
        model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length,
        loss_fn=objective,
    )


def train_dpo(
    model: nn.Module,
    reference_model: nn.Module,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any | None = None,
    args: DPOTrainingArgs | None = None,
    *,
    training_callback=None,
    seed: int | None = None,
) -> dict:
    """Train a prepared preference dataset using the shared task and engine."""
    from mlx_vlm.trainer.common.engine import train

    args = args or DPOTrainingArgs()
    args.validate()
    task = make_task(model, args, reference_model=reference_model)
    return train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        args,
        task,
        training_callback,
        seed,
    )


__all__ = [
    "DPOTrainingArgs",
    "PreferenceVisionDataset",
    "dpo_loss",
    "dpo_objective",
    "evaluate_dpo",
    "iterate_vlm_preference_batches",
    "train_dpo",
]


def make_task(
    model, args, processing_class=None, dataset_config=None, *, reference_model=None
):
    """Bind the DPO objective to the shared preference task."""
    from mlx_vlm.trainer.vlm.preference.trainer import make_task as preference_task

    _prepare_reference_model(model, reference_model)
    objective = partial(
        dpo_loss,
        reference_model=reference_model,
        beta=args.beta,
        loss_type=args.loss_type,
        delta=args.delta,
    )

    task = preference_task(
        model, args, processing_class, dataset_config, loss_fn=objective
    )
    task.state.append(reference_model.state)
    return task
