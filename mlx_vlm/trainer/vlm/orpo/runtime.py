"""Odds Ratio Preference Optimization for vision-language models."""

from __future__ import annotations

from functools import partial
from typing import Any

import mlx.nn as nn

from mlx_vlm.trainer.vlm.orpo.config import ORPOTrainingArgs
from mlx_vlm.trainer.vlm.preference.dataset import (
    PreferenceVisionDataset,
    iterate_vlm_preference_batches,
)
from mlx_vlm.trainer.vlm.preference.loss import orpo_objective, orpo_vlm_loss
from mlx_vlm.trainer.vlm.preference.trainer import evaluate_preference_vlm

orpo_loss = orpo_vlm_loss


def evaluate_orpo(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int,
    max_seq_length: int = 2048,
    *,
    beta: float = 0.1,
    eps: float = 1e-6,
) -> float:
    objective = partial(orpo_loss, beta=beta, eps=eps)
    return evaluate_preference_vlm(
        model,
        dataset,
        batch_size,
        num_batches,
        max_seq_length,
        loss_fn=objective,
    )


def train_orpo(
    model: nn.Module,
    optimizer: Any,
    train_dataset: Any,
    val_dataset: Any | None = None,
    args: ORPOTrainingArgs | None = None,
    *,
    training_callback=None,
    seed: int | None = None,
) -> dict:
    """Train a prepared preference dataset using the shared task and engine."""
    from mlx_vlm.trainer.common.engine import train

    args = args or ORPOTrainingArgs()
    args.validate()
    task = make_task(model, args)
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
    "ORPOTrainingArgs",
    "PreferenceVisionDataset",
    "evaluate_orpo",
    "iterate_vlm_preference_batches",
    "orpo_loss",
    "orpo_objective",
    "train_orpo",
]


def make_task(
    model, args, processing_class=None, dataset_config=None, *, reference_model=None
):
    """Bind the ORPO objective to the shared preference task."""
    from mlx_vlm.trainer.vlm.preference.trainer import make_task as preference_task

    objective = partial(orpo_loss, beta=args.beta, eps=args.eps)

    return preference_task(
        model, args, processing_class, dataset_config, loss_fn=objective
    )
