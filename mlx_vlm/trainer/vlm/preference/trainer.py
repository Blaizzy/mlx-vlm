"""Training lifecycle shared by VLM DPO and ORPO."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.nn as nn

from mlx_vlm.trainer.common.utils import evaluate_weighted_batches
from mlx_vlm.trainer.vlm.preference.dataset import iterate_vlm_preference_batches
from mlx_vlm.trainer.vlm.sft.runtime import _make_vlm_task


def evaluate_preference_vlm(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int,
    max_seq_length: int,
    *,
    loss_fn: Callable,
    iterate_batches: Callable = iterate_vlm_preference_batches,
    pad_to_multiple: int = 32,
    pad_token_id: int = 0,
) -> float:
    """Return the distributed pair-weighted objective on a preference split."""
    if dataset is None or len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty preference dataset.")
    batches = iterate_batches(
        dataset,
        batch_size,
        max_seq_length,
        train=False,
        pad_to_multiple=pad_to_multiple,
        pad_token_id=pad_token_id,
    )
    return evaluate_weighted_batches(
        model,
        batches,
        loss_fn,
        num_batches,
        empty_error="Preference evaluation produced zero pairs.",
    )


def make_task(
    model,
    args,
    processing_class=None,
    dataset_config=None,
    *,
    loss_fn,
    iterate_batches=iterate_vlm_preference_batches,
):
    """Share media preprocessing and pair batching across preference algorithms."""
    from mlx_vlm.trainer.vlm.preference.dataset import PreferenceVisionDataset

    return _make_vlm_task(
        args,
        lambda rows: PreferenceVisionDataset(
            rows,
            model.config,
            processing_class,
            config=dataset_config or args,
            **{
                f"{name}_feature": getattr(dataset_config, f"{name}_feature", name)
                for name in ("prompt", "chosen", "rejected")
            },
        ),
        loss_fn=loss_fn,
        iterate_batches=iterate_batches,
        unit="preference_pairs",
    )


def train_preference_vlm(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args,
    *,
    loss_fn,
    iterate_batches=iterate_vlm_preference_batches,
    reference_model=None,
    training_callback=None,
    seed=None,
):
    from mlx_vlm.trainer.common.engine import train

    args.validate()
    task = make_task(model, args, loss_fn=loss_fn, iterate_batches=iterate_batches)
    if reference_model is not None:
        reference_model.freeze()
        reference_model.eval()
        task.state.append(reference_model.state)
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
