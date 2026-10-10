"""Response scoring and model-bound DPO/ORPO objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy
from mlx_vlm.trainer.losses.preference import dpo_objective, orpo_objective
from mlx_vlm.trainer.vlm.sft.runtime import forward_vlm_logits, vlm_batch_metrics


def response_token_logps(
    model: nn.Module,
    batch: Mapping[str, Any],
) -> tuple[mx.array, mx.array, mx.array]:
    """Return response-only token log-probs, boolean target mask, and logits."""
    required = {"input_ids", "attention_mask", "completion_mask"}
    missing = required.difference(batch)
    if missing:
        raise KeyError("Missing preference batch fields: " + ", ".join(sorted(missing)))
    ids = batch["input_ids"]
    attention = batch["attention_mask"]
    completion = batch["completion_mask"]
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("Preference input_ids must have shape [batch, time>=2].")
    if attention.shape != ids.shape or completion.shape != ids.shape:
        raise ValueError("Preference attention/completion masks must match input_ids.")
    targets = ids[:, 1:]
    valid = mx.logical_and(
        attention[:, 1:].astype(mx.bool_), attention[:, :-1].astype(mx.bool_)
    )
    valid = mx.logical_and(valid, completion[:, 1:].astype(mx.bool_))
    logits = forward_vlm_logits(model, ids[:, :-1], attention[:, :-1], batch)
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "VLM logits must align with shifted preference targets; "
            f"got {logits.shape} and {targets.shape}."
        )
    logps = -token_cross_entropy(logits, targets)
    return mx.where(valid, logps, 0), valid, logits


def sequence_logps(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    length_normalize: bool = False,
) -> tuple[mx.array, mx.array, mx.array]:
    """Score each response by summed log probability, optionally averaging tokens."""
    token_logps, mask, logits = response_token_logps(model, batch)
    counts = mask.sum(axis=-1).astype(mx.float32)
    scores = token_logps.sum(axis=-1)
    if length_normalize:
        scores = scores / mx.maximum(counts, 1.0)
    return scores, counts, logits


def dpo_vlm_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    reference_model: nn.Module,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    delta: float = 50.0,
    **_: Any,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute batch-mean DPO and report rewards, pairs, and token counts."""
    chosen, chosen_tokens, _ = sequence_logps(
        model, batch["chosen"], length_normalize=loss_type == "ipo"
    )
    rejected, rejected_tokens, _ = sequence_logps(
        model, batch["rejected"], length_normalize=loss_type == "ipo"
    )
    reference_chosen, _, _ = sequence_logps(
        reference_model, batch["chosen"], length_normalize=loss_type == "ipo"
    )
    reference_rejected, _, _ = sequence_logps(
        reference_model, batch["rejected"], length_normalize=loss_type == "ipo"
    )
    losses, metrics = dpo_objective(
        chosen,
        rejected,
        mx.stop_gradient(reference_chosen),
        mx.stop_gradient(reference_rejected),
        beta=beta,
        loss_type=loss_type,
        delta=delta,
    )
    metrics.update(vlm_batch_metrics(batch, model))
    metrics["num_tokens"] = (chosen_tokens.sum() + rejected_tokens.sum()).astype(
        mx.int32
    )
    metrics["num_reference_tokens"] = metrics["num_processed_tokens"]
    return mx.mean(losses), metrics


def orpo_vlm_loss(
    model: nn.Module,
    batch: Mapping[str, Any],
    *,
    beta: float = 0.1,
    eps: float = 1e-6,
    **_: Any,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Compute ORPO and report preference metrics, pairs, and token counts."""
    chosen, chosen_tokens, _ = sequence_logps(
        model, batch["chosen"], length_normalize=True
    )
    rejected, rejected_tokens, _ = sequence_logps(
        model, batch["rejected"], length_normalize=True
    )
    losses, metrics = orpo_objective(chosen, rejected, beta=beta, eps=eps)
    metrics.update(vlm_batch_metrics(batch, model))
    metrics["num_tokens"] = (chosen_tokens.sum() + rejected_tokens.sum()).astype(
        mx.int32
    )
    return mx.mean(losses), metrics
