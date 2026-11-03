"""Token cross-entropy shared by language, vision-language, and speech models.

Callers own model forwarding and causal shifting: logits [B, T, V] must
already align with targets [B, T], e.g. input_ids[:, 1:] for next-token loss.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def make_loss_mask(targets: mx.array, lengths: mx.array) -> mx.array:
    """Mask inclusive [start, end] positions in the unshifted token sequence.

    Targets are shifted by one, so their first column has position 1. Each
    row in lengths has shape [2]. This matches the ASR transcript boundaries.
    """
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, time].")
    if lengths.shape != (targets.shape[0], 2):
        raise ValueError("lengths must have shape [batch, 2].")
    steps = mx.arange(1, targets.shape[1] + 1)[None, :]
    return mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:2])


def token_cross_entropy(
    logits: mx.array,
    targets: mx.array,
    *,
    ignore_index: int | None = -100,
) -> mx.array:
    """Return unreduced token cross-entropy, zeroing ignored labels.

    Preference objectives need sequence log-probabilities, while SFT needs a
    mean over supervised tokens. Keeping the unreduced calculation here makes
    both paths share identical target validation and CE semantics.
    """
    if targets.ndim != 2 or logits.ndim != 3:
        raise ValueError(
            "Expected logits [batch, time, vocab] and targets [batch, time]."
        )
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and targets must have matching batch/time dimensions.")
    mask = mx.ones(targets.shape, dtype=mx.bool_)
    if ignore_index is not None:
        mask = targets != ignore_index
    safe_targets = mx.where(mask, targets, 0)
    token_loss = nn.losses.cross_entropy(logits, safe_targets, reduction="none")
    return mx.where(mask, token_loss, 0).astype(mx.float32)


def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array | None = None,
    *,
    ignore_index: int | None = -100,
    vision_mask: mx.array | None = None,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Return mean supervised-token loss and scalar auxiliary metrics.

    An optional boolean mask selects target positions. Ignored labels are
    excluded even when selected by the mask; masked or ignored labels are
    replaced before cross-entropy so sentinels are never used as class IDs.
    ``weight`` and ``num_tokens`` count supervised targets; ``token_accuracy``
    is masked next-token accuracy and ``nll`` enables report-level perplexity.
    An optional ``vision_mask`` identifies valid image/video positions in the
    model's forward inputs, aligned with logits [B, T]. Its count is reported
    independently of the supervision mask and does not change the objective.
    With no supervised tokens, returns zero loss and zero count. All selected
    labels must be valid vocabulary indices. No shifting happens here.
    """
    if targets.ndim != 2 or logits.ndim != 3:
        raise ValueError(
            "Expected logits [batch, time, vocab] and targets [batch, time]."
        )
    if logits.shape[:-1] != targets.shape:
        raise ValueError("Logits and targets must have matching batch/time dimensions.")
    if vision_mask is not None and vision_mask.shape != targets.shape:
        raise ValueError("vision_mask must align with logits batch/time dimensions.")
    if mask is None:
        mask = mx.ones(targets.shape, dtype=mx.bool_)
    elif mask.shape != targets.shape:
        raise ValueError("mask must have the same shape as targets.")
    else:
        mask = mask.astype(mx.bool_)
    if ignore_index is not None:
        mask = mx.logical_and(mask, targets != ignore_index)

    safe_targets = mx.where(mask, targets, 0)
    token_loss = token_cross_entropy(logits, safe_targets, ignore_index=None)
    ntokens = mask.sum()
    denominator = mx.maximum(ntokens.astype(mx.float32), 1.0)
    loss = mx.where(mask, token_loss, 0).sum() / denominator
    correct = mx.logical_and(mx.argmax(logits, axis=-1) == safe_targets, mask).sum()
    return loss, {
        "weight": ntokens,
        "num_tokens": ntokens,
        "num_vision_tokens": (
            mx.array(0, dtype=mx.int32)
            if vision_mask is None
            else vision_mask.astype(mx.bool_).sum()
        ),
        "token_accuracy": correct.astype(mx.float32) / denominator,
        "nll": loss,
    }


def _extract_logits(output: Any) -> mx.array:
    """
    Extract logits from either a raw MLX array or a model output object.
    """
    logits = getattr(output, "logits", output)

    if not isinstance(logits, mx.array):
        raise TypeError(
            "The STT model must return an mx.array or an object containing "
            f"an mx.array in `.logits`, but got {type(output)!r}."
        )

    return logits


def _transcript_mask(
    targets: mx.array,
    lengths: mx.array,
) -> mx.array:
    """
    Build the transcript supervision mask.

    ``lengths`` has shape ``[batch, 2]`` and stores inclusive positions in the
    original unshifted token sequence:

        lengths[:, 0] = first supervised target-token position
        lengths[:, 1] = last supervised target-token position

    Because ``targets = token_ids[:, 1:]``, target index zero corresponds to
    original token position one.
    """
    if lengths.ndim != 2 or lengths.shape[1] != 2:
        raise ValueError(
            f"Expected lengths with shape [batch_size, 2], but got {lengths.shape}."
        )

    if lengths.shape[0] != targets.shape[0]:
        raise ValueError(
            "The lengths batch dimension must match the targets batch "
            f"dimension, but got {lengths.shape[0]} and {targets.shape[0]}."
        )

    positions = mx.arange(
        1,
        targets.shape[1] + 1,
    )[None, :]

    return mx.logical_and(
        positions >= lengths[:, 0:1],
        positions <= lengths[:, 1:2],
    )


def stt_cross_entropy(
    model: nn.Module,
    batch: dict[str, mx.array],
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Transcript-only next-token cross-entropy for autoregressive ASR models.

    Required batch fields:

        token_ids:
            Combined prompt, audio-placeholder, and transcript token IDs.

        input_features:
            Preprocessed audio features.

        feature_attention_mask:
            Valid audio-feature positions.

        lengths:
            Inclusive transcript supervision boundaries with shape [batch, 2].

    Returns:
        ``(loss, metrics)`` including ``weight`` and ``num_tokens``.
    """
    required_keys = {
        "token_ids",
        "input_features",
        "feature_attention_mask",
        "lengths",
    }
    missing_keys = required_keys.difference(batch)

    if missing_keys:
        raise KeyError(
            "Missing required STT batch fields: " + ", ".join(sorted(missing_keys))
        )

    token_ids = batch["token_ids"]

    if token_ids.ndim != 2:
        raise ValueError(
            "Expected token_ids with shape [batch_size, sequence_length], "
            f"but got {token_ids.shape}."
        )

    if token_ids.shape[1] < 2:
        raise ValueError("STT cross-entropy requires at least two token positions.")

    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]

    output = model(
        input_ids=inputs,
        input_features=batch["input_features"],
        feature_attention_mask=batch["feature_attention_mask"],
    )
    logits = _extract_logits(output)

    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "STT model logits must align with shifted token targets. "
            f"Got logits.shape={logits.shape} and targets.shape={targets.shape}."
        )

    mask = _transcript_mask(
        targets=targets,
        lengths=batch["lengths"],
    )

    return cross_entropy(
        logits=logits,
        targets=targets,
        mask=mask,
    )
