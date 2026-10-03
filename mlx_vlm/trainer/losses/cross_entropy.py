from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


def cross_entropy(
    logits: mx.array,
    targets: mx.array,
    mask: mx.array,
) -> tuple[mx.array, mx.array]:
    """Compute masked next-token cross-entropy and its supervised-token count."""
    per_token_loss = nn.losses.cross_entropy(logits, targets)
    token_count = mask.sum()
    loss = (per_token_loss * mask).sum() / mx.maximum(token_count, 1)
    return loss, token_count


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
            "Expected lengths with shape [batch_size, 2], " f"but got {lengths.shape}."
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
) -> tuple[mx.array, mx.array]:
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
        ``(loss, num_supervised_tokens)``.
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
