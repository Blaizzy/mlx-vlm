"""Modality-independent multi-token prediction (next-next-token) objective."""

import mlx.core as mx

from mlx_vlm.trainer.losses.cross_entropy import cross_entropy


def mtp_objective(logits, input_ids, loss_mask, context_mask):
    """Return CE of MTP-head ``logits`` against tokens two positions ahead.

    ``logits`` predict ``input_ids[:, 2:]`` from context positions ``[:, :-2]``;
    ``loss_mask`` is the policy's next-token mask aligned with ``input_ids[:, 1:]``.
    """
    loss, metrics = cross_entropy(
        logits.astype(mx.float32),
        input_ids[:, 2:],
        mx.logical_and(loss_mask[:, 1:], context_mask),
    )
    return loss, {"mtp_loss": loss, "num_mtp_tokens": metrics["num_tokens"]}
