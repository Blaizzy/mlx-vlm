"""Attach native MTP weights as trainable children of an SFT target model."""

import mlx.core as mx
from mlx.utils import tree_flatten


def attach_mtp_head(model, head):
    """Attach an unbound Qwen3.5 dense/MoE MTP head before training setup.

    Load ``head`` with ``load_drafter`` from the matching original checkpoint or
    a split MTP directory. The registered ``mtp_head`` child participates in
    differentiation and trainable-weight checkpoints alongside the target.
    """
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
        Qwen3_5MTPDraftModel,
    )

    config = model.config
    model_type = (
        config.get("model_type") if isinstance(config, dict) else config.model_type
    )
    if model_type not in {
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    } or not isinstance(head, Qwen3_5MTPDraftModel):
        raise ValueError(
            "SFT MTP currently supports Qwen3.5 dense/MoE native heads only."
        )
    if head._input_embed is not None:
        raise ValueError("Attach a fresh, unbound MTP head for training.")
    target = model.language_model.args
    for name in ("hidden_size", "vocab_size", "tie_word_embeddings", "num_experts"):
        if getattr(target, name, None) != getattr(head.config.text_config, name, None):
            raise ValueError(f"MTP head and target disagree on {name}.")
    if any(
        not mx.issubdtype(value.dtype, mx.floating)
        for _, value in tree_flatten(head.parameters())
    ):
        raise ValueError(
            "SFT MTP requires a floating-point head; load an unquantized MTP checkpoint."
        )
    model.mtp_head = head
    return model
