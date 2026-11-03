"""Optional VLM MTP auxiliary loss shared by SFT, DPO, ORPO, and GRPO."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

MTP_PREFIX = "mtp_head."


def mtp_head_dir(weights_file):
    """Return the drafter directory saved beside a trainable-weights file."""
    path = Path(weights_file)
    step, sep, _ = path.name.partition("_")
    return path.with_name(f"{step}_mtp" if sep and step.isdigit() else "mtp")


def split_mtp_weights(weights):
    """Split flat weights into ``(main, head)`` with the head prefix removed."""
    main = {k: v for k, v in weights.items() if not k.startswith(MTP_PREFIX)}
    head = {
        k.removeprefix(MTP_PREFIX): v
        for k, v in weights.items()
        if k.startswith(MTP_PREFIX)
    }
    return main, head


def save_mtp_head(head, weights, directory):
    """Write a trained MTP head as a standalone drafter loadable by ``load_drafter``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(directory / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    config = asdict(head.config)
    config["text_config"]["mtp_num_hidden_layers"] = len(head.layers)
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return directory


def attach_mtp_head(model, head, *, checkpoint_path=None):
    """Attach an unbound Qwen3.5 dense/MoE MTP head after base training setup.

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
            "Native MTP training currently supports Qwen3.5 dense/MoE heads only."
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
    if checkpoint_path is not None:
        from mlx_vlm.trainer.common.model import _read_checkpoint

        path, _ = _read_checkpoint(checkpoint_path, require_config=False)
        _, weights = split_mtp_weights(mx.load(str(path)))
        sidecar = mtp_head_dir(path) / "model.safetensors"
        if not weights and sidecar.is_file():
            weights = mx.load(str(sidecar))
        if weights:
            head.load_weights(list(weights.items()), strict=True)
    head.unfreeze()
    model.mtp_head = head
    return model


def vlm_mtp_loss(model, batch, *, output=None, loss_mask=None):
    """Return next-next-token CE and metrics for a policy's native MTP head.

    A head implements ``training_logits(tokens, hidden, attention_mask, model)``.
    Forwarding and causal shifting belong here, outside shared cross-entropy.
    An existing policy output/mask avoids another forward during SFT.
    """
    from mlx_vlm.trainer.losses.cross_entropy import cross_entropy
    from mlx_vlm.trainer.vlm.sft.runtime import _forward_vlm

    if not hasattr(model, "mtp_head"):
        raise ValueError(
            "MTP loss requires attach_mtp_head(model, head) before training."
        )
    ids, attention = batch["input_ids"], batch["attention_mask"]
    if ids.ndim != 2 or attention.shape != ids.shape:
        raise ValueError(
            "MTP input_ids and attention_mask must have shape [batch, time]."
        )
    if ids.shape[1] < 3:
        return mx.array(0.0), {"mtp_loss": mx.array(0.0), "num_mtp_tokens": mx.array(0)}
    if output is None:
        output = _forward_vlm(
            model, ids[:, :-1], attention[:, :-1], batch, return_hidden=True
        )
    hidden = (
        output.get("hidden_states")
        if isinstance(output, Mapping)
        else getattr(output, "hidden_states", None)
    )
    if not hidden or hidden[-1].shape[:2] != ids[:, :-1].shape:
        raise ValueError("MTP requires target hidden states aligned with input_ids.")
    if loss_mask is None:
        if "completion_mask" not in batch:
            raise ValueError("MTP auxiliary loss requires a completion_mask.")
        if batch["completion_mask"].shape != ids.shape:
            raise ValueError("completion_mask must have the same shape as input_ids.")
        loss_mask = mx.logical_and(attention[:, 1:], batch["completion_mask"][:, 1:])
    context_mask = mx.logical_and(attention[:, :-2], attention[:, 1:-1])
    logits = model.mtp_head.training_logits(
        ids[:, 1:-1], hidden[-1][:, :-1], context_mask, model
    )
    loss, metrics = cross_entropy(
        logits.astype(mx.float32),
        ids[:, 2:],
        mx.logical_and(loss_mask[:, 1:], context_mask),
    )
    return loss, {"mtp_loss": loss, "num_mtp_tokens": metrics["num_tokens"]}


def add_mtp_loss(model, batch, loss, metrics, weight, *, output=None, loss_mask=None):
    """Add optional MTP supervision, reusing the policy forward when available."""
    if weight:
        auxiliary, auxiliary_metrics = vlm_mtp_loss(
            model, batch, output=output, loss_mask=loss_mask
        )
        loss = loss + weight * auxiliary
        metrics.update(auxiliary_metrics)
    return loss, metrics
