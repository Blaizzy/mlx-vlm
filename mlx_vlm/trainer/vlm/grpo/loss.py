"""Completion-only scoring and stable on-policy GRPO/GSPO objectives."""

import mlx.core as mx

from mlx_vlm.trainer.losses.cross_entropy import token_cross_entropy
from mlx_vlm.trainer.vlm.sft.runtime import _model_type, forward_vlm_logits


@mx.compile
def grpo_objective(
    logps,
    ref_logps,
    mask,
    advantages,
    *,
    beta=0.0,
    epsilon=1e-4,
    epsilon_high=None,
    max_tokens=512,
    importance_sampling_level="token",
    grpo_loss_type="grpo",
):
    logps = mx.where(mask, logps.astype(mx.float32), 0)
    ref_logps = mx.where(mask, mx.stop_gradient(ref_logps.astype(mx.float32)), 0)
    # The fixed reference is NOT the generating policy. Each rollout is used once.
    log_ratio = logps - mx.stop_gradient(logps)
    if importance_sampling_level == "sequence":
        log_ratio = log_ratio.sum(axis=1, keepdims=True) / mx.maximum(
            mask.sum(axis=1, keepdims=True), 1
        )
    elif importance_sampling_level != "token":
        raise ValueError("Unknown importance_sampling_level.")
    ratio = mx.exp(log_ratio)
    high = epsilon if epsilon_high is None else epsilon_high
    advantage = mx.stop_gradient(advantages)[:, None]
    per_token = -mx.minimum(
        ratio * advantage, mx.clip(ratio, 1 - epsilon, 1 + high) * advantage
    )
    delta = ref_logps - logps
    bounded = mx.minimum(delta, 20.0)
    kl = mx.maximum(mx.expm1(bounded) - bounded, 0)
    if beta:
        per_token += beta * kl
    counts = mask.sum(axis=1)
    total = mx.maximum(counts.sum(), 1)
    losses = mx.where(mask, per_token, 0).sum(axis=1)
    if grpo_loss_type == "grpo":
        loss = (losses / mx.maximum(counts, 1)).mean()
    elif grpo_loss_type == "bnpo":
        loss = losses.sum() / total
    elif grpo_loss_type == "dr_grpo":
        loss = losses.sum() / (logps.shape[0] * max_tokens)
    else:
        raise ValueError("Unknown grpo_loss_type.")
    low_clip = (ratio < 1 - epsilon) & (advantage < 0)
    high_clip = (ratio > 1 + high) & (advantage > 0)
    return loss, {
        "kl": (mx.where(mask, kl, 0).sum(axis=1) / mx.maximum(counts, 1)).mean(),
        "kl_clip_ratio": ((delta > 20.0) & mask).sum() / total,
        "clip_ratio": ((low_clip | high_clip) & mask).sum() / total,
    }


def completion_logps(model, batch, start, chunk_size=128):
    """Preserve full causal context; normalize only completion target positions.

    Qwen3.5's conventional body/head permits projecting completion hidden states
    in bounded vocabulary chunks. Other architectures retain their public forward.
    """
    ids, attention = batch["input_ids"], batch["attention_mask"]
    targets = ids[:, start + 1 :]
    if _model_type(model) == "qwen3_5":
        media = {
            k: v for k, v in batch.items() if k not in {"input_ids", "attention_mask"}
        }
        pixels = media.pop("pixel_values", None)
        embedding = model.get_input_embeddings(
            ids[:, :-1], pixels, mask=attention[:, :-1], **media
        )
        lm = model.language_model
        hidden = lm.model(
            ids[:, :-1],
            inputs_embeds=embedding.inputs_embeds,
            mask=attention[:, :-1],
            position_ids=embedding.position_ids,
        )[:, start:, :]
        head = (
            lm.model.embed_tokens.as_linear
            if lm.args.tie_word_embeddings
            else lm.lm_head
        )
        scores = []
        for position in range(0, targets.shape[1], chunk_size):
            logits = head(hidden[:, position : position + chunk_size]).astype(
                mx.float32
            )
            scores.append(
                -token_cross_entropy(
                    logits, targets[:, position : position + chunk_size]
                )
            )
        return mx.concatenate(scores, axis=1)
    logits = forward_vlm_logits(model, ids[:, :-1], attention[:, :-1], batch)
    return -token_cross_entropy(logits[:, start:], targets)
