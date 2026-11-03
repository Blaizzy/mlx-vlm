"""Modality-independent GRPO/GSPO policy objectives over token log-probabilities."""

import mlx.core as mx


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
