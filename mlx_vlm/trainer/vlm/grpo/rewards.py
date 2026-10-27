"""Batched reward callbacks and CPU group normalization (MLX-LM-LoRA style)."""

import re

import mlx.core as mx
import numpy as np


def accuracy_reward(*, completions, answer, **kwargs):
    """Exact answer match, accepting an optional <answer>...</answer> wrapper."""
    result = []
    for completion, target in zip(completions, answer):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        text = match.group(1) if match else completion
        result.append(
            None if target is None else float(text.strip() == str(target).strip())
        )
    return result


def calculate_rewards_and_advantages(
    reward_funcs,
    prompts,
    completions,
    answers,
    indices,
    *,
    metadata=None,
    reward_weights=None,
    epsilon=1e-4,
):
    if not reward_funcs or not completions or len(indices) != len(completions):
        raise ValueError(
            "Reward callbacks and aligned non-empty completions are required."
        )
    weights = np.asarray(
        [1.0] * len(reward_funcs) if reward_weights is None else reward_weights,
        dtype=np.float64,
    )
    if weights.shape != (len(reward_funcs),) or not np.isfinite(weights).all():
        raise ValueError("Finite reward_weights must match reward_funcs.")
    columns, metrics = [], {}
    for i, callback in enumerate(reward_funcs):
        raw = callback(
            prompts=prompts,
            completions=completions,
            answer=answers,
            types=[row.get("type") for row in (metadata or [{}] * len(completions))],
            metadata=metadata or [{}] * len(completions),
        )
        raw = [None] * len(completions) if raw is None else raw
        if len(raw) != len(completions):
            raise ValueError(
                "Each reward callback must return one reward per completion."
            )
        values = np.asarray([np.nan if r is None else float(r) for r in raw])
        if np.isinf(values).any():
            raise ValueError("Reward callbacks returned an infinite reward.")
        valid = values[~np.isnan(values)]
        name = getattr(callback, "__name__", f"reward_{i}")
        metrics[f"reward_{i}_{name}"] = float(valid.mean()) if valid.size else 0.0
        metrics[f"reward_{i}_coverage"] = valid.size / values.size
        columns.append(values)
    values = np.stack(columns, axis=1)
    if np.isnan(values).all(axis=1).any():
        raise ValueError("All reward functions returned None or NaN for a completion.")
    rewards = (np.nan_to_num(values, nan=0.0) * weights).sum(axis=1)
    if not np.isfinite(rewards).all():
        raise ValueError("Weighted rewards must be finite.")
    advantages = np.zeros_like(rewards)
    stds = []
    for index in sorted(set(indices)):
        positions = np.flatnonzero(np.asarray(indices) == index)
        group = rewards[positions]
        std = group.std()
        advantages[positions] = (group - group.mean()) / (std + epsilon)
        stds.append(std)
    metrics.update(
        reward_mean=rewards.mean(),
        reward_std=rewards.std(),
        grouped_reward_std=np.mean(stds),
    )
    if any(
        not np.isfinite(v) or abs(v) > np.finfo(np.float32).max
        for v in metrics.values()
    ):
        raise ValueError("Reward statistics must be representable in float32.")
    return mx.array(advantages, dtype=mx.float32), {
        key: mx.array(value, dtype=mx.float32) for key, value in metrics.items()
    }
