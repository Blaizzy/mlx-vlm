"""Modality-independent DPO and ORPO objectives over sequence scores."""

import mlx.core as mx
import mlx.nn as nn


def dpo_objective(
    policy_chosen: mx.array,
    policy_rejected: mx.array,
    reference_chosen: mx.array,
    reference_rejected: mx.array,
    *,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    delta: float = 50.0,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Return canonical DPO per-pair losses and scalar reward metrics."""
    if beta <= 0:
        raise ValueError("DPO beta must be greater than zero.")
    if delta < 0:
        raise ValueError("DPOP delta must be nonnegative.")
    logits = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    if loss_type == "sigmoid":
        losses = -nn.log_sigmoid(beta * logits)
    elif loss_type == "hinge":
        losses = nn.relu(1.0 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    elif loss_type == "dpop":
        penalty = mx.maximum(
            mx.zeros_like(policy_chosen), reference_chosen - policy_chosen
        )
        losses = -nn.log_sigmoid(beta * logits) + delta * penalty
    else:
        raise ValueError("loss_type must be sigmoid, hinge, ipo, or dpop.")
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    pairs = mx.array(losses.size, dtype=mx.int32)
    return losses, {
        "weight": pairs,
        "num_pairs": pairs,
        "reward_accuracy": mx.mean(
            (chosen_reward > rejected_reward).astype(mx.float32)
        ),
        "reward_margin": mx.mean(chosen_reward - rejected_reward),
        "chosen_reward": mx.mean(chosen_reward),
        "rejected_reward": mx.mean(rejected_reward),
        "chosen_logps": mx.mean(policy_chosen),
        "rejected_logps": mx.mean(policy_rejected),
    }


def _log1mexp(log_probability: mx.array, eps: float) -> mx.array:
    """Stable ``log(1 - exp(x))`` for non-positive log probabilities."""
    value = mx.minimum(log_probability.astype(mx.float32), -eps)
    split = -0.6931471805599453
    upper = mx.maximum(value, split)
    lower = mx.minimum(value, split)
    upper_value = mx.log(-mx.expm1(upper))
    lower_value = mx.log1p(-mx.exp(lower))
    return mx.where(value > split, upper_value, lower_value)


def orpo_objective(
    chosen_logps: mx.array,
    rejected_logps: mx.array,
    *,
    beta: float = 0.1,
    eps: float = 1e-6,
) -> tuple[mx.array, dict[str, mx.array]]:
    """Return ORPO per-pair losses and scalar preference metrics."""
    if beta <= 0:
        raise ValueError("ORPO beta must be greater than zero.")
    if not 0 < eps < 1:
        raise ValueError("ORPO eps must be in the interval (0, 1).")
    chosen_logps = mx.nan_to_num(chosen_logps, nan=-1000.0, posinf=0.0, neginf=-1000.0)
    rejected_logps = mx.nan_to_num(
        rejected_logps, nan=-1000.0, posinf=0.0, neginf=-1000.0
    )
    chosen_log_odds = chosen_logps - _log1mexp(chosen_logps, eps)
    rejected_log_odds = rejected_logps - _log1mexp(rejected_logps, eps)
    log_odds_ratio = chosen_log_odds - rejected_log_odds
    preference_loss = -nn.log_sigmoid(log_odds_ratio)
    losses = -chosen_logps + beta * preference_loss
    pairs = mx.array(losses.size, dtype=mx.int32)
    return losses, {
        "weight": pairs,
        "num_pairs": pairs,
        "preference_accuracy": mx.mean((log_odds_ratio > 0).astype(mx.float32)),
        "odds_margin": mx.mean(log_odds_ratio),
        "chosen_nll": mx.mean(-chosen_logps),
        "preference_loss": mx.mean(preference_loss),
        "chosen_logps": mx.mean(chosen_logps),
        "rejected_logps": mx.mean(rejected_logps),
    }
