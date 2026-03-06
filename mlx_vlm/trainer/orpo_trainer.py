# Copyright © 2026 MLX-VLM

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map
from tqdm import tqdm

from .utils import Colors, grad_checkpoint, save_adapter
from .sft_trainer import TrainingArgs


@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = field(default=0.1, metadata={"help": "Exponential moving average parameter for reward normalization"})
    eps: float = field(default=1e-8, metadata={"help": "Small constant for numerical stability in log calculations"})


def get_logps(model, batch, train_on_completions=False, assistant_id=77091):
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    shifted_input_ids = input_ids[:, :-1]
    shifted_attention_mask = attention_mask[:, :-1]
    targets = input_ids[:, 1:]

    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    outputs = model(shifted_input_ids, pixel_values, shifted_attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    def align_logits_with_targets(logits, targets):
        if logits.shape[1] < targets.shape[1]:
            pad_length = targets.shape[1] - logits.shape[1]
            pad_width = ((0, 0), (0, pad_length), (0, 0))
            return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
        if logits.shape[1] > targets.shape[1]:
            return logits[:, -targets.shape[1] :, :]
        return logits

    logits = align_logits_with_targets(logits, targets)

    lengths = mx.sum(shifted_attention_mask, axis=1)
    lengths = mx.minimum(lengths, shifted_input_ids.shape[1])
    steps = mx.arange(shifted_input_ids.shape[1])[None, :]
    base_mask = steps < lengths[:, None]

    if train_on_completions:
        assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
        input_ids_np = np.array(input_ids)
        for row_idx, row in enumerate(input_ids_np):
            positions = np.where(row == assistant_id)[0]
            if positions.size > 0:
                assistant_response_index[row_idx] = positions[0]

        assistant_mask = steps <= mx.array(assistant_response_index).reshape(-1, 1)
        mask = mx.where(assistant_mask, mx.zeros_like(base_mask), base_mask)
    else:
        mask = base_mask

    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    mask_f = mask.astype(log_probs.dtype)
    token_counts = mx.maximum(mask_f.sum(-1), 1)
    logp_seq_avg = (log_probs * mask_f).sum(-1) / token_counts
    logits_mean = logits.sum() / mx.maximum(mask_f.sum(), 1)
    return logp_seq_avg, logits_mean


def orpo_loss(
    chosen_logps,
    chosen_logits_mean,
    rejected_logps,
    rejected_logits_mean,
    chosen_masks,
    rejected_masks,
    preference_scores,
    beta: float = 0.1,
    eps: float = 1e-8,
):
    # ORPO uses log-odds ratio: log(p/(1-p))_chosen - log(p/(1-p))_rejected.
    # Clamp to keep exp/log1p numerically stable near probability 1.
    chosen_logps = mx.minimum(chosen_logps, -eps)
    rejected_logps = mx.minimum(rejected_logps, -eps)

    chosen_log_odds = chosen_logps - mx.log1p(-mx.exp(chosen_logps))
    rejected_log_odds = rejected_logps - mx.log1p(-mx.exp(rejected_logps))
    log_odds_ratio = chosen_log_odds - rejected_log_odds

    # ORPO objective: chosen NLL anchor + preference odds-ratio term.
    sft_term = -chosen_logps
    pref_term = -nn.log_sigmoid(log_odds_ratio)
    loss = sft_term + beta * pref_term

    if preference_scores is not None:
        loss = loss * preference_scores

    # Track rewards from log-odds (higher is better).
    chosen_reward = beta * chosen_log_odds
    rejected_reward = beta * rejected_log_odds
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    num_tokens = chosen_masks.sum() + rejected_masks.sum()

    metrics = {
        "accuracies": mx.mean((log_odds_ratio > 0).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_chosen_logps": mx.mean(chosen_logps),
        "policy_rejected_logps": mx.mean(rejected_logps),
        "sft_term": mx.mean(sft_term),
        "orpo_pref_term": mx.mean(pref_term),
        "chosen_logits_mean": chosen_logits_mean,
        "rejected_logits_mean": rejected_logits_mean,
    }

    mx.clear_cache()
    return mx.mean(loss), reward, num_tokens, metrics