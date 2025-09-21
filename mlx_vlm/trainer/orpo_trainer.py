# Copyright © 2024 MLX-VLM

import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from .trainer import TrainingArgs
from .utils import grad_checkpoint, Colors, get_learning_rate

@dataclass
class OrpoTrainingArgs(TrainingArgs):
    beta: float = field(default=0.1, metadata={"help": "Exponential moving average parameter for reward normalization"})


def get_logps(model, inputs, targets, lengths, train_on_completions=False, assistant_id=77091):
    outputs = model(inputs)
    logits = outputs.logits[:, :-1].astype(mx.float32)
    targets = targets[:, 1:]
    _, seq_len = targets.shape
    steps = mx.arange(seq_len)[None, :]
    base_mask = steps < lengths[:, None]
    if train_on_completions:
        eq = (inputs[:, :-1] == assistant_id)
        idxs = mx.arange(seq_len)[None, :]
        last_ass_idx = mx.where(eq, idxs, mx.full(idxs.shape, -1)).max(axis=1)
        comp_mask = steps > last_ass_idx[:, None]
        mask = base_mask & comp_mask
    else:
        mask = base_mask
    
    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    logp_seq_avg = (log_probs * mask).sum(-1) / mask.sum(-1)
    logits_mean = logits.sum() / mask.sum()
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
):
    chosen_logps = chosen_logps * preference_scores

    # Stable log-odds computation
    log_odds = chosen_logps - rejected_logps
    ratio = nn.log_sigmoid(log_odds)
    loss = -beta * ratio

    # Reward estimation
    chosen_reward = beta * chosen_logps
    rejected_reward = beta * rejected_logps
    reward = mx.stack([mx.mean(chosen_reward), mx.mean(rejected_reward)])

    num_tokens = chosen_masks.sum() + rejected_masks.sum()

    metrics = {
        "accuracies": mx.mean((chosen_reward > rejected_reward).astype(mx.float32)),
        "margins": mx.mean(chosen_reward - rejected_reward),
        "policy_chosen_logps": mx.mean(chosen_logps),
        "policy_rejected_logps": mx.mean(rejected_logps),
        "chosen_logits_mean": chosen_logits_mean,
        "rejected_logits_mean": rejected_logits_mean,
    }

    mx.clear_cache()
    return mx.mean(loss), reward, num_tokens, metrics