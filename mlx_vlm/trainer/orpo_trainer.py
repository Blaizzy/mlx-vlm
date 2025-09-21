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
from .utils import grad_checkpoint, Colors, get_learning_rate, save_adapter

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


def iterate_batches(dataset, batch_size, max_seq_length, train=False):
    indices = list(range(len(dataset)))
    if len(dataset) < batch_size:
        raise ValueError(f"Dataset must have at least {batch_size} examples")

    offset, step = mx.distributed.init().rank(), mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by number of workers")

    batch_indices = [
        indices[i + offset : i + offset + batch_size : step]
        for i in range(0, len(indices) - batch_size + 1, batch_size)
    ]

    while True:
        order = np.random.permutation(len(batch_indices)) if train else range(len(batch_indices))
        for b in order:
            items = [dataset[idx] for idx in batch_indices[b]]

            chosen_lengths = [min(len(x["chosen"]["input_ids"]), max_seq_length) for x in items]
            rejected_lengths = [min(len(x["rejected"]["input_ids"]), max_seq_length) for x in items]

            max_chosen_len = min(max(chosen_lengths), max_seq_length)
            max_rejected_len = min(max(rejected_lengths), max_seq_length)

            pad_to = 32
            padded_chosen_len = 1 + pad_to * ((max_chosen_len + pad_to - 1) // pad_to)
            padded_chosen_len = min(padded_chosen_len, max_seq_length)

            padded_rejected_len = 1 + pad_to * ((max_rejected_len + pad_to - 1) // pad_to)
            padded_rejected_len = min(padded_rejected_len, max_seq_length)

            chosen_input_ids_batch = np.zeros((len(items), padded_chosen_len), dtype=np.int32)
            chosen_attention_mask_batch = np.zeros((len(items), padded_chosen_len), dtype=np.int32)

            rejected_input_ids_batch = np.zeros((len(items), padded_rejected_len), dtype=np.int32)
            rejected_attention_mask_batch = np.zeros((len(items), padded_rejected_len), dtype=np.int32)

            for i, item in enumerate(items):
                chosen_arr = np.array(item["chosen"]["input_ids"]).reshape(-1)
                Lc = min(len(chosen_arr), padded_chosen_len)
                chosen_input_ids_batch[i, :Lc] = chosen_arr[:Lc]

                if "attention_mask" in item["chosen"]:
                    chosen_mask = np.array(item["chosen"]["attention_mask"]).reshape(-1)
                    chosen_attention_mask_batch[i, :Lc] = chosen_mask[:Lc]
                else:
                    chosen_attention_mask_batch[i, :Lc] = 1

                rejected_arr = np.array(item["rejected"]["input_ids"]).reshape(-1)
                Lr = min(len(rejected_arr), padded_rejected_len)
                rejected_input_ids_batch[i, :Lr] = rejected_arr[:Lr]

                if "attention_mask" in item["rejected"]:
                    rejected_mask = np.array(item["rejected"]["attention_mask"]).reshape(-1)
                    rejected_attention_mask_batch[i, :Lr] = rejected_mask[:Lr]
                else:
                    rejected_attention_mask_batch[i, :Lr] = 1

            chosen_batch = {
                "input_ids": mx.array(chosen_input_ids_batch),
                "attention_mask": mx.array(chosen_attention_mask_batch),
            }
            rejected_batch = {
                "input_ids": mx.array(rejected_input_ids_batch),
                "attention_mask": mx.array(rejected_attention_mask_batch),
            }

            yield (chosen_batch, rejected_batch)
        if not train:
            break
        mx.clear_cache()


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=orpo_loss,
    train_on_completions=False,
    assistant_id=77091,
): # TODO updat this with the new iterate batches and orpo loss return report
    """
    Evaluate the model on validation dataset.
    """
    model.eval()
    total_loss = mx.array(0.0)
    total_tokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, (chosen_batch, rejected_batch) in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Calculating loss...",
        total=min(len(dataset) // batch_size, num_batches) if num_batches != -1 else len(dataset) // batch_size,
    ):
        chosen_inputs = chosen_batch["input_ids"][:, :-1]
        chosen_targets = chosen_batch["input_ids"][:, 1:]
        chosen_lengths = chosen_batch["attention_mask"].sum(axis=1)

        rejected_inputs = rejected_batch["input_ids"][:, :-1]
        rejected_targets = rejected_batch["input_ids"][:, 1:]
        rejected_lengths = rejected_batch["attention_mask"].sum(axis=1)

        chosen_logps, chosen_logits_mean = get_logps(
            model,
            chosen_inputs,
            chosen_targets,
            chosen_lengths,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        )
        rejected_logps, rejected_logits_mean = get_logps(
            model,
            rejected_inputs,
            rejected_targets,
            rejected_lengths,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        )

        preference_scores = mx.ones_like(chosen_logps) # TODO: change this if preference scores are available

        losses, reward, num_tokens, metrics = loss_fn(
            chosen_logps,
            chosen_logits_mean,
            rejected_logps,
            rejected_logits_mean,
            chosen_lengths,
            rejected_lengths,
            preference_scores,
            beta=0.1,
        )

        total_loss += losses * num_tokens
        total_tokens += num_tokens
        mx.eval(total_loss, total_tokens)

    total_loss = mx.distributed.all_sum(total_loss, stream=mx.cpu)
    total_tokens = mx.distributed.all_sum(total_tokens, stream=mx.cpu)

    mx.clear_cache()

    return (total_loss / mx.maximum(total_tokens, 1)).item()