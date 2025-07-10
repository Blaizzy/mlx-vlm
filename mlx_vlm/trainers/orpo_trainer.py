import time
from dataclasses import dataclass, field
from pathlib import Path
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .sft_trainer import TrainingArgs, average_gradients, grad_checkpoint
from .callback import TrainingCallback


@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for ORPO training."}
    )
    reward_scaling: float = field(
        default=1.0,
        metadata={"help": "Reward scaling factor for ORPO training, not implemented."},
    )


def get_logps(model: nn.Module, tokens, mask, **kwargs):
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    output = model(inputs, mask=mask, pixel_values=None, **kwargs)
    logits = output.logits
    log_probs = -nn.losses.cross_entropy(logits, targets, reduction="none")
    mask = mask[:, :-1]
    seq_lengths = mask.sum(-1)
    logp_seq_avg = (log_probs * mask).sum(-1) / seq_lengths
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

    log_odds = chosen_logps - rejected_logps
    ratio = nn.log_sigmoid(log_odds)
    loss = -beta * ratio

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


def iterate_orpo_batches(dataset, batch_size, max_seq_length, train=False):
    """Batch iterator for ORPO with preference scores"""
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]["chosen"]))

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("Batch size must be divisible by number of workers")

    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = (
            np.random.permutation(len(batch_idx)) if train else range(len(batch_idx))
        )
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]

            chosen_lengths = [len(x["chosen"]) for x in batch]
            rejected_lengths = [len(x["rejected"]) for x in batch]
            max_length = min(
                max(max(chosen_lengths), max(rejected_lengths)), max_seq_length
            )
            pad_to = 8
            max_length_in_batch = pad_to * ((max_length + pad_to - 1) // pad_to)

            batch_size_per_device = batch_size // step
            chosen_arr = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.int32
            )
            rejected_arr = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.int32
            )
            chosen_masks = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.float32
            )
            rejected_masks = np.zeros(
                (batch_size_per_device, max_length_in_batch), np.float32
            )

            preference_scores = np.array(
                [x.get("preference_score", 1.0) for x in batch], np.float32
            )
            pixel_values = [x.get("pixel_values") for x in batch]
            pixel_values = mx.stack(pixel_values) if all(p is not None for p in pixel_values) else None

            for j in range(batch_size_per_device):
                chosen_length = min(chosen_lengths[j], max_length_in_batch)
                rejected_length = min(rejected_lengths[j], max_length_in_batch)

                chosen_arr[j, :chosen_length] = batch[j]["chosen"][:chosen_length]
                chosen_masks[j, :chosen_length] = 1.0
                rejected_arr[j, :rejected_length] = batch[j]["rejected"][
                    :rejected_length
                ]
                rejected_masks[j, :rejected_length] = 1.0

            yield (
                mx.array(chosen_arr),
                mx.array(rejected_arr),
                mx.array(chosen_masks),
                mx.array(rejected_masks),
                mx.array(preference_scores),
                pixel_values
            )

        if not train:
            break


def train_orpo(
    model: nn.Module,
    processor,
    optimizer,
    dataset,
    args: ORPOTrainingArgs = ORPOTrainingArgs(),
    loss_fn: callable = orpo_loss,
    training_callback: TrainingCallback = None,
    clip_gradients=None
):
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        chosen, rejected, chosen_masks, rejected_masks, preference_scores, pixel_values = batch

        chosen_logps, chosen_logits_mean = get_logps(model, chosen, chosen_masks, pixel_values=pixel_values)
        rejected_logps, rejected_logits_mean = get_logps(model, rejected, rejected_masks, pixel_values=pixel_values)

        (lvalue, reward, toks, metrics), grad = loss_value_and_grad(
            chosen_logps,
            chosen_logits_mean,
            rejected_logps,
            rejected_logits_mean,
            chosen_masks,
            rejected_masks,
            preference_scores=preference_scores,
        )

        if clip_gradients is not None:
            grad = tree_map(
                lambda g: mx.clip(g, -clip_gradients, clip_gradients), grad
            )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return lvalue, toks, metrics

    def loss_wrapper(
        chosen_logps, chosen_logits_mean, rejected_logps, rejected_logits_mean, chosen_masks, rejected_masks, preference_scores
    ):
        return loss_fn(
            chosen_logps=chosen_logps,
            chosen_logits_mean=chosen_logits_mean,
            rejected_logps=rejected_logps,
            rejected_logits_mean=rejected_logits_mean,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            preference_scores=preference_scores,
            beta=args.beta,
        )

    loss_value_and_grad = nn.value_and_grad(model, loss_wrapper)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    accumulated_metrics = {
        "accuracies": 0,
        "margins": 0,
        "policy_rejected_logps": 0,
        "policy_chosen_logps": 0,
        "rejected_logits_mean": 0,
        "chosen_logits_mean": 0,
    }
    start = time.perf_counter()

    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_orpo_batches(
            dataset=dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        lvalue, toks, metrics = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1

        mx.clear_cache()

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses).item() / (steps * world_size)
            train_rewards = [
                r / (steps * world_size)
                for r in mx.distributed.all_sum(rewards).tolist()
            ]
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"\nIter {it}: "
                    f"loss {train_loss:.3f}, "
                    f"chosen_r {train_rewards[0]:.3f}, "
                    f"rejected_r {train_rewards[1]:.3f}, "
                    f"acc {avg_metrics['accuracies']:.3f}, "
                    f"margin {avg_metrics['margins']:.3f}, "
                    f"lr {learning_rate:.3e}, "
                    f"it/s {it_sec:.3f}, "
                    f"tok/s {tokens_sec:.3f}, "
                    f"peak_mem {peak_mem:.3f}GB"
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "train_chosen_reward": train_rewards[0],
                    "train_rejected_reward": train_rewards[1],
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            rewards = mx.zeros((2,))
            n_tokens = 0
            steps = 0
            accumulated_metrics = {k: 0 for k in accumulated_metrics}
            start = time.perf_counter()

        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
