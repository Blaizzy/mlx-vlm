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

from .sft_trainer import TrainingArgs, _squeeze_leading_batch_dim
from .utils import Colors, grad_checkpoint, save_adapter


@dataclass
class ORPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1,
        metadata={
            "help": "Exponential moving average parameter for reward normalization"
        },
    )
    eps: float = field(
        default=1e-8,
        metadata={"help": "Small constant for numerical stability in log calculations"},
    )


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


def _pad_and_collate(items, prefix, max_seq_length):
    """Pad and collate input_ids, attention_mask, pixel_values for a given prefix."""
    id_key = f"{prefix}_input_ids"
    mask_key = f"{prefix}_attention_mask"
    pv_key = f"{prefix}_pixel_values"

    lengths = [min(len(x[id_key]), max_seq_length) for x in items]
    max_len = min(max(lengths), max_seq_length)
    pad_to = 32
    padded_len = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
    padded_len = min(padded_len, max_seq_length)

    input_ids_batch = np.zeros((len(items), padded_len), dtype=np.int32)
    attention_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)

    for i, item in enumerate(items):
        arr = np.array(item[id_key]).reshape(-1)
        L = min(len(arr), padded_len)
        input_ids_batch[i, :L] = arr[:L]

        if mask_key in item:
            mask = np.array(item[mask_key]).reshape(-1)
            attention_mask_batch[i, :L] = mask[:L]
        else:
            attention_mask_batch[i, :L] = 1

    pixel_values_batch = None
    if pv_key in items[0] and items[0][pv_key] is not None:
        pixel_values_batch = mx.stack(
            [_squeeze_leading_batch_dim(item[pv_key]) for item in items]
        )

    result = {
        "input_ids": mx.array(input_ids_batch),
        "attention_mask": mx.array(attention_mask_batch),
        "pixel_values": pixel_values_batch,
    }

    skip = {id_key, mask_key, pv_key}
    for k in items[0]:
        if k.startswith(f"{prefix}_") and k not in skip:
            vals = [_squeeze_leading_batch_dim(item[k]) for item in items]
            if isinstance(vals[0], mx.array):
                try:
                    result[k.removeprefix(f"{prefix}_")] = mx.stack(vals)
                except Exception:
                    result[k.removeprefix(f"{prefix}_")] = vals[0]
            else:
                result[k.removeprefix(f"{prefix}_")] = vals[0]

    return result


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
        order = (
            np.random.permutation(len(batch_indices))
            if train
            else range(len(batch_indices))
        )
        for b in order:
            items = [dataset[idx] for idx in batch_indices[b]]

            chosen_batch = _pad_and_collate(items, "chosen", max_seq_length)
            rejected_batch = _pad_and_collate(items, "rejected", max_seq_length)

            yield {
                "chosen": chosen_batch,
                "rejected": rejected_batch,
            }
        if not train:
            break


def evaluate_orpo(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=orpo_loss,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Evaluate the model on validation dataset.
    """
    model.eval()
    total_loss = mx.array(0.0)
    total_tokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in tqdm(
        zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
        ),
        desc="Calculating loss...",
        total=(
            min(len(dataset) // batch_size, num_batches)
            if num_batches != -1
            else len(dataset) // batch_size
        ),
    ):
        chosen_batch = batch["chosen"]
        rejected_batch = batch["rejected"]

        chosen_logps, chosen_logits_mean = get_logps(
            model,
            chosen_batch,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        )
        rejected_logps, rejected_logits_mean = get_logps(
            model,
            rejected_batch,
            train_on_completions=train_on_completions,
            assistant_id=assistant_id,
        )

        losses, reward, num_tokens, metrics = loss_fn(
            chosen_logps,
            chosen_logits_mean,
            rejected_logps,
            rejected_logits_mean,
            chosen_batch["attention_mask"],
            rejected_batch["attention_mask"],
            beta=0.1,
        )

        total_loss += losses * num_tokens
        total_tokens += num_tokens
        mx.eval(total_loss, total_tokens)

    total_loss = mx.distributed.all_sum(total_loss, stream=mx.cpu)
    total_tokens = mx.distributed.all_sum(total_tokens, stream=mx.cpu)

    mx.clear_cache()

    return (total_loss / mx.maximum(total_tokens, 1)).item()


def train_orpo(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss_fn=orpo_loss,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Main training function for vision-language models.
    """
    # Set memory limit if using Metal
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    print(f"{Colors.HEADER}Starting training..., iterations: {args.iters}{Colors.ENDC}")

    # Initialize distributed training
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if val_dataset is None and rank == 0:
        print(
            f"{Colors.OKBLUE}No validation dataset provided — training will run without validation.{Colors.ENDC}"
        )

    # Enable gradient checkpointing if requested
    if args.grad_checkpoint:
        if hasattr(model, "layers"):
            grad_checkpoint(model.layers[0])

    # Compile the training step (like MLX-LM)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(chosen_batch, rejected_batch):
        def loss_fn_wrapper():
            chosen_logps, chosen_logits_mean = get_logps(
                model,
                chosen_batch,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
            )
            rejected_logps, rejected_logits_mean = get_logps(
                model,
                rejected_batch,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
            )
            losses, reward, num_tokens, metrics = loss_fn(
                chosen_logps,
                chosen_logits_mean,
                rejected_logps,
                rejected_logits_mean,
                chosen_batch["attention_mask"],
                rejected_batch["attention_mask"],
                beta=args.beta,
            )
            return losses, num_tokens

        (lvalue, toks), grad = nn.value_and_grad(model, loss_fn_wrapper)()

        # Gradient clipping
        if args.grad_clip is not None:
            grad = tree_map(lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grad)

        # Average gradients for distributed training
        grad = average_gradients(grad)

        # Update model
        optimizer.update(model, grad)

        return lvalue, toks

    # Training metrics
    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        chosen_batch = batch["chosen"]
        rejected_batch = batch["rejected"]
        tic = time.perf_counter()

        # Validation (only if a validation dataset is provided)
        if val_dataset is not None and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic_val = time.perf_counter()
            val_loss = evaluate_orpo(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
            )
            model.train()
            val_time = time.perf_counter() - tic_val

            if rank == 0:
                print(
                    f"{Colors.OKCYAN}Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s{Colors.ENDC}",
                    flush=True,
                )

            tic = time.perf_counter()

        # Training step
        lvalue, toks = step(chosen_batch, rejected_batch)
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)
        train_time += time.perf_counter() - tic

        # Report training metrics
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            n_tokens_total = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = (
                optimizer.learning_rate.item()
                if hasattr(optimizer.learning_rate, "item")
                else args.learning_rate
            )
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens_total) / train_time
            trained_tokens += n_tokens_total
            peak_mem = mx.get_peak_memory() / 1e9

            if rank == 0:
                print(
                    f"Iter {it}: Train loss {Colors.OKGREEN}{train_loss:.3f}{Colors.ENDC}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            # Reset metrics
            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        # Save checkpoint
        if it % args.steps_per_save == 0 and rank == 0:
            save_adapter(model, args.adapter_file)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            save_adapter(model, checkpoint)
            print(
                f"{Colors.OKBLUE}Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}.{Colors.ENDC}",
                flush=True,
            )

    # Save final weights
    if rank == 0:
        save_adapter(model, args.adapter_file)
        print(
            f"{Colors.OKGREEN}Saved final adapter weights to {args.adapter_file}.{Colors.ENDC}"
        )
