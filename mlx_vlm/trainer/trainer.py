# Copyright © 2024 MLX-VLM

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


def _squeeze_leading_batch_dim(value):
    if isinstance(value, mx.array) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


@dataclass
class TrainingArgs:
    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(
        default=25,
        metadata={
            "help": "Number of validation batches, -1 uses the entire validation set."
        },
    )
    steps_per_report: int = field(
        default=10,
        metadata={"help": "Number of training steps between loss reporting."},
    )
    steps_per_eval: int = field(
        default=200, metadata={"help": "Number of training steps between validations."}
    )
    steps_per_save: int = field(
        default=100, metadata={"help": "Save the model every number steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(
        default="adapters.safetensors",
        metadata={"help": "Save/load path for the trained adapter weights."},
    )
    grad_checkpoint: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing to reduce memory use."},
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate."},
    )
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping value."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps for learning rate."},
    )
    min_learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Minimum learning rate after decay."},
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Fine-tune the full model instead of adapters."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of steps to accumulate gradients before updating."},
    )


def vision_language_loss_fn(
    model, batch, train_on_completions=False, assistant_id=77091
):
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    if train_on_completions:
        weight_mask = mx.ones_like(attention_mask)

        assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
        input_ids_np = np.array(input_ids)
        for row_idx, row in enumerate(input_ids_np):
            positions = np.where(row == assistant_id)[0]
            if positions.size > 0:
                assistant_response_index[row_idx] = positions[0]

        range_matrix = mx.repeat(
            mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
        )
        assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(-1, 1)
        weight_mask = mx.where(assistant_mask, mx.zeros_like(weight_mask), weight_mask)[
            :, 1:
        ]
    else:
        weight_mask = None

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    lengths = mx.sum(attention_mask, axis=1)

    labels = batch["input_ids"][:, 1:]

    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)

    def align_logits_with_labels(logits, labels):
        if logits.shape[1] < labels.shape[1]:
            pad_length = labels.shape[1] - logits.shape[1]
            pad_width = ((0, 0), (0, pad_length), (0, 0))
            return mx.pad(logits, pad_width, mode="constant", constant_values=-100)
        elif logits.shape[1] > labels.shape[1]:
            return logits[:, -labels.shape[1] :, :]
        return logits

    logits = align_logits_with_labels(logits, labels)

    seq_len = input_ids.shape[1]
    lengths = mx.minimum(lengths, seq_len)
    length_mask = mx.arange(seq_len)[None, :] < lengths[:, None]

    ce = (
        nn.losses.cross_entropy(
            logits,
            labels,
            weights=weight_mask,
        )
        * length_mask
    )
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks

    return (ce * length_mask).sum() / length_mask.sum()


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
            lengths = [min(len(x["input_ids"]), max_seq_length) for x in items]

            max_len = min(max(lengths), max_seq_length)
            pad_to = 32
            padded_len = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
            padded_len = min(padded_len, max_seq_length)

            input_ids_batch = np.zeros((len(items), padded_len), dtype=np.int32)
            attention_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)

            for i, item in enumerate(items):
                arr = np.array(item["input_ids"]).reshape(-1)
                L = min(len(arr), padded_len)
                input_ids_batch[i, :L] = arr[:L]

                if "attention_mask" in item:
                    mask = np.array(item["attention_mask"]).reshape(-1)
                    attention_mask_batch[i, :L] = mask[:L]
                else:
                    attention_mask_batch[i, :L] = 1

            pixel_values_batch = None
            if "pixel_values" in items[0] and items[0]["pixel_values"] is not None:
                pixel_values_batch = mx.stack(
                    [_squeeze_leading_batch_dim(item["pixel_values"]) for item in items]
                )

            batch = {
                "input_ids": mx.array(input_ids_batch),
                "attention_mask": mx.array(attention_mask_batch),
                "pixel_values": pixel_values_batch,
            }

            extra_keys = [
                k for k in items[0]
                if k not in ("input_ids", "attention_mask", "pixel_values")
            ]
            for k in extra_keys:
                vals = [_squeeze_leading_batch_dim(item[k]) for item in items]
                if isinstance(vals[0], mx.array):
                    try:
                        batch[k] = mx.stack(vals)
                    except Exception:
                        batch[k] = vals[0]  # fallback for non-stackable
                else:
                    batch[k] = vals[0]

            yield batch
        if not train:
            break


def evaluate(
    model,
    dataset,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss_fn=vision_language_loss_fn,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Evaluate the model on validation dataset.
    """
    model.eval()
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    loss_fn_partial = partial(
        loss_fn, train_on_completions=train_on_completions, assistant_id=assistant_id
    )

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
        # Calculate number of tokens for averaging
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full(
                (batch["input_ids"].shape[0],), batch["input_ids"].shape[1]
            )

        ntoks = lengths.sum()
        losses = loss_fn_partial(model, batch)

        all_losses += losses * ntoks
        ntokens += ntoks
        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

    mx.clear_cache()
    return (all_losses / mx.maximum(ntokens, 1)).item()


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset=None,
    args: TrainingArgs = TrainingArgs(),
    loss_fn=vision_language_loss_fn,
    train_on_completions=False,
    assistant_id=77091,
):
    """
    Main training function for vision-language models.
    """
    # Set memory limit if using Metal
    if mx.metal.is_available():
        device_info = mx.device_info()
        max_working_set_size = device_info.get("max_recommended_working_set_size")
        if max_working_set_size is not None:
            mx.set_wired_limit(max_working_set_size)
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
        for module in model.children().values():
            if hasattr(module, "layers"):
                grad_checkpoint(module.layers[0])

    grad_accum_steps = args.gradient_accumulation_steps
    if grad_accum_steps < 1 and args:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    # Create loss function with partial application
    loss_fn_partial = partial(
        loss_fn, train_on_completions=train_on_completions, assistant_id=assistant_id
    )

    state = [model.state, optimizer.state, mx.random.state]

    def step(batch, prev_grad, do_update):
        # Calculate number of tokens for metrics
        if "attention_mask" in batch:
            lengths = batch["attention_mask"].sum(axis=1)
        else:
            lengths = mx.full(
                (batch["input_ids"].shape[0],), batch["input_ids"].shape[1]
            )

        toks = lengths.sum()
        lvalue, grad = loss_value_and_grad(model, batch)

        # Gradient clipping
        if args.grad_clip is not None:
            grad = tree_map(lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grad)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, toks, grad

    # Create value and grad function
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)

    # Training metrics
    model.train()
    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    grad_accum = None

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
        tic = time.perf_counter()

        # Validation (only if a validation dataset is provided)
        if val_dataset is not None and (
            it == 1 or it % args.steps_per_eval == 0 or it == args.iters
        ):
            tic_val = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn_partial,
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
        lvalue, toks, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        mx.clear_cache()
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
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
                    f"Iter {it}: Train loss {Colors.OKGREEN}{train_loss:.8f}{Colors.ENDC}, "
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