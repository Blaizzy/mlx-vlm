# Copyright © 2026 MLX-VLM

import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_map
from tqdm import tqdm

from ...common.metrics import (
    MetricAccumulator,
    format_metric_report,
    normalize_loss_output,
    report_loss_metrics,
)
from ...core import Colors, TrainingArgs, grad_checkpoint, save_adapter
from ...losses.cross_entropy import cross_entropy
from .runtime import vlm_batch_metrics, vlm_vision_mask


def _squeeze_leading_batch_dim(value):
    if isinstance(value, mx.array) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def _flat_seq_len(value):
    """Return token sequence length for values shaped as (seq,) or (1, seq)."""
    return np.array(_squeeze_leading_batch_dim(value)).reshape(-1).shape[0]


def _collate_arrays(values):
    """Stack same-shaped arrays, or concatenate variable-length feature rows."""
    try:
        return mx.stack(values)
    except ValueError:
        first = values[0]
        if (
            isinstance(first, mx.array)
            and first.ndim > 1
            and all(
                isinstance(v, mx.array)
                and v.ndim == first.ndim
                and v.shape[1:] == first.shape[1:]
                for v in values
            )
        ):
            return mx.concatenate(values, axis=0)
        raise


def _collate_grid_thw(values):
    """Pack per-example image/video grids as (total_media, 3)."""
    rows = []
    for value in values:
        value = value if isinstance(value, mx.array) else mx.array(value)
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thw values with shape (num_media, 3), got {value.shape}"
            )
        rows.append(value)
    return mx.concatenate(rows, axis=0)


def _resolve_adapter_file(args: TrainingArgs) -> Path:
    adapter_file = getattr(args, "adapter_file", None)
    if adapter_file:
        return Path(adapter_file)

    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path:
        return Path(adapter_path) / "adapters.safetensors"

    return Path(TrainingArgs.__dataclass_fields__["adapter_file"].default)


def _model_type(model):
    model_type = getattr(model, "model_type", None)
    if model_type is not None:
        return model_type

    config = getattr(model, "config", None)
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def vision_language_loss_fn(
    model, batch, train_on_completions=False, assistant_id=77091
):
    pixel_values = batch["pixel_values"]
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    batch_size, seq_length = input_ids.shape

    input_ids = input_ids[:, :-1]
    attention_mask = attention_mask[:, :-1]

    lengths = mx.sum(attention_mask, axis=1)

    labels = batch["input_ids"][:, 1:]

    kwargs = {
        k: v
        for k, v in batch.items()
        if k not in ["input_ids", "pixel_values", "attention_mask", "completion_mask"]
    }

    model_attention_mask = (
        None if _model_type(model) == "gemma4_unified" else attention_mask
    )
    outputs = model(input_ids, pixel_values, model_attention_mask, **kwargs)
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
    loss_mask = length_mask

    if train_on_completions:
        if "completion_mask" in batch:
            loss_mask = loss_mask * batch["completion_mask"][:, 1:]
        else:
            completion_mask = mx.ones_like(batch["attention_mask"])

            assistant_response_index = np.full((batch_size,), -1, dtype=np.int32)
            input_ids_np = np.array(batch["input_ids"])
            for row_idx, row in enumerate(input_ids_np):
                positions = np.where(row == assistant_id)[0]
                if positions.size > 0:
                    assistant_response_index[row_idx] = positions[0]

            range_matrix = mx.repeat(
                mx.expand_dims(mx.arange(seq_length), 0), batch_size, axis=0
            )
            assistant_mask = range_matrix <= mx.array(assistant_response_index).reshape(
                -1, 1
            )
            completion_mask = mx.where(
                assistant_mask,
                mx.zeros_like(completion_mask),
                completion_mask,
            )
            loss_mask = loss_mask * completion_mask[:, 1:]

    loss, metrics = cross_entropy(
        logits, labels, loss_mask, vision_mask=vlm_vision_mask(model, batch)
    )
    return loss, {**metrics, **vlm_batch_metrics(batch)}


def _dataset_image_token_id(dataset):
    """The placeholder token id a vision dataset expands one image feature into.

    Returns ``None`` for text-only datasets, which are never image-aligned and so
    are unaffected by the truncation guard below.
    """
    config = getattr(dataset, "config", None)
    if not isinstance(config, dict):
        return None
    token_id = config.get("image_token_index") or config.get("image_token_id")
    return int(token_id) if token_id else None


def _drops_image_tokens(input_ids, image_token_id, max_seq_length):
    """True when truncating to ``max_seq_length`` would cut image placeholders.

    The model derives one image feature per placeholder token from
    ``pixel_values``, which truncation does not shrink. Losing placeholders
    therefore breaks the feature/token alignment the model asserts on.
    """
    if image_token_id is None or len(input_ids) <= max_seq_length:
        return False
    kept = int((input_ids[:max_seq_length] == image_token_id).sum())
    total = int((input_ids == image_token_id).sum())
    return kept != total


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

    image_token_id = _dataset_image_token_id(dataset)
    warned_indices = set()

    while True:
        order = (
            np.random.permutation(len(batch_indices))
            if train
            else range(len(batch_indices))
        )
        yielded_any = False
        for b in order:
            items = [dataset[idx] for idx in batch_indices[b]]

            # Drop examples whose image placeholders would not survive truncation:
            # pixel_values keeps every sub-image, so training on them would fail the
            # model's feature/token alignment check.
            if image_token_id is not None:
                kept = []
                for idx, item in zip(batch_indices[b], items):
                    input_ids = np.array(
                        _squeeze_leading_batch_dim(item["input_ids"])
                    ).reshape(-1)
                    if _drops_image_tokens(input_ids, image_token_id, max_seq_length):
                        if idx not in warned_indices:
                            warned_indices.add(idx)
                            total = int((input_ids == image_token_id).sum())
                            fits = int(
                                (input_ids[:max_seq_length] == image_token_id).sum()
                            )
                            logging.warning(
                                f"Skipping example {idx}: its {total} image tokens do "
                                f"not fit in max_seq_length={max_seq_length} (only "
                                f"{fits} would remain). Raise max_seq_length or use a "
                                "lower image resolution to train on this example."
                            )
                        continue
                    kept.append(item)
                if not kept:
                    # Every example here was skipped; move on rather than failing
                    # the run, since other batches may still be trainable.
                    continue
                items = kept

            lengths = [
                min(_flat_seq_len(x["input_ids"]), max_seq_length) for x in items
            ]

            max_len = min(max(lengths), max_seq_length)
            pad_to = 32
            padded_len = 1 + pad_to * ((max_len + pad_to - 1) // pad_to)
            padded_len = min(padded_len, max_seq_length)

            input_ids_batch = np.zeros((len(items), padded_len), dtype=np.int32)
            attention_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)
            has_completion_mask = any("completion_mask" in item for item in items)
            completion_mask_batch = np.zeros((len(items), padded_len), dtype=np.int32)

            for i, item in enumerate(items):
                arr = np.array(_squeeze_leading_batch_dim(item["input_ids"])).reshape(
                    -1
                )
                L = min(len(arr), padded_len)
                input_ids_batch[i, :L] = arr[:L]

                if "attention_mask" in item:
                    mask = np.array(
                        _squeeze_leading_batch_dim(item["attention_mask"])
                    ).reshape(-1)
                    attention_mask_batch[i, :L] = mask[:L]
                else:
                    attention_mask_batch[i, :L] = 1

                if "completion_mask" in item:
                    completion_mask = np.array(
                        _squeeze_leading_batch_dim(item["completion_mask"])
                    ).reshape(-1)
                    completion_mask_batch[i, :L] = completion_mask[:L]

            pixel_values_batch = None
            if "pixel_values" in items[0] and items[0]["pixel_values"] is not None:
                pixel_values_batch = _collate_arrays(
                    [_squeeze_leading_batch_dim(item["pixel_values"]) for item in items]
                )

            batch = {
                "input_ids": mx.array(input_ids_batch),
                "attention_mask": mx.array(attention_mask_batch),
                "pixel_values": pixel_values_batch,
            }
            if has_completion_mask:
                batch["completion_mask"] = mx.array(completion_mask_batch)

            extra_keys = [
                k
                for k in items[0]
                if k
                not in (
                    "input_ids",
                    "attention_mask",
                    "completion_mask",
                    "pixel_values",
                )
            ]
            for k in extra_keys:
                if k in ("image_grid_thw", "video_grid_thw"):
                    batch[k] = _collate_grid_thw([item[k] for item in items])
                    continue

                vals = [_squeeze_leading_batch_dim(item[k]) for item in items]
                if isinstance(vals[0], mx.array):
                    try:
                        batch[k] = _collate_arrays(vals)
                    except Exception:
                        batch[k] = vals[0]
                else:
                    batch[k] = vals[0]

            yielded_any = True
            yield batch

        if not yielded_any:
            raise ValueError(
                "No trainable examples: every example has more image tokens than "
                f"max_seq_length={max_seq_length} allows. Raise max_seq_length or "
                "lower the image resolution."
            )
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
    return_metrics=False,
):
    """
    Evaluate the model on validation dataset.
    """
    model.eval()
    accumulator = MetricAccumulator()

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
        loss, metrics = normalize_loss_output(loss_fn_partial(model, batch))
        accumulator.add(loss, metrics)
        mx.eval(accumulator.state())

    values = accumulator.compute()
    if values["weight"] <= 0:
        raise ValueError("Evaluation produced no supervised tokens or examples.")
    info = report_loss_metrics(values, "val")
    mx.clear_cache()
    return info if return_metrics else info["val_loss"]


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

    adapter_file = _resolve_adapter_file(args)

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
        output, grad = loss_value_and_grad(model, batch)
        lvalue, metrics = normalize_loss_output(output)

        # Gradient clipping
        if args.grad_clip is not None:
            grad = tree_map(lambda g: mx.clip(g, -args.grad_clip, args.grad_clip), grad)

        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            # Average grads across ranks entirely on the CPU stream and
            # materialize before any GPU-stream consumer, so the blocking ring
            # all-reduce never keeps a Metal command buffer in flight — a slow
            # peer would otherwise stall it past Metal's ~5s command-buffer
            # watchdog and crash the run. See issue #2179.
            if world_size > 1:
                with mx.stream(mx.cpu):
                    grad = average_gradients(grad, communication_stream=mx.cpu)
                mx.eval(grad)
            if grad_accum_steps > 1:
                grad = tree_map(lambda x: x / grad_accum_steps, grad)
            optimizer.update(model, grad)
            grad = None

        return lvalue, metrics, grad

    # Create value and grad function
    loss_value_and_grad = nn.value_and_grad(model, loss_fn_partial)

    # Training metrics
    model.train()
    accumulator = MetricAccumulator()
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
            val_info = evaluate(
                model=model,
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                loss_fn=loss_fn_partial,
                train_on_completions=train_on_completions,
                assistant_id=assistant_id,
                return_metrics=True,
            )
            model.train()
            val_time = time.perf_counter() - tic_val

            if rank == 0:
                val_info.update(iteration=it, val_time=val_time)
                print(format_metric_report(val_info, "Validation"), flush=True)

            tic = time.perf_counter()

        # Training step
        lvalue, metrics, grad_accum = step(
            batch,
            grad_accum,
            it % grad_accum_steps == 0,
        )
        mx.clear_cache()
        accumulator.add(lvalue, metrics)
        steps += 1
        mx.eval(state, accumulator.state(), grad_accum)
        train_time += time.perf_counter() - tic

        # Report training metrics
        if it % args.steps_per_report == 0 or it == args.iters:
            values = accumulator.compute()
            n_tokens_total = values.get("num_tokens", values["weight"])
            learning_rate = (
                optimizer.learning_rate.item()
                if hasattr(optimizer.learning_rate, "item")
                else args.learning_rate
            )
            trained_tokens += n_tokens_total
            info = {
                "iteration": it,
                **report_loss_metrics(values, "train"),
                "learning_rate": learning_rate,
                "iterations_per_second": steps / max(train_time, 1e-8),
                "tokens_per_second": n_tokens_total / max(train_time, 1e-8),
                "trained_tokens": trained_tokens,
                "peak_memory": mx.get_peak_memory() / 1e9,
            }
            if "num_processed_tokens" in values:
                info["processed_tokens_per_second"] = values[
                    "num_processed_tokens"
                ] / max(train_time, 1e-8)
            if rank == 0:
                print(format_metric_report(info, "Training"), flush=True)
            accumulator = MetricAccumulator()
            steps, train_time = 0, 0

        # Save checkpoint
        if it % args.steps_per_save == 0 and rank == 0:
            save_adapter(model, adapter_file)
            checkpoint = adapter_file.parent / f"{it:07d}_adapters.safetensors"
            save_adapter(model, checkpoint)
            print(
                f"{Colors.OKBLUE}Iter {it}: Saved adapter weights to "
                f"{adapter_file} and {checkpoint}.{Colors.ENDC}",
                flush=True,
            )

    # Save final weights
    if rank == 0:
        save_adapter(model, adapter_file)
        print(
            f"{Colors.OKGREEN}Saved final adapter weights to {adapter_file}.{Colors.ENDC}"
        )


@dataclass
class SFTTrainer:
    """Configure and run the functional supervised fine-tuning recipe."""

    model: nn.Module
    optimizer: Any
    train_dataset: Any
    val_dataset: Any = None
    args: TrainingArgs = field(default_factory=TrainingArgs)
    train_on_completions: bool = False
    assistant_id: int = 77091

    def fit(self):
        """Run training with the configured model, data, and options."""
        return train(
            model=self.model,
            optimizer=self.optimizer,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            args=self.args,
            train_on_completions=self.train_on_completions,
            assistant_id=self.assistant_id,
        )
