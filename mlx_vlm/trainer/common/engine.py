"""One MLX training loop for all modalities and objectives."""

import time
from itertools import islice

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from mlx_vlm.trainer.common.callbacks import TrainingCallback
from mlx_vlm.trainer.common.config import CoreTrainingArgs
from mlx_vlm.trainer.common.metrics import (
    MetricAccumulator,
    format_metric_report,
    normalize_loss_output,
    report_loss_metrics,
)
from mlx_vlm.trainer.common.task import TrainingTask
from mlx_vlm.trainer.common.utils import (
    apply_final_accumulated_gradients,
    apply_gradient_step,
    save_trainable_weights,
)


def evaluate(
    model, dataset, task: TrainingTask, args: CoreTrainingArgs, *, return_metrics=False
):
    """Evaluate loss and dynamic metrics, preserving the caller's model mode.

    Evaluation compute time measures synchronized forward/loss work, excluding
    dataset preparation. Throughput counts policy inputs, not reference-model
    forwards or gradient-checkpoint recomputation.
    """
    if dataset is None or len(dataset) == 0:
        raise ValueError("Evaluation requires a non-empty dataset.")
    training = model.training
    accumulator = MetricAccumulator()
    started, compute_time = time.perf_counter(), 0.0
    try:
        model.eval()
        iterator = task.batches(dataset, train=False)
        for batch in islice(
            iterator, None if args.val_batches == -1 else args.val_batches
        ):
            start = time.perf_counter()
            loss, metrics = normalize_loss_output(task.objective(model, batch))
            accumulator.add(loss, metrics)
            mx.eval(accumulator.state())
            compute_time += time.perf_counter() - start
        values = accumulator.compute()
        if values["weight"] <= 0:
            raise ValueError("Evaluation produced no supervised tokens or examples.")
        info = report_loss_metrics(values, "val")
        info["val_time"] = time.perf_counter() - started
        info["val_compute_time"] = compute_time
        for key, value in values.items():
            if key.startswith("num_"):
                info[f"val_{key[4:]}_per_second"] = value / max(compute_time, 1e-8)
        return info if return_metrics else info["val_loss"]
    finally:
        model.train(training)


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: CoreTrainingArgs,
    task: TrainingTask,
    training_callback: TrainingCallback | None = None,
    seed: int | None = None,
    *,
    log_history: list[dict] | None = None,
) -> dict:
    """Optimize microbatches; return the final training metrics.

    Evaluate the graph once per microstep, release temporary gradients before
    evaluation, and synchronize Python metrics only at reporting intervals.
    ``iters`` counts microsteps; optimizer schedules count actual updates.
    """
    args.validate()
    if len(train_dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")
    if val_dataset is not None and len(val_dataset) == 0:
        raise ValueError("Validation dataset must be non-empty when provided.")
    world = mx.distributed.init()
    rank = world.rank()
    if args.batch_size % world.size():
        raise ValueError("Global batch size must be divisible by world_size.")
    if args.grad_checkpoint:
        if task.checkpoint is None:
            raise ValueError("This task has no gradient checkpointing implementation.")
        task.checkpoint(model)
    if seed is not None:
        mx.random.seed(seed)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state, mx.random.state]
    value_and_grad = nn.value_and_grad(model, task.loss)

    def step(batch, previous, update, *extra):
        return apply_gradient_step(
            model,
            optimizer,
            value_and_grad,
            batch,
            previous,
            update=update,
            accumulation_steps=args.gradient_accumulation_steps,
            loss_args=extra,
        )

    if args.compile:
        step = mx.compile(step, inputs=state + task.state, outputs=state)
    iterator = iter(task.batches(train_dataset, train=True, seed=seed))
    save = task.save_weights or save_trainable_weights
    model.train()
    gradients = None
    accumulator = MetricAccumulator()
    elapsed, data_elapsed, report_steps, trained, optimizer_step = 0.0, 0.0, 0, 0, 0
    compute_total, data_total, reported_updates = 0.0, 0.0, 0
    cumulative_counts = {}
    started = time.perf_counter()
    metrics = {"iteration": 0, "optimizer_step": 0}
    progress = tqdm(range(1, args.iters + 1), desc="Training", disable=rank != 0)
    for iteration in progress:
        data_start = time.perf_counter()
        batch = next(iterator)
        extra = task.loss_args(batch) if task.loss_args is not None else ()
        data_elapsed += time.perf_counter() - data_start
        start = time.perf_counter()
        update = iteration % args.gradient_accumulation_steps == 0
        loss, batch_metrics, gradients = step(batch, gradients, update, *extra)
        optimizer_step += int(update)
        accumulator.add(loss, batch_metrics)
        del loss, batch_metrics, batch, extra
        mx.eval(state, accumulator.state(), gradients)
        report_steps += 1
        if iteration == args.iters:
            partial = apply_final_accumulated_gradients(
                model, optimizer, gradients, iteration, args.gradient_accumulation_steps
            )
            optimizer_step += int(partial > 0)
            gradients = None
        elapsed += time.perf_counter() - start
        if iteration % args.steps_per_report == 0 or iteration == args.iters:
            values = accumulator.compute()
            total_weight = values["weight"]
            if total_weight <= 0:
                raise ValueError(
                    "Training batches contain no supervised tokens or examples."
                )
            trained += int(total_weight)
            compute_total += elapsed
            data_total += data_elapsed
            duration = max(elapsed, 1e-8)
            wall_time = time.perf_counter() - started
            learning_rate = optimizer.learning_rate
            learning_rate = float(
                learning_rate.item()
                if hasattr(learning_rate, "item")
                else learning_rate
            )
            metrics = {
                "iteration": iteration,
                "optimizer_step": optimizer_step,
                **report_loss_metrics(values, "train"),
                "learning_rate": learning_rate,
                "iterations_per_second": report_steps / duration,
                "optimizer_steps_per_second": (optimizer_step - reported_updates)
                / duration,
                f"{task.unit}_per_second": total_weight / duration,
                f"trained_{task.unit}": trained,
                "step_time": elapsed / report_steps,
                "data_time_per_step": data_elapsed / report_steps,
                "train_time": compute_total,
                "data_time": data_total,
                "elapsed_time": wall_time,
                "remaining_time": wall_time / iteration * (args.iters - iteration),
                "progress": iteration / args.iters,
                "peak_memory": mx.get_peak_memory() / 1e9,
                "active_memory": mx.get_active_memory() / 1e9,
                "cache_memory": mx.get_cache_memory() / 1e9,
            }
            for key, value in values.items():
                if key.startswith("num_"):
                    cumulative_counts[key] = cumulative_counts.get(key, 0) + int(value)
                    metrics[f"total_{key[4:]}"] = cumulative_counts[key]
                    metrics[f"{key[4:]}_per_second"] = value / duration
            if "num_tokens" in values:
                metrics["trained_tokens"] = cumulative_counts["num_tokens"]
                metrics["tokens_per_second"] = values["num_tokens"] / duration
            if "num_processed_tokens" in values:
                processed = values["num_processed_tokens"]
                padding = values.get("num_padded_tokens", 0)
                sequences = values.get("num_sequences", 0)
                metrics.update(
                    {
                        "processed_tokens": cumulative_counts["num_processed_tokens"],
                        "processed_tokens_per_second": processed / duration,
                        "end_to_end_tokens_per_second": processed
                        / max(elapsed + data_elapsed, 1e-8),
                        "padding_fraction": padding / max(processed + padding, 1),
                        "average_sequence_length": processed / max(sequences, 1),
                        "sequences_per_second": sequences / duration,
                    }
                )
            if rank == 0:
                progress.set_postfix(
                    loss=f"{metrics['train_loss']:.4f}", lr=f"{learning_rate:.2e}"
                )
                tqdm.write(format_metric_report(metrics, "Training"))
                if log_history is not None:
                    log_history.append(metrics.copy())
                if training_callback is not None:
                    training_callback.on_train_loss_report(metrics.copy())
            accumulator = MetricAccumulator()
            elapsed, data_elapsed, report_steps = 0.0, 0.0, 0
            reported_updates = optimizer_step
        if (
            val_dataset is not None
            and args.steps_per_eval is not None
            and (iteration % args.steps_per_eval == 0 or iteration == args.iters)
        ):
            info = {
                "iteration": iteration,
                **evaluate(model, val_dataset, task, args, return_metrics=True),
            }
            metrics.update(info)
            if rank == 0:
                tqdm.write(format_metric_report(info, "Validation"))
                if log_history is not None:
                    log_history.append(info.copy())
                if training_callback is not None:
                    training_callback.on_val_loss_report(info.copy())
        if iteration % args.steps_per_save == 0 and rank == 0:
            save(model, args.adapter_file, iteration=iteration)
        if (
            args.clear_cache_threshold
            and mx.get_cache_memory() > args.clear_cache_threshold
        ):
            mx.clear_cache()
    if rank == 0:
        save(model, args.adapter_file)
    metrics["elapsed_time"] = time.perf_counter() - started
    return metrics
