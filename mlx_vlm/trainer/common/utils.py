"""Shared configuration, batching, evaluation, and training helpers."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import wraps
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm

from mlx_vlm.trainer.common.metrics import normalize_loss_output


def pad_arrays(
    arrays: Sequence[np.ndarray],
    *,
    padding_value: int | float = 0,
    pad_to_multiple: int = 1,
    min_length: int = 1,
    max_length: int | None = None,
) -> np.ndarray:
    """Stack arrays, right-padding only the last axis without truncation.

    Leading dimensions and dtypes must match. A maximum length caps alignment
    padding, never input data. Suitable for token vectors and feature matrices.
    """
    if not arrays:
        raise ValueError("Cannot pad an empty collection.")
    if pad_to_multiple < 1 or min_length < 1:
        raise ValueError("Padding multiple and minimum length must be positive.")
    if max_length is not None and max_length < min_length:
        raise ValueError("max_length must be at least min_length.")
    first = arrays[0]
    if first.ndim < 1:
        raise ValueError("Arrays must have at least one dimension.")
    for array in arrays:
        if array.shape[:-1] != first.shape[:-1] or array.ndim != first.ndim:
            raise ValueError("Arrays must have matching non-time dimensions.")
        if array.dtype != first.dtype:
            raise ValueError("Arrays must have matching dtypes.")
    longest = max(array.shape[-1] for array in arrays)
    if max_length is not None and longest > max_length:
        raise ValueError(
            f"Input length {longest} exceeds max_length={max_length}; truncation is disabled."
        )
    length = max(
        min_length,
        ((longest + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple,
    )
    if max_length is not None:
        length = min(length, max_length)
    result = np.full(
        (len(arrays), *first.shape[:-1], length), padding_value, dtype=first.dtype
    )
    for row, array in enumerate(arrays):
        result[row, ..., : array.shape[-1]] = array
    return result


def iterate_batch_indices(
    size: int,
    batch_size: int,
    *,
    length_key: Callable[[int], int],
    train: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int | None = None,
    group_key: Callable[[int], Any] | None = None,
) -> Iterator[list[int]]:
    """Yield length-grouped local indices, dropping incomplete global batches.

    Training repeats with shuffled batch order; evaluation is finite and sorted.
    Each rank must use the same dataset, length key, and seed. Distributed runs
    default to seed 0 so independent process RNG states cannot mix global batches.
    Single-process runs without a seed retain NumPy's global RNG behavior.
    """
    if batch_size < 1 or world_size < 1:
        raise ValueError("batch_size and world_size must be positive.")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size).")
    if batch_size % world_size:
        raise ValueError("Global batch size must be divisible by world_size.")
    if size < batch_size:
        raise ValueError(
            f"Dataset must contain at least batch_size={batch_size} examples, got {size}."
        )
    groups = {}
    for index in range(size):
        groups.setdefault(group_key(index) if group_key else (), []).append(index)
    batches = []
    for group in sorted(groups):
        indices = sorted(groups[group], key=length_key)
        batches.extend(
            indices[start : start + batch_size]
            for start in range(0, len(indices) - batch_size + 1, batch_size)
        )
    if not batches:
        raise ValueError(
            "No complete batches fit within a single media type; reduce batch_size."
        )
    # Only the complete batches are needed for subsequent epochs.
    del groups, indices
    local_size = batch_size // world_size
    local_start = rank * local_size
    if seed is None and world_size > 1:
        seed = 0
    rng = np.random if seed is None else np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(batches)) if train else range(len(batches))
        for position in order:
            yield batches[int(position)][local_start : local_start + local_size]
        if not train:
            return


def evaluate_weighted_batches(
    model: nn.Module,
    batches: Iterable[Any],
    loss_fn: Callable[[nn.Module, Any], tuple[Any, Any]],
    num_batches: int,
    *,
    empty_error: str = "Evaluation produced no weighted examples.",
) -> float:
    """Evaluate batches and return a distributed weighted mean loss."""
    if num_batches == 0 or num_batches < -1:
        raise ValueError("num_batches must be positive or -1.")

    mx.distributed.init()
    model.eval()
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_weight = mx.array(0, dtype=mx.int32)

    for batch in islice(batches, None if num_batches == -1 else num_batches):
        batch_loss, metrics = normalize_loss_output(loss_fn(model, batch))
        batch_weight = metrics["weight"]
        total_loss += batch_loss * batch_weight
        total_weight += batch_weight
        mx.eval(total_loss, total_weight)

    total_loss = mx.distributed.all_sum(total_loss, stream=mx.cpu)
    total_weight = mx.distributed.all_sum(total_weight, stream=mx.cpu)
    if int(total_weight.item()) == 0:
        raise ValueError(empty_error)

    return float((total_loss / total_weight.astype(mx.float32)).item())


def resolve_args(
    args: Any,
    parser: argparse.ArgumentParser,
    defaults: Mapping[str, Any],
    *,
    config_name: str,
) -> SimpleNamespace:
    """Merge parser values, optional YAML values, and pipeline defaults.

    Pipeline-specific validation remains in each command entry point. Defaults
    are copied deeply so mutable configuration values cannot be shared between
    independent runs.
    """
    if isinstance(args, argparse.Namespace):
        values = vars(args).copy()
    elif isinstance(args, Mapping):
        values = vars(parser.parse_args([]))
        values.update(args)
    else:
        values = vars(parser.parse_args([]))

    config_path = values.get("config")
    if config_path:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("YAML configuration requires `pyyaml`.") from error
        with Path(config_path).open("r", encoding="utf-8") as file:
            config_values = yaml.safe_load(file) or {}
        if not isinstance(config_values, Mapping):
            raise ValueError(f"{config_name} trainer config must contain a mapping.")
        for key, value in config_values.items():
            if values.get(key) is None:
                values[key] = value

    for key, value in defaults.items():
        if values.get(key) is None:
            values[key] = copy.deepcopy(value)
    return SimpleNamespace(**values)


def grad_checkpoint(layer: nn.Module) -> None:
    """Checkpoint all instances of the layer class, once, as in MLX-LM-LoRA.

    The class-wide patch also affects future instances of the same layer type.
    """
    original = type(layer).__call__
    if getattr(original, "_vlm_checkpoint_enabled", False):
        return

    @wraps(original)
    def checkpointed(model, *args, **kwargs):
        def inner(parameters, *args, **kwargs):
            model.update(parameters)
            return original(model, *args, **kwargs)

        return mx.checkpoint(inner)(model.trainable_parameters(), *args, **kwargs)

    checkpointed._vlm_checkpoint_enabled = True
    type(layer).__call__ = checkpointed


def enable_gradient_checkpointing(layers: Sequence[nn.Module]) -> None:
    """Enable checkpointing for every distinct layer class in a layer stack."""
    if len(layers) == 0:
        raise ValueError(
            "Cannot enable gradient checkpointing on an empty layer stack."
        )
    for layer in layers:
        grad_checkpoint(layer)


def enable_model_gradient_checkpointing(model: nn.Module) -> None:
    """Checkpoint each transformer layer or block stack in the model."""
    seen = set()
    for _, module in model.named_modules():
        for name in ("layers", "blocks"):
            layers = getattr(module, name, None)
            if isinstance(layers, (list, tuple)) and layers and id(layers) not in seen:
                enable_gradient_checkpointing(layers)
                seen.add(id(layers))
    if not seen:
        raise ValueError(
            "Could not locate transformer layers or blocks for checkpointing."
        )


def accumulate_gradients(gradients, previous=None):
    """Add a microbatch's gradients to an optional accumulated gradient tree."""
    if previous is None:
        return gradients
    return tree_map(
        lambda current, accumulated: current + accumulated, gradients, previous
    )


def apply_accumulated_gradients(
    model: nn.Module, optimizer, gradients, steps: int
) -> None:
    """Average across workers and actual microsteps, then update the model."""
    if steps < 1:
        raise ValueError("Accumulation steps must be at least 1.")
    gradients = average_gradients(gradients)
    if steps > 1:
        gradients = tree_map(lambda value: value / steps, gradients)
    apply_optimizer_update(model, optimizer, gradients)


def apply_gradient_step(
    model: nn.Module,
    optimizer,
    loss_value_and_grad: Callable,
    batch: Any,
    previous_gradients: Any = None,
    *,
    update: bool,
    accumulation_steps: int,
    loss_args: Sequence[Any] = (),
) -> tuple[Any, Any, Any]:
    """Evaluate one batch, accumulate its gradient, and optionally update.

    ``loss_value_and_grad`` must return ``((loss, metrics), gradients)``. Extra
    positional arguments are passed to it after ``batch`` for trainers whose
    loss needs auxiliary inputs, such as guide embeddings.
    """
    output, gradients = loss_value_and_grad(model, batch, *loss_args)
    loss_value, metrics = normalize_loss_output(output)
    gradients = accumulate_gradients(gradients, previous_gradients)
    if update:
        apply_accumulated_gradients(model, optimizer, gradients, accumulation_steps)
        gradients = None
    return loss_value, metrics, gradients


def apply_final_accumulated_gradients(
    model: nn.Module,
    optimizer,
    gradients: Any,
    total_steps: int,
    accumulation_steps: int,
) -> int:
    """Apply a final incomplete accumulation and return its microstep count."""
    if total_steps < 0:
        raise ValueError("total_steps must be nonnegative.")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be at least 1.")

    partial_steps = total_steps % accumulation_steps
    if gradients is None or partial_steps == 0:
        return 0

    apply_accumulated_gradients(model, optimizer, gradients, partial_steps)
    mx.eval(model.state, optimizer.state)
    return partial_steps


def apply_optimizer_update(
    model: nn.Module,
    optimizer,
    gradients,
) -> None:
    """Portable optimizer update for MLX versions with Module update issues."""
    updated_parameters = optimizer.apply_gradients(
        gradients, model.trainable_parameters()
    )
    model.update(updated_parameters)


def save_trainable_weights(
    model: nn.Module,
    adapter_file: str | Path,
    iteration: Optional[int] = None,
) -> None:
    """Save trainable parameters and optionally a numbered training checkpoint.

    Call on rank zero. This is a training checkpoint, not a fused model export.
    """
    destination = Path(adapter_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destination), weights)
    config = getattr(model, "config", None)
    adapter_config = getattr(model, "_vlm_adapter_config", None)
    if adapter_config is None:
        adapter_config = (
            config.get("lora")
            if isinstance(config, Mapping)
            else getattr(config, "lora", None)
        )
    if adapter_config is not None:
        save_json(adapter_config, destination.parent / "adapter_config.json")

    if iteration is None:
        tqdm.write(f"Saved final trainable weights to {destination}.")
        return

    checkpoint = destination.with_name(f"{iteration:07d}_{destination.name}")
    mx.save_safetensors(str(checkpoint), weights)
    tqdm.write(
        f"Iter {iteration}: saved trainable weights to {destination} and {checkpoint}."
    )


def resolve_pad_token_id(tokenizer) -> int:
    token_id = tokenizer.pad_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("The tokenizer has neither pad_token_id nor eos_token_id.")
    return int(token_id)


def get_learning_rate(
    iters: int,
    step: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
):
    if iters < 1:
        raise ValueError("iters must be at least 1.")
    if warmup_steps < 0 or warmup_steps > iters:
        raise ValueError("warmup_steps must be between 0 and iters.")
    if learning_rate < 0 or min_learning_rate < 0:
        raise ValueError("learning rates must be nonnegative.")
    if step < warmup_steps:
        if warmup_steps == 0:
            return learning_rate
        return learning_rate * (step / warmup_steps)

    progress = (
        1.0 if iters == warmup_steps else (step - warmup_steps) / (iters - warmup_steps)
    )
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (learning_rate - min_learning_rate) * cosine_decay


def get_module_by_name(model, name):
    parts = name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model, name, new_module):
    parent, _, key = name.rpartition(".")
    module = get_module_by_name(model, parent) if parent else model
    if key.isdigit():
        module[int(key)] = new_module
    else:
        setattr(module, key, new_module)


def count_parameters(model):
    """Return parameter count in millions, accounting for quantized weights."""

    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6

    return total_p


def print_trainable_parameters(model):
    total_p = count_parameters(model)
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )

    print(
        f"trainable params: {trainable_p} M || all params: {total_p} M || trainable%: {(trainable_p * 100 / total_p):.3f}%"
    )


def calculate_iters(train_set: Sequence[Any], batch_size: int, epochs: int) -> int:
    """Calculate optimizer iterations for a finite number of dataset epochs."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    batches_per_epoch = math.ceil(len(train_set) / batch_size)
    iterations = epochs * batches_per_epoch
    tqdm.write(
        f"Calculated {iterations} iterations from {epochs} epoch(s) "
        f"(dataset size: {len(train_set)}, batch size: {batch_size})."
    )
    return iterations


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Write a deterministic, human-readable JSON configuration file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(dict(data), file, indent=2, sort_keys=True, default=str)
