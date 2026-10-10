"""Shared trainer primitives that are independent of task and modality.

These primitives preserve the functional SFT/ORPO recipes. The notebook API's
configuration, model preparation, and optimization lifecycle live in ``common``.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


not_supported_for_training = {"gemma3n", "qwen3_omni"}


@dataclass
class TrainingArgs:
    """Configuration shared by all iterative training recipes."""

    batch_size: int = field(default=4, metadata={"help": "Minibatch size."})
    iters: int = field(default=100, metadata={"help": "Iterations to train for."})
    val_batches: int = field(default=25, metadata={"help": "Validation batches."})
    steps_per_report: int = field(default=10, metadata={"help": "Report cadence."})
    steps_per_eval: int = field(default=200, metadata={"help": "Evaluation cadence."})
    steps_per_save: int = field(default=100, metadata={"help": "Checkpoint cadence."})
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length."}
    )
    adapter_file: str = field(default="adapters.safetensors")
    grad_checkpoint: bool = field(default=False)
    learning_rate: float = field(default=1e-5)
    grad_clip: float | None = field(default=1.0)
    warmup_steps: int = field(default=100)
    min_learning_rate: float = field(default=1e-6)
    full_finetune: bool = field(default=False)
    gradient_accumulation_steps: int = field(default=1)


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


def get_learning_rate(
    iters: int,
    step: int,
    warmup_steps: int,
    learning_rate: float,
    min_learning_rate: float,
):
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)

    progress = (step - warmup_steps) / (iters - warmup_steps)
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
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    if parts[-1].isdigit():
        module[int(parts[-1])] = new_module
    else:
        setattr(module, parts[-1], new_module)


def count_parameters(model):
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
    def nparams(m):
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            return m.weight.size * (32 // m.bits)
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )
    total_p = sum(nparams(m) for _, m in leaf_modules) / 10**6
    trainable_p = (
        sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    )

    print(
        f"#trainable params: {trainable_p} M || all params: {total_p} M || trainable%: {(trainable_p * 100 / total_p):.3f}%"
    )


def save_adapter(model: nn.Module, adapter_file: Union[str, Path]):
    """Save adapter weights and config."""
    path = Path(adapter_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save adapter config if available
    if hasattr(model, "config") and hasattr(model.config, "lora"):
        with open(path.parent / "adapter_config.json", "w") as f:
            json.dump(model.config.lora, f, indent=2)

    # Save weights
    flattened_tree = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(adapter_file), dict(flattened_tree))
