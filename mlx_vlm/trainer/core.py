import json
import math
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
