"""Prepare upstream MLX models without importing optional libraries at startup."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.nn as nn


def _record_adapter_config(model, config):
    model_config = getattr(model, "config", None)
    if isinstance(model_config, dict):
        model_config["lora"] = config
    elif model_config is not None:
        model_config.lora = config
    model._vlm_adapter_config = config


def _read_checkpoint(path, *, require_config=True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"The checkpoint path does not exist: {path}")
    if path.is_dir():
        weights = path / "adapters.safetensors"
        if not weights.is_file():
            weights = path / "weights.safetensors"
    else:
        weights = path
    if not weights.is_file():
        raise FileNotFoundError(
            f"No checkpoint weights found in {path}. Pass the saved .safetensors file directly."
        )
    config_path = weights.parent / "adapter_config.json"
    if not config_path.is_file():
        if require_config:
            raise FileNotFoundError(
                f"Missing adapter metadata: {config_path}. "
                "Keep adapter_config.json beside the saved weights."
            )
        return weights, None
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {config_path}")
    return weights, config


def _quantization_config(model):
    import mlx.nn as nn

    return {
        name: {
            "group_size": layer.group_size,
            "bits": layer.bits,
            "mode": getattr(layer, "mode", "affine"),
        }
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding))
    }


def _restore_quantization(model, config):
    """Recreate the saved base representation before attaching adapters."""
    import mlx.nn as nn

    modules = dict(model.named_modules())
    pending = {}
    for name, parameters in config.items():
        layer = modules.get(name)
        if isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            actual = {
                "group_size": layer.group_size,
                "bits": layer.bits,
                "mode": getattr(layer, "mode", "affine"),
            }
            if actual != parameters:
                raise ValueError(
                    f"Quantization differs for {name}: expected {parameters}, got {actual}. "
                    "Load the original base checkpoint or a floating-point base."
                )
        elif isinstance(layer, (nn.Linear, nn.Embedding)):
            pending[name] = parameters
        else:
            raise ValueError(
                f"Cannot restore quantization for missing or unsupported layer {name}."
            )
    if pending:
        nn.quantize(model, class_predicate=lambda name, _: pending.get(name, False))


def _quantize_model(model, bits, group_size):
    import mlx.nn as nn

    config = _quantization_config(model)
    parameters = {"bits": bits, "group_size": group_size, "mode": "affine"}
    for name in config:
        config[name] = parameters
    for name, layer in model.named_modules():
        if (
            isinstance(layer, (nn.Linear, nn.Embedding))
            and layer.weight.shape[-1] % group_size == 0
        ):
            config[name] = parameters
    if not config:
        raise ValueError(
            f"No layers support quantization with group size {group_size}. "
            "Use a compatible group size or leave quantization_bits=None."
        )
    _restore_quantization(model, config)


def _dequantize_model(model):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    replacements = []
    for name, layer in model.named_modules():
        if not isinstance(layer, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        weight = mx.dequantize(
            layer.weight,
            layer.scales,
            layer.biases,
            group_size=layer.group_size,
            bits=layer.bits,
            mode=getattr(layer, "mode", "affine"),
        )
        if isinstance(layer, nn.QuantizedLinear):
            replacement = nn.Linear(
                weight.shape[1], weight.shape[0], bias="bias" in layer
            )
            if "bias" in layer:
                replacement.bias = layer.bias
        else:
            replacement = nn.Embedding(weight.shape[0], weight.shape[1])
        replacement.weight = weight
        replacements.append((name, replacement))
    if replacements:
        model.update_modules(tree_unflatten(replacements))
    return model


def prepare_model_for_training(
    model: nn.Module,
    *,
    train_type: str = "lora",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    quantization_bits: int | None = None,
    quantization_group_size: int = 64,
    checkpoint_path: str | Path | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Make a loaded model ready to pass directly to Trainer.

    Args:
        model: A fresh base model loaded by its upstream library.
        train_type: "lora", "dora", or "full". Adapter modes freeze the base;
            full training dequantizes packed weights and unfreezes everything.
        lora_rank: Adapter rank for a new adapter.
        lora_alpha: Adapter scaling numerator; the update scale is alpha / rank.
        lora_dropout: Adapter input dropout for a new adapter.
        target_modules: Optional Linear suffixes or qualified module names.
            Defaults to the language/embedding transformer's Linear layers.
        quantization_bits: Optionally quantize eligible Linear/Embedding layers
            before attaching adapters. None keeps the loaded representation,
            or restores the saved representation when loading a checkpoint.
        quantization_group_size: Group size for optional affine quantization.
        checkpoint_path: Saved weights file or folder. Adapter checkpoints
            restore their saved LoRA/DoRA settings; full checkpoints restore
            all model weights. Optimizer state is not restored.
        verbose: Print trainable parameter counts.

    Returns:
        The same model, modified in place and in training mode.
    """
    if train_type not in {"lora", "dora", "full"}:
        raise ValueError("train_type must be 'lora', 'dora', or 'full'.")
    if quantization_bits is not None:
        if quantization_bits not in {4, 8}:
            raise ValueError("quantization_bits must be 4 or 8, or None.")
        if quantization_group_size not in {32, 64, 128}:
            raise ValueError("quantization_group_size must be 32, 64, or 128.")
        if train_type == "full":
            raise ValueError(
                "Full training needs floating-point weights; leave quantization_bits=None."
            )

    import mlx.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("Pass the loaded MLX model, not its tokenizer or processor.")
    if any(
        hasattr(layer, "lora_a") or type(layer).__name__ == "LoRaLayer"
        for _, layer in model.named_modules()
    ):
        raise ValueError(
            "The model already has adapters. Load a fresh base model before preparation."
        )
    weights, config = (None, None)
    if checkpoint_path is not None:
        weights, config = _read_checkpoint(
            checkpoint_path, require_config=train_type != "full"
        )

    if train_type == "full":
        if config is not None and config.get("fine_tune_type") != "full":
            raise ValueError(
                "This is an adapter checkpoint; use train_type='lora' or 'dora'."
            )
        _dequantize_model(model)
        if weights is not None:
            model.load_weights(str(weights), strict=True)
        model.unfreeze()
        _record_adapter_config(model, {"fine_tune_type": "full"})
    else:
        if config is not None and config.get("fine_tune_type") == "full":
            raise ValueError(
                "This is a full-training checkpoint; use train_type='full'."
            )
        from mlx_vlm.trainer.peft import get_peft_model, load_adapters
        from mlx_vlm.trainer.peft.utils import _resolve_lora_targets

        if weights is None:
            if type(lora_rank) is not int or lora_rank < 1 or lora_alpha <= 0:
                raise ValueError(
                    "LoRA rank must be a positive integer and alpha must be positive."
                )
            if not 0 <= lora_dropout < 1:
                raise ValueError("LoRA dropout must be in [0, 1).")
            _resolve_lora_targets(model, target_modules)
        saved_quantization = (config or {}).get("base_quantization", {})
        if quantization_bits is not None:
            if saved_quantization and any(
                item
                != {
                    "bits": quantization_bits,
                    "group_size": quantization_group_size,
                    "mode": "affine",
                }
                for item in saved_quantization.values()
            ):
                raise ValueError(
                    "Quantization differs from the saved checkpoint; leave quantization_bits=None."
                )
            if not saved_quantization:
                _quantize_model(model, quantization_bits, quantization_group_size)

        quantization = saved_quantization or _quantization_config(model)

        if weights is not None:
            load_adapters(model, weights)
        else:
            get_peft_model(
                model,
                target_modules,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_dora=train_type == "dora",
                verbose=False,
            )
        if quantization:
            model._vlm_adapter_config["base_quantization"] = quantization

    model.train()
    if verbose:
        from mlx_vlm.trainer.common.utils import print_trainable_parameters

        print_trainable_parameters(model)
    return model
