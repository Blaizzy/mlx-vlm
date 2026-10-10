"""Create, load, and save adapters for composite language/audio/vision models."""

from pathlib import Path
from typing import Union

import mlx.nn as nn

from mlx_vlm.trainer.common.model import (
    _dequantize_model,
    _read_checkpoint,
    _record_adapter_config,
    _restore_quantization,
)
from mlx_vlm.trainer.common.utils import (
    get_module_by_name,
    print_trainable_parameters,
    save_trainable_weights,
    set_module_by_name,
)

from .adapter_utils import linear_to_lora_layers
from .dora_layers import DoRALinear
from .lora import LoRaLayer
from .lora_layers import LoRALinear

DEFAULT_LORA_NUM_LAYERS = -1


def _lora_scale(alpha: float, rank: int) -> float:
    return alpha / rank


def _to_lora(layer, lora_parameters, use_dora=False):
    if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
        if use_dora:
            return DoRALinear.from_base(
                layer,
                r=lora_parameters["rank"],
                scale=lora_parameters["scale"],
                dropout=lora_parameters["dropout"],
            )

        return LoRALinear.from_base(
            layer,
            r=lora_parameters["rank"],
            scale=lora_parameters["scale"],
            dropout=lora_parameters["dropout"],
        )

    raise ValueError(f"Can't convert layer of type {type(layer).__name__} to LoRA")


def _adapter_target(model):
    """Return the transformer module that owns the target layers."""
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return language_model
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "layers"):
        return inner_model
    return model


def _resolve_lora_targets(model, linear_layers):
    target = _adapter_target(model)
    prefix = next(
        (name for name, module in model.named_modules() if module is target), ""
    )
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.QuantizedLinear))
    }
    candidates = [
        name for name in modules if not prefix or name.startswith(prefix + ".")
    ]
    if linear_layers is None:
        suffixes = set(find_all_linear_names(target))
        keys = [name for name in candidates if name.split(".")[-1] in suffixes]
    else:
        if isinstance(linear_layers, str):
            raise ValueError(
                "Pass layer names as a list, such as ['q_proj', 'v_proj']."
            )
        keys = []
        for name in linear_layers:
            if name in modules:
                matches = [name]
            elif "." in name:
                matches = [key for key in (name, f"{prefix}.{name}") if key in modules]
            else:
                matches = [key for key in candidates if key.split(".")[-1] == name]
            if not matches:
                raise ValueError(
                    f"No matching Linear targets for {name}. "
                    f"Available layer names: {sorted(modules)}."
                )
            keys.extend(matches)
    if not keys:
        raise ValueError(
            f"No matching Linear targets. Available layer names: {sorted(modules)}. "
            "Load a fresh base model and choose names from this list."
        )
    return list(dict.fromkeys(keys))


def _apply_lora_layers(model, config):
    lora_parameters = dict(config["lora_parameters"])
    fine_tune_type = config.get("fine_tune_type", "lora")
    use_dora = fine_tune_type == "dora"

    if fine_tune_type == "full":
        return model

    if "keys" not in lora_parameters:
        linear_to_lora_layers(
            _adapter_target(model),
            config.get("num_layers", DEFAULT_LORA_NUM_LAYERS),
            lora_parameters,
            use_dora=use_dora,
        )
        return model

    for name in lora_parameters["keys"]:
        target = model
        try:
            module = get_module_by_name(target, name)
        except (AttributeError, IndexError, KeyError):
            target = _adapter_target(model)
            module = get_module_by_name(target, name)
        set_module_by_name(
            target, name, _to_lora(module, lora_parameters, use_dora=use_dora)
        )
    return model


def _lora_config(rank, alpha, dropout, *, use_dora=False):
    return {
        "fine_tune_type": "dora" if use_dora else "lora",
        "num_layers": DEFAULT_LORA_NUM_LAYERS,
        "lora_parameters": {
            "rank": rank,
            "dropout": dropout,
            "scale": _lora_scale(alpha, rank),
        },
    }


def _apply_legacy_lora_layers(model, config):
    list_of_modules = find_all_linear_names(model.language_model)
    return get_peft_model(
        model,
        list_of_modules,
        rank=config["rank"],
        alpha=config.get("alpha", 0.1),
        dropout=config.get("dropout", 0.1),
        legacy=True,
    )


def get_peft_model(
    model,
    linear_layers=None,
    rank=10,
    alpha=0.1,
    dropout=0.1,
    freeze=True,
    verbose=True,
    legacy=False,
    use_dora=False,
):
    """Attach LoRA/DoRA adapters, discovering transformer targets by default.

    Args:
        model: A loaded MLX model.
        linear_layers: Linear suffixes or qualified names, or None to discover
            transformer targets automatically (excluding the language head).
        rank: Adapter rank.
        alpha: Adapter scaling numerator; the update scale is alpha / rank.
        dropout: Dropout probability for adapter inputs.
        freeze: Freeze the entire base model before attaching trainable adapters.
        verbose: Print the trainable parameter count.
        legacy: Use the legacy adapter format.
        use_dora: Use DoRA instead of LoRA.

    Returns:
        The same model with adapters attached and checkpoint metadata recorded.
    """
    keys = _resolve_lora_targets(model, linear_layers)
    if rank <= 0:
        raise ValueError("LoRA rank must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError(
            "LoRA dropout must be between 0 (inclusive) and 1 (exclusive)."
        )
    if freeze:
        freeze_model(model)
        model.freeze()
    if legacy:
        for name in keys:
            module = get_module_by_name(model, name)
            set_module_by_name(model, name, LoRaLayer(module, rank, alpha, dropout))
        config = {"rank": rank, "alpha": alpha, "dropout": dropout}
    else:
        config = _lora_config(rank, alpha, dropout, use_dora=use_dora)
        config["lora_parameters"]["keys"] = keys
        _apply_lora_layers(model, config)
    _record_adapter_config(model, config)
    if verbose:
        print_trainable_parameters(model)
    return model


def apply_lora_layers(model: nn.Module, adapter_path: Union[str, Path]) -> nn.Module:
    """Restore adapter settings, base quantization, and weights onto a fresh base.

    Pass the saved weights file with adapter_config.json beside it, or a folder
    containing adapters.safetensors or weights.safetensors. The base stays frozen
    and the restored adapters remain trainable.
    """
    weights_path, config = _read_checkpoint(adapter_path)
    if config.get("fine_tune_type") == "full":
        _dequantize_model(model)
        model.load_weights(str(weights_path), strict=True)
        _record_adapter_config(model, config)
        return model
    if "rank" not in config and "lora_parameters" not in config:
        raise ValueError("The adapter does not have lora params in the config")
    _restore_quantization(model, config.get("base_quantization", {}))
    model.freeze()
    if "lora_parameters" in config:
        _apply_lora_layers(model, config)
    else:
        _apply_legacy_lora_layers(model, config)
    model.load_weights(str(weights_path), strict=False)
    _record_adapter_config(model, config)
    return model


def load_adapters(model: nn.Module, adapter_path: Union[str, Path]) -> nn.Module:
    """Restore adapters from a saved weights file or adapter directory.

    Load the original base model first. The saved metadata restores the target
    layers, rank, scale, and dropout, so no manual LoRA setup is needed.
    """
    return apply_lora_layers(model, adapter_path)


def dequantize(model: nn.Module) -> nn.Module:
    """Replace quantized linear and embedding modules with float modules."""
    return _dequantize_model(model)


def save_adapter(model: nn.Module, adapter_file: Union[str, Path]):
    """Save adapter weights and config."""
    save_trainable_weights(model, adapter_file)


def freeze_model(model):
    top_level_to_freeze = {
        "language_model",
        "vision_model",
        "vision_tower",
        "aligner",
        "connector",
        "multi_modal_projector",
        "mm_projector",
        "audio_tower",
        "embed_audio",
        "embed_vision",
    }
    for name, module in model.named_modules():
        name = name.split(".")[0]
        if name in top_level_to_freeze and hasattr(model, name):
            try:
                model[f"{name}"].freeze()
            except Exception:
                # Fallback for towers whose .freeze() errors on non-Module
                # sub-objects (e.g. Gemma 4 audio_tower).
                try:
                    from mlx.utils import tree_flatten

                    top = model[f"{name}"]
                    leaves = tree_flatten(
                        top.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
                    )
                    for _, m in leaves:
                        m.freeze(recurse=False)
                except Exception:
                    pass


def find_all_linear_names(model):
    cls = nn.Linear
    quantized_cls = nn.QuantizedLinear
    lora_module_names = set()
    multimodal_keywords = [
        "mm_projector",
        "vision_tower",
        "vision_resampler",
        "aligner",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) or isinstance(module, quantized_cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def unfreeze_modules(model: nn.Module, module_names):
    """Unfreeze modules whose qualified names match any of the given patterns.

    This scans model.named_modules() so nested components like
    "vision_tower.layers.0" are handled as well.
    """
    targets = set(module_names)
    found = set()
    for full_name, sub in model.named_modules():
        top = full_name.split(".")[0] if full_name else ""
        if any((name == top) or (name in full_name) for name in targets):
            if hasattr(sub, "unfreeze"):
                sub.unfreeze()
                found.add(top or full_name)
    if not found:
        print(
            "[warn] unfreeze_modules: no matching modules found for patterns:",
            ", ".join(module_names),
        )
