"""Weight quantization helpers vendored from mlx-lm (mlx-lm 0.31.3) so that
mlx_vlm.convert no longer imports mlx_lm. Pure mlx.core/mlx.nn; behaviour is
identical. Part of the mlx-lm removal series."""

import copy
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce, tree_unflatten

QUANTIZATION_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


def get_quantization_params(
    group_size: Optional[int],
    bits: Optional[int],
    mode: str,
) -> dict:
    if mode not in QUANTIZATION_MODE_DEFAULTS:
        raise ValueError(f"Unsupported quantization mode: {mode}")
    default_group_size, default_bits = QUANTIZATION_MODE_DEFAULTS[mode]
    resolved_group_size = group_size or default_group_size
    resolved_bits = bits or default_bits
    if mode != "affine" and (resolved_group_size, resolved_bits) != (
        default_group_size,
        default_bits,
    ):
        raise ValueError(
            f"{mode} requires group_size={default_group_size}, bits={default_bits}"
        )
    return {
        "group_size": resolved_group_size,
        "bits": resolved_bits,
        "mode": mode,
    }


def get_total_parameters(model):
    leaf_modules = tree_flatten(
        model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)
    )

    def nparams(m):
        if hasattr(m, "bits"):
            n = 0 if not hasattr(m, "bias") else m.bias.size
            return n + m.weight.size * 32 // m.bits
        return sum(v.size for _, v in tree_flatten(m.parameters()))

    return sum(nparams(m) for _, m in leaf_modules)


def compute_bits_per_weight(model):
    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    model_params = get_total_parameters(model)
    return model_bytes * 8 / model_params


def quantize_model(
    model: nn.Module,
    config: dict,
    group_size: Optional[int],
    bits: Optional[int],
    mode: str = "affine",
    quant_predicate: Optional[Callable[[str, nn.Module], Union[bool, dict]]] = None,
) -> Tuple[nn.Module, dict]:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        group_size (Optional[int]): Group size for quantization.
        bits (Optional[int]): Bits per weight for quantization.
        mode (str): The quantization mode.
        quant_predicate (Callable): A callable that decides how to quantize
          each layer based on the path. Accepts the layer `path` and the
          `module`. Returns either a bool to signify quantize/no quantize or
          a dict of quantization parameters to pass to `to_quantized`.

    Returns:
        Tuple: Tuple containing quantized model and config.
    """

    quantized_config = copy.deepcopy(config)

    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    quant_params = get_quantization_params(group_size, bits, mode)
    group_size = quant_params["group_size"]
    bits = quant_params["bits"]
    if "quantization" in quantized_config:
        # If the model is already partially quantized, return params so that
        # the config is set on a per-layer basis
        fine_grained_config = True
    else:
        fine_grained_config = False
        quantized_config["quantization"] = dict(quant_params)

    def wrapped_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False

        input_dims = module.weight.shape[-1]
        bool_or_params = True
        default_group_is_compatible = input_dims % group_size == 0
        if not default_group_is_compatible:
            if quant_predicate is None:
                return False
            bool_or_params = quant_predicate(path, module)
            if not (
                isinstance(bool_or_params, dict)
                and "fallback_group_size" in bool_or_params
            ):
                return False
        elif quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)

        if isinstance(bool_or_params, dict) and "fallback_group_size" in bool_or_params:
            overrides = dict(bool_or_params)
            fallback_group_size = overrides.pop("fallback_group_size")
            bool_or_params = {**quant_params, **overrides}
            if (
                input_dims % bool_or_params["group_size"]
                and fallback_group_size is not None
                and input_dims % fallback_group_size == 0
            ):
                bool_or_params["group_size"] = fallback_group_size
        module_group_size = (
            bool_or_params.get("group_size", group_size)
            if isinstance(bool_or_params, dict)
            else group_size
        )
        if input_dims % module_group_size != 0:
            return False
        if isinstance(bool_or_params, dict):
            quantized_config["quantization"][path] = bool_or_params
        elif fine_grained_config and bool_or_params:
            quantized_config["quantization"][path] = quant_params
        return bool_or_params

    nn.quantize(
        model,
        group_size,
        bits,
        mode=mode,
        class_predicate=wrapped_predicate,
    )
    # support hf model tree #957
    quantized_config["quantization_config"] = quantized_config["quantization"]

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Quantized model with {bpw:.3f} bits per weight.")

    return model, quantized_config


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized layers in the model.

    Args:
        model (nn.Module): The model with quantized layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    dequantize_layers = []
    for name, module in model.named_modules():
        bias = "bias" in module
        if isinstance(module, nn.QuantizedLinear):
            cls = nn.Linear
            kwargs = {"bias": bias}
        elif isinstance(module, nn.QuantizedEmbedding):
            kwargs = {}
            cls = nn.Embedding
        elif isinstance(module, QuantizedSwitchLinear):
            kwargs = {"bias": bias}
            cls = SwitchLinear
        else:
            continue
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            module.group_size,
            module.bits,
            module.mode,
        )
        args = weight.shape[::-1]
        m = cls(*args, **kwargs)
        if bias:
            m.bias = module.bias
        m.weight = weight
        dequantize_layers.append((name, m))

    if len(dequantize_layers) > 0:
        model.update_modules(tree_unflatten(dequantize_layers))
    return model
