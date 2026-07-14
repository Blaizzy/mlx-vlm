"""Weight quantization helpers vendored from mlx-lm (mlx-lm 0.31.3) so that
mlx_vlm.convert no longer imports mlx_lm. Pure mlx.core/mlx.nn; behaviour is
identical. Part of the mlx-lm removal series."""

import copy
from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_reduce, tree_unflatten


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

    def defaults_for_mode(mode, group_size, bits):
        mode_defaults = {
            "affine": (64, 4),
            "mxfp4": (32, 4),
            "nvfp4": (16, 4),
            "mxfp8": (32, 8),
        }
        default_group_size, default_bits = mode_defaults[mode]
        return group_size or default_group_size, bits or default_bits

    quantized_config = copy.deepcopy(config)

    quant_predicate = quant_predicate or getattr(model, "quant_predicate", None)
    group_size, bits = defaults_for_mode(mode, group_size, bits)
    quant_params = {"group_size": group_size, "bits": bits, "mode": mode}
    if "quantization" in quantized_config:
        # If the model is already partially quantized, return params so that
        # the config is set on a per-layer basis
        fine_grained_config = True
    else:
        fine_grained_config = False
        quantized_config["quantization"] = quant_params

    def wrapped_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        bool_or_params = True
        if quant_predicate is not None:
            bool_or_params = quant_predicate(path, module)
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
