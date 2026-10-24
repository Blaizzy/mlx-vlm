"""Export trained models using the standard MLX-VLM inference layout."""

import json
import shutil
from pathlib import Path


def export_model(
    model,
    processor,
    output_path,
    *,
    model_path,
    quantize: bool | None = None,
    bits: int | None = None,
    group_size: int | None = None,
):
    """Save a standalone inference model without changing the live training model.

    Args:
        model: Loaded or trained MLX model. LoRA/DoRA adapters are fused on export.
        processor: Processor returned by load.
        output_path: New directory for weights, config, and processor assets.
        model_path: Original model directory or Hub identifier for config/assets.
        quantize: None automatically packs QAT layers at their trained settings
            and otherwise preserves existing packing. True also quantizes other
            eligible layers. False saves QAT master weights without new packing;
            already packed layers remain packed.
        bits: Bit width for forced quantization (default 4). QAT layers use their
            trained setting unless explicitly overridden with quantize=True.
        group_size: Forced quantization group size (default 64), with the same
            QAT override behavior as bits.

    Returns:
        Path loadable directly with mlx_vlm.load.
    """
    from mlx import nn
    from mlx.utils import tree_unflatten

    from .trainer.common.qat import QATLinear
    from .trainer.peft import (
        DoRAEmbedding,
        DoRALinear,
        LoRAEmbedding,
        LoRALinear,
        LoRASwitchLinear,
    )
    from .utils import get_model_path, save_config, save_weights

    if quantize is not None and type(quantize) is not bool:
        raise TypeError("quantize must be True, False, or None (automatic).")
    if (bits is not None or group_size is not None) and quantize is not True:
        raise ValueError("Set quantize=True to override export bits/group_size.")
    if bits is not None and bits not in {4, 8}:
        raise ValueError("bits must be 4 or 8.")
    if group_size is not None and group_size not in {32, 64, 128}:
        raise ValueError("group_size must be 32, 64, or 128.")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Use a new directory for model export: {destination}")
    source = get_model_path(str(model_path))
    config = json.loads((source / "config.json").read_text())
    # Remember replaced modules by stage, restoring children before their parents.
    stages = []

    def replace(replacements):
        modules = dict(model.named_modules())
        stages.append([(name, modules[name]) for name, _ in replacements])
        if replacements:
            model.update_modules(tree_unflatten(replacements))

    try:
        adapters = (
            LoRALinear,
            DoRALinear,
            LoRAEmbedding,
            DoRAEmbedding,
            LoRASwitchLinear,
        )
        replace(
            [
                (name, layer.fuse(dequantize=quantize is True))
                for name, layer in model.named_modules()
                if isinstance(layer, adapters)
            ]
        )
        if any(
            type(layer).__name__ == "LoRaLayer" for _, layer in model.named_modules()
        ):
            raise ValueError("Legacy LoRaLayer export requires fusing adapters first.")
        saved_qat = getattr(model, "_vlm_adapter_config", {}).get("qat", {})
        replacements = []
        for name, layer in model.named_modules():
            if isinstance(layer, QATLinear) or (
                saved_qat
                and type(layer) is nn.Linear
                and layer.weight.shape[-1] % saved_qat["group_size"] == 0
            ):
                if quantize is False:
                    output_dims, input_dims = layer.weight.shape
                    packed = nn.Linear(input_dims, output_dims, bias="bias" in layer)
                    packed.weight = layer.weight
                    if "bias" in layer:
                        packed.bias = layer.bias
                else:
                    packed = nn.QuantizedLinear.from_linear(
                        layer,
                        bits=(
                            bits
                            if bits is not None
                            else getattr(layer, "bits", saved_qat.get("bits"))
                        ),
                        group_size=(
                            group_size
                            if group_size is not None
                            else getattr(
                                layer, "group_size", saved_qat.get("group_size")
                            )
                        ),
                        mode="affine",
                    )
                replacements.append((name, packed))
            elif (
                quantize is True
                and hasattr(layer, "to_quantized")
                and not hasattr(layer, "scales")
                and layer.weight.shape[-1] % (group_size or 64) == 0
            ):
                replacements.append(
                    (
                        name,
                        layer.to_quantized(
                            bits=bits or 4, group_size=group_size or 64, mode="affine"
                        ),
                    )
                )
        replace(replacements)
        # Derive metadata from the exported layout, never the original base config.
        parameters = {
            name: {
                "bits": layer.bits,
                "group_size": layer.group_size,
                "mode": getattr(layer, "mode", "affine"),
            }
            for name, layer in model.named_modules()
            if hasattr(layer, "scales")
            and hasattr(layer, "bits")
            and hasattr(layer, "group_size")
        }
        config.pop("quantization", None)
        config.pop("quantization_config", None)
        config.pop("lora", None)
        if parameters:
            config["quantization"] = {**next(iter(parameters.values())), **parameters}
            config["quantization_config"] = config["quantization"]
        save_weights(destination, model)
    finally:
        for originals in reversed(stages):
            if originals:
                model.update_modules(tree_unflatten(originals))

    for pattern in ("*.json", "*.py", "*.txt", "*.jinja", "*.model", "*.tiktoken"):
        for asset in source.glob(pattern):
            if asset.name not in {
                "config.json",
                "model.safetensors.index.json",
                "adapter_config.json",
            }:
                shutil.copyfile(asset, destination / asset.name)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(destination)
    save_config(config, destination / "config.json")
    return destination
