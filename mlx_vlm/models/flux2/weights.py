from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.flux2.config import Flux2Variant
from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.qwen.text_encoder import Qwen3TextEncoder
from mlx_vlm.models.flux2.transformer import Flux2Transformer
from mlx_vlm.models.flux2.vae import Flux2VAE

FULL_DECODER_CHANNELS = (128, 256, 512, 512)


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    bits: int
    group_size: int
    mode: str = "affine"

    def as_dict(self) -> dict[str, int | str]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


def load_text_encoder(
    model_path: str | Path, variant: Flux2Variant
) -> Qwen3TextEncoder:
    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "text_encoder")
    weights = {}
    for key, value in raw.items():
        mapped = key.removeprefix("model.")
        if not mapped.startswith(("embed_tokens.", "layers.", "norm.")):
            continue
        weights[mapped] = _cast_float(value)
    text_encoder = Qwen3TextEncoder(**variant.text_encoder_overrides)
    return _apply_weights(text_encoder, weights, metadata)


def load_transformer(model_path: str | Path, variant: Flux2Variant) -> Flux2Transformer:
    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "transformer")
    weights = {}
    for key, value in raw.items():
        mapped = key
        mapped = mapped.replace(
            "time_guidance_embed.timestep_embedder.", "time_guidance_embed."
        )
        mapped = mapped.replace(".to_out.0.", ".to_out.")
        weights[mapped] = _cast_float(value)
    transformer = Flux2Transformer(**variant.transformer_overrides)
    return _apply_weights(transformer, weights, metadata)


def load_vae(model_path: str | Path, *, include_encoder: bool = False) -> Flux2VAE:
    raw, metadata = _load_safetensors(Path(model_path).expanduser() / "vae")
    vae = Flux2VAE(
        decoder_block_out_channels=FULL_DECODER_CHANNELS,
        include_encoder=include_encoder,
        encoder_block_out_channels=FULL_DECODER_CHANNELS,
    )
    target_shapes = {
        key: tuple(value.shape) for key, value in tree_flatten(vae.parameters())
    }
    prefixes = ("decoder.", "post_quant_conv.", "bn.")
    if include_encoder:
        prefixes += ("encoder.", "quant_conv.")
    weights = {}
    for key, value in raw.items():
        if key.endswith(".num_batches_tracked"):
            continue
        if not key.startswith(prefixes):
            continue
        mapped = key.replace(".to_out.0.", ".to_out.")
        tensor = _cast_float(value)
        if tensor.ndim == 4:
            tensor = _match_conv_layout(
                tensor,
                target_shape=target_shapes.get(mapped),
                key=mapped,
            )
        weights[mapped] = tensor
    return _apply_weights(vae, weights, metadata)


def _cast_float(value: mx.array) -> mx.array:
    if mx.issubdtype(value.dtype, mx.floating):
        return value.astype(ModelConfig.precision)
    return value


def _match_conv_layout(
    value: mx.array,
    *,
    target_shape: tuple[int, ...] | None,
    key: str,
) -> mx.array:
    if target_shape is None or tuple(value.shape) == target_shape:
        return value
    transposed = value.transpose(0, 2, 3, 1)
    if tuple(transposed.shape) == target_shape:
        return transposed
    raise ValueError(
        f"Unsupported convolution weight shape for {key}: "
        f"checkpoint={tuple(value.shape)}, expected={target_shape}"
    )


def _apply_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
):
    quantization = _quantization_config(model, weights, metadata)
    if quantization is not None:
        quantized_paths = {
            key[: -len(".scales")] for key in weights if key.endswith(".scales")
        }
        nn.quantize(
            model,
            group_size=quantization.group_size,
            bits=quantization.bits,
            mode=quantization.mode,
            class_predicate=lambda path, module: path in quantized_paths,
        )
        model.quantization_config = quantization.as_dict()
    else:
        model.quantization_config = None
    model.update(tree_unflatten(list(weights.items())), strict=True)
    return model


def _quantization_config(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> QuantizationConfig | None:
    quantized_paths = sorted(
        key[: -len(".scales")] for key in weights if key.endswith(".scales")
    )
    if not quantized_paths:
        return None

    modules = dict(model.named_modules())
    inferred_bits = set()
    inferred_group_sizes = set()
    for path in quantized_paths:
        module = modules.get(path)
        packed = weights.get(f"{path}.weight")
        scales = weights.get(f"{path}.scales")
        if module is None or not hasattr(module, "to_quantized"):
            raise ValueError(
                f"Quantized checkpoint path has no matching module: {path}"
            )
        if packed is None or scales is None:
            raise ValueError(f"Incomplete quantized weights for module: {path}")
        if packed.dtype != mx.uint32:
            raise ValueError(
                f"Expected packed uint32 weights for {path}, got {packed.dtype}"
            )
        if not hasattr(module, "weight") or module.weight.ndim != 2:
            raise ValueError(f"Unsupported quantized module shape for: {path}")

        output_dims, input_dims = module.weight.shape
        if packed.shape[0] != output_dims or scales.shape[0] != output_dims:
            raise ValueError(
                f"Quantized output shape mismatch for {path}: "
                f"weight={tuple(packed.shape)}, scales={tuple(scales.shape)}, "
                f"expected output={output_dims}"
            )
        packed_bits = packed.shape[-1] * 32
        if packed_bits % input_dims:
            raise ValueError(f"Could not infer quantization bits for {path}")
        if input_dims % scales.shape[-1]:
            raise ValueError(f"Could not infer quantization group size for {path}")
        inferred_bits.add(packed_bits // input_dims)
        inferred_group_sizes.add(input_dims // scales.shape[-1])

        if f"{path}.biases" not in weights:
            raise ValueError(f"Quantized module {path} is missing affine bias tensors")

    if len(inferred_bits) != 1 or len(inferred_group_sizes) != 1:
        raise ValueError(
            "FLUX.2 checkpoints with mixed quantization parameters are not supported"
        )
    inferred_bits_value = inferred_bits.pop()
    inferred_group_size = inferred_group_sizes.pop()
    configured_bits = _metadata_int(
        metadata,
        "bits",
        "num_bits",
        "quantization_level",
    )
    configured_group_size = _metadata_int(
        metadata,
        "group_size",
        "quantization_group_size",
    )
    if configured_bits is not None and configured_bits != inferred_bits_value:
        raise ValueError(
            "Quantization bits disagree with checkpoint tensor shapes: "
            f"configured={configured_bits}, inferred={inferred_bits_value}"
        )
    if (
        configured_group_size is not None
        and configured_group_size != inferred_group_size
    ):
        raise ValueError(
            "Quantization group size disagrees with checkpoint tensor shapes: "
            f"configured={configured_group_size}, inferred={inferred_group_size}"
        )
    mode = str(_metadata_value(metadata, "mode") or "affine")
    if mode != "affine":
        raise ValueError(f"Unsupported FLUX.2 quantization mode: {mode}")
    return QuantizationConfig(
        bits=configured_bits or inferred_bits_value,
        group_size=configured_group_size or inferred_group_size,
        mode=mode,
    )


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int | None:
    value = _metadata_value(metadata, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid quantization metadata value: {value!r}") from exc


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    candidates = [metadata]
    for container_key in ("quantization", "quantization_config"):
        value = metadata.get(container_key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        for key in keys:
            if key in candidate:
                return candidate[key]
    return None


def _load_safetensors(
    directory: Path,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing weight directory: {directory}")
    weights: dict[str, mx.array] = {}
    metadata: dict[str, Any] = {}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            index_metadata = index.get("metadata", {})
            if isinstance(index_metadata, dict):
                metadata.update(index_metadata)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensor index: {index_path}") from exc
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    for file in files:
        shard, shard_metadata = mx.load(str(file), return_metadata=True)
        weights.update(shard)
        if isinstance(shard_metadata, dict):
            metadata.update(shard_metadata)
    return weights, metadata
