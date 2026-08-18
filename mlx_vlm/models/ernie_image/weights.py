from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.flux2.vae import Flux2VAE

from .config import ErnieImageTransformerConfig
from .prompting import ErnieImagePromptEnhancer
from .text_encoder import ErnieImageTextConfig, ErnieImageTextEncoder
from .transformer import ErnieImageTransformer

_QUANTIZATION_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    bits: int
    group_size: int
    mode: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


def load_transformer(model_path: str | Path) -> ErnieImageTransformer:
    root = Path(model_path).expanduser()
    config_path = root / "transformer" / "config.json"
    config = (
        ErnieImageTransformerConfig.from_dict(_load_json(config_path))
        if config_path.exists()
        else ErnieImageTransformerConfig()
    )
    transformer = ErnieImageTransformer(config)
    target_shapes = _parameter_shapes(transformer)
    raw, metadata = load_safetensors(root / "transformer")
    weights = sanitize_transformer_weights(raw, target_shapes=target_shapes)
    return apply_weights(transformer, weights, metadata)


def sanitize_transformer_weights(
    raw: dict[str, mx.array],
    *,
    target_shapes: dict[str, tuple[int, ...]],
) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for raw_key, value in raw.items():
        key = raw_key.replace("adaLN_modulation.1.", "adaln_modulation.")
        key = key.replace(".to_out.0.", ".to_out.")
        tensor = _cast_float(value)
        if key == "x_embedder.proj.weight":
            tensor = match_conv_layout(
                tensor, target_shape=target_shapes.get(key), key=key
            )
        weights[key] = tensor
    return weights


def load_text_encoder(model_path: str | Path) -> ErnieImageTextEncoder:
    root = Path(model_path).expanduser()
    config_path = root / "text_encoder" / "config.json"
    config = (
        ErnieImageTextConfig.from_dict(_load_json(config_path))
        if config_path.exists()
        else ErnieImageTextConfig()
    )
    encoder = ErnieImageTextEncoder(config)
    raw, metadata = load_safetensors(root / "text_encoder")
    weights = sanitize_text_encoder_weights(raw)
    return apply_weights(encoder, weights, metadata)


def sanitize_text_encoder_weights(
    raw: dict[str, mx.array],
) -> dict[str, mx.array]:
    weights = {}
    for raw_key, value in raw.items():
        key = raw_key
        for prefix in ("language_model.model.", "model."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        if "rotary_emb.inv_freq" in key:
            continue
        if key.startswith(("embed_tokens.", "layers.", "norm.")):
            weights[key] = _cast_float(value)
    return weights


def load_prompt_enhancer(
    model_path: str | Path, *, max_new_tokens: int | None = None
) -> ErnieImagePromptEnhancer:
    root = Path(model_path).expanduser()
    config_path = root / "pe" / "config.json"
    config = (
        ErnieImageTextConfig.from_dict(_load_json(config_path))
        if config_path.exists()
        else ErnieImageTextConfig()
    )
    encoder = ErnieImageTextEncoder(config)
    raw, metadata = load_safetensors(root / "pe")
    weights = sanitize_text_encoder_weights(raw)
    apply_weights(encoder, weights, metadata)
    return ErnieImagePromptEnhancer(
        model=encoder,
        model_path=root,
        max_new_tokens=max_new_tokens,
    )


def load_vae(model_path: str | Path) -> Flux2VAE:
    root = Path(model_path).expanduser()
    vae = Flux2VAE(decoder_block_out_channels=(128, 256, 512, 512))
    vae.bn.eps = 1e-5
    target_shapes = _parameter_shapes(vae)
    raw, metadata = load_safetensors(root / "vae")
    weights = {}
    for raw_key, value in raw.items():
        if raw_key.endswith(".num_batches_tracked"):
            continue
        if not raw_key.startswith(("decoder.", "post_quant_conv.", "bn.")):
            continue
        key = raw_key.replace(".to_out.0.", ".to_out.")
        tensor = _cast_float(value)
        if tensor.ndim == 4:
            tensor = match_conv_layout(
                tensor, target_shape=target_shapes.get(key), key=key
            )
        weights[key] = tensor
    return apply_weights(vae, weights, metadata)


def _parameter_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(value.shape) for key, value in tree_flatten(model.parameters())
    }


def _cast_float(value: mx.array) -> mx.array:
    if mx.issubdtype(value.dtype, mx.floating):
        return value.astype(mx.bfloat16)
    return value


def match_conv_layout(
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


def apply_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
):
    quantization = infer_quantization_config(model, weights, metadata)
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


def infer_quantization_config(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> QuantizationConfig | None:
    quantized_paths = sorted(
        key[: -len(".scales")] for key in weights if key.endswith(".scales")
    )
    if not quantized_paths:
        return None

    mode_value = _metadata_value(
        metadata, "mode", "quantization_mode", "q_mode"
    )
    mode = str(mode_value).lower() if mode_value is not None else None
    bits = _metadata_int(
        metadata, "bits", "num_bits", "quantization_level", "q_bits"
    )
    group_size = _metadata_int(
        metadata, "group_size", "quantization_group_size", "q_group_size"
    )
    if mode is None:
        mode = _infer_quantization_mode(model, weights, quantized_paths)
    if mode not in _QUANTIZATION_DEFAULTS:
        raise ValueError(f"Unsupported ERNIE-Image quantization mode: {mode}")
    default_group_size, default_bits = _QUANTIZATION_DEFAULTS[mode]
    config = QuantizationConfig(
        bits=bits or default_bits,
        group_size=group_size or default_group_size,
        mode=mode,
    )
    _validate_quantized_paths(model, weights, quantized_paths, config)
    return config


def _infer_quantization_mode(
    model: nn.Module,
    weights: dict[str, mx.array],
    paths: list[str],
) -> str:
    if any(f"{path}.biases" in weights for path in paths):
        return "affine"
    modules = dict(model.named_modules())
    candidates = set(_QUANTIZATION_DEFAULTS) - {"affine"}
    for path in paths:
        module = modules.get(path)
        packed = weights[f"{path}.weight"]
        scales = weights[f"{path}.scales"]
        if module is None or not hasattr(module, "weight"):
            raise ValueError(
                f"Quantized checkpoint path has no matching module: {path}"
            )
        input_dims = module.weight.shape[-1]
        output_dims = module.weight.shape[0]
        candidates &= {
            mode
            for mode, (group_size, bits) in _QUANTIZATION_DEFAULTS.items()
            if mode != "affine"
            and tuple(packed.shape)
            == (output_dims, (input_dims * bits + 31) // 32)
            and tuple(scales.shape) == (output_dims, input_dims // group_size)
        }
    if len(candidates) != 1:
        raise ValueError(
            "Quantization metadata is required when tensor shapes do not identify "
            f"one mode, candidates={sorted(candidates)}"
        )
    return candidates.pop()


def _validate_quantized_paths(
    model: nn.Module,
    weights: dict[str, mx.array],
    paths: list[str],
    config: QuantizationConfig,
) -> None:
    modules = dict(model.named_modules())
    for path in paths:
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
        output_dims, input_dims = module.weight.shape
        expected_weight = (
            output_dims,
            (input_dims * config.bits + 31) // 32,
        )
        expected_scales = (
            output_dims,
            input_dims // config.group_size,
        )
        if tuple(packed.shape) != expected_weight:
            raise ValueError(
                f"Quantized weight shape mismatch for {path}: "
                f"{tuple(packed.shape)} vs {expected_weight}"
            )
        if tuple(scales.shape) != expected_scales:
            raise ValueError(
                f"Quantized scale shape mismatch for {path}: "
                f"{tuple(scales.shape)} vs {expected_scales}"
            )
        has_biases = f"{path}.biases" in weights
        if config.mode == "affine" and not has_biases:
            raise ValueError(f"Affine quantized module {path} is missing biases")
        if config.mode != "affine" and has_biases:
            raise ValueError(
                f"{config.mode} quantized module {path} must not contain biases"
            )


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


def _metadata_int(metadata: dict[str, Any], *keys: str) -> int | None:
    value = _metadata_value(metadata, *keys)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid quantization metadata value: {value!r}") from exc


def load_safetensors(
    directory: Path,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing weight directory: {directory}")
    weights: dict[str, mx.array] = {}
    metadata: dict[str, Any] = {}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json(index_path)
        index_metadata = index.get("metadata")
        if isinstance(index_metadata, dict):
            metadata.update(index_metadata)
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    for path in files:
        shard, shard_metadata = mx.load(str(path), return_metadata=True)
        duplicate = set(weights).intersection(shard)
        if duplicate:
            raise ValueError(
                f"Duplicate tensors across {directory} shards: {sorted(duplicate)[:3]}"
            )
        weights.update(shard)
        if isinstance(shard_metadata, dict):
            metadata.update(shard_metadata)
    return weights, metadata


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


__all__ = [
    "QuantizationConfig",
    "apply_weights",
    "infer_quantization_config",
    "load_safetensors",
    "load_prompt_enhancer",
    "load_text_encoder",
    "load_transformer",
    "load_vae",
    "match_conv_layout",
    "sanitize_text_encoder_weights",
    "sanitize_transformer_weights",
]
