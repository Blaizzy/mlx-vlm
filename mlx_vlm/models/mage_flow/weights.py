from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from mlx_vlm.models.qwen3_vl.config import ModelConfig as Qwen3VLConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.quant_utils import QUANTIZATION_MODE_DEFAULTS

from .text_encoder import MageFlowTextEncoder
from .transformer import MageFlowTransformer
from .vae import MageVAE


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


def _load_safetensors(
    directory: Path,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    weights: dict[str, mx.array] = {}
    metadata: dict[str, Any] = {}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_metadata = json.loads(index_path.read_text()).get("metadata", {})
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensor index: {index_path}") from exc
        if isinstance(index_metadata, dict):
            metadata.update(index_metadata)
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


def _apply_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> nn.Module:
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

    mode_value = _metadata_value(metadata, "mode", "quantization_mode", "q_mode")
    if mode_value is None:
        raise ValueError("Quantized checkpoint is missing its quantization mode")
    mode = str(mode_value).lower()
    bits = _metadata_int(
        metadata,
        "bits",
        "num_bits",
        "quantization_level",
        "q_bits",
    )
    group_size = _metadata_int(
        metadata, "group_size", "quantization_group_size", "q_group_size"
    )
    if bits is None or group_size is None:
        raise ValueError("Quantized checkpoint is missing its bits or group size")
    if mode not in QUANTIZATION_MODE_DEFAULTS:
        raise ValueError(f"Unsupported Mage-Flow quantization mode: {mode}")
    config = QuantizationConfig(
        bits=bits,
        group_size=group_size,
        mode=mode,
    )
    default_group_size, default_bits = QUANTIZATION_MODE_DEFAULTS[mode]
    if mode != "affine" and (config.group_size, config.bits) != (
        default_group_size,
        default_bits,
    ):
        raise ValueError(
            f"{mode} requires group_size={default_group_size}, bits={default_bits}"
        )
    _validate_quantized_paths(model, weights, quantized_paths, config)
    return config


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
        if not hasattr(module, "weight") or module.weight.ndim != 2:
            raise ValueError(f"Unsupported quantized module shape for: {path}")
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


def sanitize_transformer_weights(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    sanitized = {}
    for key, value in weights.items():
        key = key.replace(".img_mod.1.", ".img_mod.linear.")
        key = key.replace(".txt_mod.1.", ".txt_mod.linear.")
        key = key.replace(".img_mlp.net.0.proj.", ".img_mlp.linear_in.")
        key = key.replace(".img_mlp.net.2.", ".img_mlp.linear_out.")
        key = key.replace(".txt_mlp.net.0.proj.", ".txt_mlp.linear_in.")
        key = key.replace(".txt_mlp.net.2.", ".txt_mlp.linear_out.")
        key = key.replace(".attn.to_out.0.", ".attn.to_out.")
        sanitized[key] = value
    return sanitized


def load_transformer(model_path: str | Path) -> MageFlowTransformer:
    root = Path(model_path).expanduser()
    config = json.loads((root / "transformer" / "config.json").read_text())
    transformer = MageFlowTransformer(
        in_channels=int(config.get("in_channels", 128)),
        out_channels=int(config.get("out_channels", 128)),
        context_in_dim=int(config.get("context_in_dim", 2560)),
        hidden_size=int(config.get("hidden_size", 3072)),
        num_heads=int(config.get("num_heads", 24)),
        depth=int(config.get("depth", 12)),
        axes_dim=tuple(config.get("axes_dim", (16, 56, 56))),
        theta=float(config.get("theta", 10000)),
    )
    weights, metadata = _load_safetensors(root / "transformer")
    weights = sanitize_transformer_weights(weights)
    return _apply_weights(transformer, weights, {**config, **metadata})


def _map_vae_key(key: str) -> str | None:
    if key.startswith("student.dconv_encoder."):
        key = "dconv_encoder." + key.removeprefix("student.dconv_encoder.")
    elif key.startswith(("dconv_encoder.", "decoder_model.")):
        pass
    elif key.startswith("pipeline.y_embedder.encoder."):
        return None
    elif key.startswith("pipeline."):
        key = "decoder_model." + key.removeprefix("pipeline.")
    else:
        return None

    key = key.replace(".adaLN_modulation.1.", ".adaLN_modulation.linear.")
    key = key.replace(".ca.1.", ".ca_conv.")
    key = key.replace(".t_embedder.mlp.0.", ".t_embedder.linear_1.")
    key = key.replace(".t_embedder.mlp.2.", ".t_embedder.linear_2.")
    key = key.replace(".x_embedder.embedder.0.", ".x_embedder.linear.")
    key = re.sub(
        r"(\.dec_net\.res_blocks\.\d+)\.mlp\.0\.",
        r"\1.linear_1.",
        key,
    )
    key = re.sub(
        r"(\.dec_net\.res_blocks\.\d+)\.mlp\.2\.",
        r"\1.linear_2.",
        key,
    )
    return key


def sanitize_vae_weights(
    weights: dict[str, mx.array],
    *,
    source_layout: str = "pytorch_nchw",
) -> dict[str, mx.array]:
    if source_layout not in {"mlx_nhwc", "pytorch_nchw"}:
        raise ValueError(f"Unsupported Mage-Flow VAE tensor layout: {source_layout}")
    sanitized = {}
    for raw_key, value in weights.items():
        key = _map_vae_key(raw_key)
        if key is None or raw_key.endswith(".num_batches_tracked"):
            continue
        if value.ndim == 4 and source_layout == "pytorch_nchw":
            value = value.transpose(0, 2, 3, 1)
        sanitized[key] = value
    return sanitized


def load_vae(model_path: str | Path, *, include_encoder: bool = True) -> MageVAE:
    root = Path(model_path).expanduser()
    vae = MageVAE(include_encoder=include_encoder)
    weights, metadata = _load_safetensors(root / "vae")
    source_layout = (
        "mlx_nhwc"
        if metadata.get("tensor_layout") == "mlx_nhwc"
        or metadata.get("mlx_vlm_format") == "mage_flow"
        else "pytorch_nchw"
    )
    weights = sanitize_vae_weights(weights, source_layout=source_layout)
    if not include_encoder:
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("dconv_encoder.")
        }
    return _apply_weights(vae, weights, metadata)


def load_text_encoder(
    model_path: str | Path, *, max_length: int = 2048
) -> MageFlowTextEncoder:
    root = Path(model_path).expanduser()
    text_root = root / "text_encoder"
    config = json.loads((text_root / "config.json").read_text())
    model_config = Qwen3VLConfig.from_dict(config)
    model = Qwen3VLModel(model_config)
    weights, metadata = _load_safetensors(text_root)
    weights = model.sanitize(weights)
    weights = model.vision_tower.sanitize(weights)
    model = _apply_weights(model, weights, {**config, **metadata})
    return MageFlowTextEncoder(
        model=model,
        model_path=root,
        max_length=max_length,
    )


__all__ = [
    "QuantizationConfig",
    "infer_quantization_config",
    "load_text_encoder",
    "load_transformer",
    "load_vae",
    "sanitize_transformer_weights",
    "sanitize_vae_weights",
]
