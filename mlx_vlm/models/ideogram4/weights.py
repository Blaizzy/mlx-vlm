from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_vlm.models.flux2.qwen.text_encoder import Qwen3TextEncoder
from mlx_vlm.models.flux2.vae import Flux2VAE
from mlx_vlm.models.flux2.weights import load_vae as load_flux2_vae

from .config import Ideogram4TransformerConfig
from .transformer import Ideogram4Transformer

PRECISION = mx.bfloat16
_FLOAT_DTYPES = {mx.float16, mx.float32, mx.bfloat16}


def dequantize_fp8_weight_only(
    weights: dict[str, mx.array],
    *,
    precision: mx.Dtype = PRECISION,
) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.endswith(".weight_scale"):
            continue
        scale_key = f"{key}_scale"
        if key.endswith(".weight") and scale_key in weights:
            scale = weights[scale_key].astype(precision)
            converted[key] = mx.from_fp8(
                value.astype(mx.uint8), precision
            ) * mx.expand_dims(scale, axis=-1)
        elif value.dtype in _FLOAT_DTYPES:
            converted[key] = value.astype(precision)
        else:
            converted[key] = value
    return converted


def load_text_encoder(model_path: str | Path) -> Qwen3TextEncoder:
    root = Path(model_path).expanduser()
    config = _load_json(root / "text_encoder" / "config.json")
    text_config = config["text_config"]
    raw = _load_safetensors(root / "text_encoder")
    raw = dequantize_fp8_weight_only(raw)

    weights = {}
    for key, value in raw.items():
        if key.startswith("language_model."):
            key = key[len("language_model.") :]
        if key.startswith(("embed_tokens.", "layers.", "norm.")):
            weights[key] = value

    text_encoder = Qwen3TextEncoder(
        vocab_size=int(text_config["vocab_size"]),
        hidden_size=int(text_config["hidden_size"]),
        num_hidden_layers=int(text_config["num_hidden_layers"]),
        num_attention_heads=int(text_config["num_attention_heads"]),
        num_key_value_heads=int(text_config["num_key_value_heads"]),
        intermediate_size=int(text_config["intermediate_size"]),
        max_position_embeddings=int(text_config["max_position_embeddings"]),
        rope_theta=float(
            text_config.get("rope_theta")
            or text_config.get("rope_parameters", {}).get("rope_theta", 5_000_000)
        ),
        rms_norm_eps=float(text_config["rms_norm_eps"]),
        head_dim=int(text_config["head_dim"]),
        attention_bias=bool(text_config.get("attention_bias", False)),
    )
    text_encoder.update(tree_unflatten(list(weights.items())))
    return text_encoder


def load_transformer(
    model_path: str | Path,
    *,
    subfolder: str,
) -> Ideogram4Transformer:
    root = Path(model_path).expanduser()
    config = Ideogram4TransformerConfig.from_dict(
        _load_json(root / subfolder / "config.json")
    )
    raw = _load_safetensors(root / subfolder)
    weights = dequantize_fp8_weight_only(raw)
    transformer = Ideogram4Transformer(config)
    transformer.update(tree_unflatten(list(weights.items())))
    return transformer


def load_vae(model_path: str | Path) -> Flux2VAE:
    return load_flux2_vae(model_path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing weight directory: {directory}")
    weights: dict[str, mx.array] = {}
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {directory}")
    for file in files:
        weights.update(mx.load(str(file)))
    return weights
