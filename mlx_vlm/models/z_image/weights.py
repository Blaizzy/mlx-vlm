from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten

from .config import ZImageConfig
from .text_encoder import ZImageTextEncoder, sanitize_text_encoder_weights
from .transformer import ZImageTransformer, sanitize_transformer_weights
from .vae import ZImageVAE, sanitize_vae_weights


def _load_safetensors(
    directory: Path,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    index_path = directory / "model.safetensors.index.json"
    metadata = (
        json.loads(index_path.read_text()).get("metadata", {})
        if index_path.exists()
        else {}
    )
    return weights, metadata


def _apply_weights(
    model: nn.Module,
    weights: dict[str, mx.array],
    metadata: dict[str, Any],
) -> nn.Module:
    quantized_paths = {
        key[: -len(".scales")] for key in weights if key.endswith(".scales")
    }
    if quantized_paths:
        mode = str(metadata.get("quantization_mode", ""))
        bits = int(metadata.get("quantization_level", 0))
        group_size = int(metadata.get("quantization_group_size", 0))
        supported = {
            "affine": None,
            "mxfp4": (4, 32),
            "mxfp8": (8, 32),
            "nvfp4": (4, 16),
        }
        expected = supported.get(mode)
        if mode not in supported or (
            expected is not None and expected != (bits, group_size)
        ):
            raise ValueError(
                "Unsupported Z-Image quantization metadata: "
                f"mode={mode!r}, bits={bits}, group_size={group_size}"
            )
        modules = dict(model.named_modules())
        missing = sorted(path for path in quantized_paths if path not in modules)
        if missing:
            raise ValueError(f"Quantized weights have no matching module: {missing[0]}")
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            mode=mode,
            class_predicate=lambda path, module: path in quantized_paths,
        )
        model.quantization_config = {
            "group_size": group_size,
            "bits": bits,
            "mode": mode,
        }
    else:
        model.quantization_config = None
    model.update(tree_unflatten(list(weights.items())), strict=True)
    return model


def load_transformer(
    model_path: str | Path, config: ZImageConfig | None = None
) -> ZImageTransformer:
    root = Path(model_path).expanduser()
    config = config or ZImageConfig.from_model_path(root)
    transformer = ZImageTransformer(config.transformer)
    weights, metadata = _load_safetensors(root / "transformer")
    weights = sanitize_transformer_weights(weights)
    return _apply_weights(transformer, weights, metadata)


def load_text_encoder(
    model_path: str | Path, config: ZImageConfig | None = None
) -> ZImageTextEncoder:
    root = Path(model_path).expanduser()
    config = config or ZImageConfig.from_model_path(root)
    encoder = ZImageTextEncoder(config.text_encoder)
    weights, metadata = _load_safetensors(root / "text_encoder")
    weights = sanitize_text_encoder_weights(weights)
    return _apply_weights(encoder, weights, metadata)


def load_vae(model_path: str | Path, config: ZImageConfig | None = None) -> ZImageVAE:
    root = Path(model_path).expanduser()
    config = config or ZImageConfig.from_model_path(root)
    vae = ZImageVAE(config.vae)
    weights, metadata = _load_safetensors(root / "vae")
    weights = sanitize_vae_weights(
        weights,
        source_layout=(False if metadata.get("mlx_vlm_format") == "z_image" else None),
    )
    return _apply_weights(vae, weights, metadata)


__all__ = ["load_text_encoder", "load_transformer", "load_vae"]
