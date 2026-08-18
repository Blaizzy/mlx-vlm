"""Weight loading utilities for Z-Image."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .config import ZImageConfig
from .text_encoder import ZImageTextEncoder, sanitize_text_encoder_weights
from .transformer import ZImageTransformer, sanitize_transformer_weights
from .vae import ZImageVAE


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    """Load all safetensors files from a directory."""
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def load_transformer(model_path: str | Path, config: ZImageConfig | None = None) -> ZImageTransformer:
    """Load transformer from checkpoint."""
    config = config or ZImageConfig()
    root = Path(model_path).expanduser()
    transformer = ZImageTransformer(config.transformer)
    weights = _load_safetensors(root / "transformer")
    weights = sanitize_transformer_weights(weights)
    transformer.load_weights(list(weights.items()), strict=False)
    return transformer


def load_text_encoder(model_path: str | Path, config: ZImageConfig | None = None) -> ZImageTextEncoder:
    """Load text encoder from checkpoint."""
    config = config or ZImageConfig()
    root = Path(model_path).expanduser()
    encoder = ZImageTextEncoder(config.text_encoder)
    weights = _load_safetensors(root / "text_encoder")
    weights = sanitize_text_encoder_weights(weights)
    encoder.load_weights(list(weights.items()), strict=False)
    return encoder


def load_vae(model_path: str | Path, config: ZImageConfig | None = None) -> ZImageVAE:
    """Load VAE from checkpoint."""
    config = config or ZImageConfig()
    root = Path(model_path).expanduser()
    vae = ZImageVAE(config.vae)
    weights = _load_safetensors(root / "vae")
    vae.load_weights(list(weights.items()), strict=False)
    return vae


__all__ = ["load_text_encoder", "load_transformer", "load_vae"]
