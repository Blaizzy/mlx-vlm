"""Default architecture configuration for Z-Image Turbo (no config.json in checkpoint)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ZImageTransformerConfig:
    hidden_size: int = 3072
    num_attention_heads: int = 24
    intermediate_size: int = 8192
    in_channels: int = 16
    text_embed_dim: int = 3584
    num_hidden_layers: int = 30
    n_refiner_layers: int = 2
    n_context_refiner_layers: int = 2
    patch_size: int = 2
    f_patch_size: int = 1
    adaln_embed_dim: int = 256
    rope_sections: tuple[int, ...] = (32, 48, 48)
    rope_theta: float = 256.0
    norm_eps: float = 1e-5


@dataclass(frozen=True, slots=True)
class ZImageTextEncoderConfig:
    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 36
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 131072
    head_dim: int = 128


@dataclass(frozen=True, slots=True)
class ZImageVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159


@dataclass(frozen=True, slots=True)
class ZImageConfig:
    transformer: ZImageTransformerConfig = ZImageTransformerConfig()
    text_encoder: ZImageTextEncoderConfig = ZImageTextEncoderConfig()
    vae: ZImageVAEConfig = ZImageVAEConfig()
    default_steps: int = 9
    default_guidance: float = 1.0


def detect_z_image_layout(path: str | Path) -> bool:
    """Detect if a local path is a Z-Image checkpoint."""
    root = Path(path).expanduser()
    return (
        (root / "transformer" / "model.safetensors.index.json").exists()
        and (root / "text_encoder" / "model.safetensors.index.json").exists()
        and (root / "vae" / "model.safetensors.index.json").exists()
        and (root / "tokenizer" / "tokenizer.json").exists()
    )


__all__ = [
    "ZImageConfig",
    "ZImageTextEncoderConfig",
    "ZImageTransformerConfig",
    "ZImageVAEConfig",
    "detect_z_image_layout",
]
