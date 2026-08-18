"""Architecture configuration for Z-Image."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ZImageTransformerConfig:
    hidden_size: int = 3840
    num_attention_heads: int = 30
    num_key_value_heads: int = 30
    intermediate_size: int = 10240
    in_channels: int = 16
    text_embed_dim: int = 2560
    num_hidden_layers: int = 30
    n_refiner_layers: int = 2
    n_context_refiner_layers: int = 2
    patch_size: int = 2
    f_patch_size: int = 1
    adaln_embed_dim: int = 256
    rope_sections: tuple[int, ...] = (32, 48, 48)
    rope_theta: float = 256.0
    norm_eps: float = 1e-5
    timestep_scale: float = 1000.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ZImageTransformerConfig:
        defaults = cls()
        dim = int(config.get("dim", defaults.hidden_size))
        return cls(
            hidden_size=dim,
            num_attention_heads=int(
                config.get("n_heads", defaults.num_attention_heads)
            ),
            num_key_value_heads=int(
                config.get("n_kv_heads", defaults.num_key_value_heads)
            ),
            intermediate_size=int(config.get("intermediate_size", dim * 8 // 3)),
            in_channels=int(config.get("in_channels", defaults.in_channels)),
            text_embed_dim=int(config.get("cap_feat_dim", defaults.text_embed_dim)),
            num_hidden_layers=int(config.get("n_layers", defaults.num_hidden_layers)),
            n_refiner_layers=int(
                config.get("n_refiner_layers", defaults.n_refiner_layers)
            ),
            n_context_refiner_layers=int(
                config.get(
                    "n_context_refiner_layers",
                    config.get(
                        "n_refiner_layers",
                        defaults.n_context_refiner_layers,
                    ),
                )
            ),
            patch_size=int(config.get("all_patch_size", (defaults.patch_size,))[0]),
            f_patch_size=int(
                config.get("all_f_patch_size", (defaults.f_patch_size,))[0]
            ),
            adaln_embed_dim=int(
                config.get("adaln_embed_dim", defaults.adaln_embed_dim)
            ),
            rope_sections=tuple(config.get("axes_dims", defaults.rope_sections)),
            rope_theta=float(config.get("rope_theta", defaults.rope_theta)),
            norm_eps=float(config.get("norm_eps", defaults.norm_eps)),
            timestep_scale=float(config.get("t_scale", defaults.timestep_scale)),
        )


@dataclass(frozen=True, slots=True)
class ZImageTextEncoderConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 9728
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    head_dim: int = 128

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ZImageTextEncoderConfig:
        defaults = cls()
        return cls(
            vocab_size=int(config.get("vocab_size", defaults.vocab_size)),
            hidden_size=int(config.get("hidden_size", defaults.hidden_size)),
            num_hidden_layers=int(
                config.get("num_hidden_layers", defaults.num_hidden_layers)
            ),
            num_attention_heads=int(
                config.get("num_attention_heads", defaults.num_attention_heads)
            ),
            num_key_value_heads=int(
                config.get("num_key_value_heads", defaults.num_key_value_heads)
            ),
            intermediate_size=int(
                config.get("intermediate_size", defaults.intermediate_size)
            ),
            rms_norm_eps=float(config.get("rms_norm_eps", defaults.rms_norm_eps)),
            rope_theta=float(config.get("rope_theta", defaults.rope_theta)),
            max_position_embeddings=int(
                config.get(
                    "max_position_embeddings",
                    defaults.max_position_embeddings,
                )
            ),
            head_dim=int(config.get("head_dim", defaults.head_dim)),
        )


@dataclass(frozen=True, slots=True)
class ZImageVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ZImageVAEConfig:
        defaults = cls()
        return cls(
            in_channels=int(config.get("in_channels", defaults.in_channels)),
            out_channels=int(config.get("out_channels", defaults.out_channels)),
            latent_channels=int(
                config.get("latent_channels", defaults.latent_channels)
            ),
            block_out_channels=tuple(
                config.get("block_out_channels", defaults.block_out_channels)
            ),
            layers_per_block=int(
                config.get("layers_per_block", defaults.layers_per_block)
            ),
            scaling_factor=float(config.get("scaling_factor", defaults.scaling_factor)),
            shift_factor=float(config.get("shift_factor", defaults.shift_factor)),
        )


@dataclass(frozen=True, slots=True)
class ZImageConfig:
    transformer: ZImageTransformerConfig = ZImageTransformerConfig()
    text_encoder: ZImageTextEncoderConfig = ZImageTextEncoderConfig()
    vae: ZImageVAEConfig = ZImageVAEConfig()
    default_steps: int = 9
    default_guidance: float = 0.0
    scheduler_shift: float = 3.0

    @classmethod
    def from_model_path(cls, model_path: str | Path) -> ZImageConfig:
        root = Path(model_path).expanduser()

        def load(relative: str) -> dict[str, Any]:
            path = root / relative
            if not path.exists():
                raise FileNotFoundError(f"Missing Z-Image config: {path}")
            return json.loads(path.read_text())

        scheduler = load("scheduler/scheduler_config.json")
        defaults = cls()
        return cls(
            transformer=ZImageTransformerConfig.from_dict(
                load("transformer/config.json")
            ),
            text_encoder=ZImageTextEncoderConfig.from_dict(
                load("text_encoder/config.json")
            ),
            vae=ZImageVAEConfig.from_dict(load("vae/config.json")),
            scheduler_shift=float(scheduler.get("shift", defaults.scheduler_shift)),
        )


def detect_z_image_layout(path: str | Path) -> bool:
    """Detect if a local path is a Z-Image checkpoint."""
    root = Path(path).expanduser()
    return (
        (root / "model_index.json").exists()
        and (root / "transformer" / "config.json").exists()
        and bool(list((root / "transformer").glob("*.safetensors")))
        and (root / "text_encoder" / "config.json").exists()
        and bool(list((root / "text_encoder").glob("*.safetensors")))
        and (root / "vae" / "config.json").exists()
        and bool(list((root / "vae").glob("*.safetensors")))
        and (root / "scheduler" / "scheduler_config.json").exists()
        and (root / "tokenizer" / "tokenizer.json").exists()
    )


__all__ = [
    "ZImageConfig",
    "ZImageTextEncoderConfig",
    "ZImageTransformerConfig",
    "ZImageVAEConfig",
    "detect_z_image_layout",
]
