from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class MiniMaxH3TransformerConfig:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        integer_fields = (
            "num_attention_heads",
            "attention_head_dim",
            "hidden_size",
            "num_layers",
            "ffn_dim",
            "in_channels",
            "audio_in_channels",
            "text_dim",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
            "rope_freq_dim",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.num_refiner_layers < 0:
            raise ValueError(
                "num_refiner_layers must be non-negative, "
                f"got {self.num_refiner_layers}"
            )
        if len(self.patch_size) != 3 or any(size <= 0 for size in self.patch_size):
            raise ValueError(
                f"patch_size must contain three positive values, got {self.patch_size}"
            )
        rotary_dim = 2 * 3 * self.rope_freq_dim
        if rotary_dim > self.attention_head_dim:
            raise ValueError(
                f"rotary dimension {rotary_dim} exceeds attention head dimension "
                f"{self.attention_head_dim}"
            )

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        return self.in_channels * patch_volume

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MiniMaxH3TransformerConfig":
        names = {field.name for field in fields(cls)}
        values = {name: config[name] for name in names if name in config}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoVAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 24
    block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    spatial_padding_mode: str = "reflect"
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_num_register_tokens: int = 4
    decoder_ffn_mult: int = 4
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    latents_mean: tuple[float, ...] = (0.0,) * 24
    latents_std: tuple[float, ...] = (1.0,) * 24

    def __post_init__(self) -> None:
        levels = len(self.block_out_channels)
        if not levels or len(self.spatial_downsample_factors) != levels:
            raise ValueError("spatial downsample factors must match encoder levels")
        if len(self.temporal_downsample_factors) != levels:
            raise ValueError("temporal downsample factors must match encoder levels")
        if len(self.latents_mean) != self.latent_channels:
            raise ValueError("latents_mean must match latent_channels")
        if len(self.latents_std) != self.latent_channels:
            raise ValueError("latents_std must match latent_channels")
        rotary_dim = int(self.decoder_attention_head_dim * self.decoder_rope_dim_ratio)
        if rotary_dim % 6:
            raise ValueError("decoder rotary dimension must be divisible by 6")

    @property
    def spatial_compression_ratio(self) -> int:
        result = 1
        for factor in self.spatial_downsample_factors:
            result *= factor
        return result

    @property
    def temporal_compression_ratio(self) -> int:
        result = 1
        for factor in self.temporal_downsample_factors:
            result *= factor
        return result

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MiniMaxH3VideoVAEConfig":
        names = {field.name for field in fields(cls)}
        values = {name: config[name] for name in names if name in config}
        tuple_fields = (
            "block_out_channels",
            "spatial_downsample_factors",
            "temporal_downsample_factors",
            "latents_mean",
            "latents_std",
        )
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class MiniMaxH3AudioVAEConfig:
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    latent_dim: int = 2048
    latent_channels: int = 32
    num_attention_heads: int = 8
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    decoder_kernel_sizes: tuple[int, ...] = (9, 9, 4, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    sampling_rate: int = 32000
    latents_mean: tuple[float, ...] = (0.0,) * 32
    latents_std: tuple[float, ...] = (1.0,) * 32

    def __post_init__(self) -> None:
        if self.decoder_hop_length != self.hop_length:
            raise ValueError("decoder rates must upsample by the encoder hop length")
        if self.latent_dim % self.latent_channels:
            raise ValueError("latent_dim must be a multiple of latent_channels")
        if self.latent_dim % self.num_attention_heads:
            raise ValueError("latent_dim must be divisible by num_attention_heads")
        if self.latent_dim // self.num_attention_heads < self.latent_channels:
            raise ValueError("attention head width must be at least latent_channels")
        if len(self.decoder_rates) != len(self.decoder_kernel_sizes):
            raise ValueError("decoder rates and kernel sizes must have equal lengths")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("resblock kernels and dilation groups must match")
        if len(self.latents_mean) != self.latent_channels:
            raise ValueError("latents_mean must match latent_channels")
        if len(self.latents_std) != self.latent_channels:
            raise ValueError("latents_std must match latent_channels")

    @property
    def hop_length(self) -> int:
        result = 1
        for rate in self.encoder_rates:
            result *= rate
        return result

    @property
    def decoder_hop_length(self) -> int:
        result = 1
        for rate in self.decoder_rates:
            result *= rate
        return result

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MiniMaxH3AudioVAEConfig":
        names = {field.name for field in fields(cls)}
        values = {name: config[name] for name in names if name in config}
        tuple_fields = (
            "encoder_rates",
            "decoder_rates",
            "decoder_kernel_sizes",
            "resblock_kernel_sizes",
            "latents_mean",
            "latents_std",
        )
        for name in tuple_fields:
            if name in values:
                values[name] = tuple(values[name])
        if "resblock_dilation_sizes" in values:
            values["resblock_dilation_sizes"] = tuple(
                tuple(group) for group in values["resblock_dilation_sizes"]
            )
        return cls(**values)


__all__ = [
    "MiniMaxH3AudioVAEConfig",
    "MiniMaxH3TransformerConfig",
    "MiniMaxH3VideoVAEConfig",
]
