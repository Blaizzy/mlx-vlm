from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from .config import ErnieImageTransformerConfig


def timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    *,
    max_period: float = 10_000.0,
) -> mx.array:
    timesteps = timesteps.astype(mx.float32).reshape(-1)
    half = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    frequencies = mx.exp(exponent)
    phases = timesteps[:, None] * frequencies[None, :]
    embedding = mx.concatenate([mx.sin(phases), mx.cos(phases)], axis=-1)
    if embedding_dim % 2:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


def rotate_half(hidden_states: mx.array) -> mx.array:
    first, second = mx.split(hidden_states, 2, axis=-1)
    return mx.concatenate([-second, first], axis=-1)


def rope_frequencies(
    position_ids: mx.array,
    *,
    axes_dim: tuple[int, int, int],
    theta: float,
) -> tuple[mx.array, mx.array]:
    if position_ids.shape[-1] != len(axes_dim):
        raise ValueError(
            f"Expected {len(axes_dim)} position axes, got {position_ids.shape}"
        )
    frequencies = []
    positions = position_ids.astype(mx.float32)
    for axis, dim in enumerate(axes_dim):
        omega = 1.0 / (
            theta
            ** (
                mx.arange(0, dim, 2, dtype=mx.float32) / mx.array(dim, dtype=mx.float32)
            )
        )
        angles = positions[..., axis, None] * omega
        angles = mx.stack([angles, angles], axis=-1).reshape(*angles.shape[:-1], dim)
        frequencies.append(angles)
    angles = mx.concatenate(frequencies, axis=-1)
    return mx.cos(angles)[:, None, :, :], mx.sin(angles)[:, None, :, :]


class ErnieImagePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states.transpose(0, 2, 3, 1)
        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0], -1, hidden_states.shape[-1]
        )


class ErnieImageTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(hidden_states)))


class ErnieImageAttention(nn.Module):
    def __init__(self, config: ErnieImageTransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm_q = (
            nn.RMSNorm(self.head_dim, eps=config.eps) if config.qk_layernorm else None
        )
        self.norm_k = (
            nn.RMSNorm(self.head_dim, eps=config.eps) if config.qk_layernorm else None
        )
        self.to_out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        batch, sequence, _ = hidden_states.shape
        shape = (batch, sequence, self.num_heads, self.head_dim)
        queries = self.to_q(hidden_states).reshape(shape)
        keys = self.to_k(hidden_states).reshape(shape)
        values = self.to_v(hidden_states).reshape(shape)
        if self.norm_q is not None and self.norm_k is not None:
            queries = self.norm_q(queries)
            keys = self.norm_k(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = queries * cos + rotate_half(queries) * sin
        keys = keys * cos + rotate_half(keys) * sin
        attended = scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch, sequence, self.num_heads * self.head_dim
        )
        return self.to_out(attended)


class ErnieImageMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_fc2(
            self.up_proj(hidden_states) * nn.gelu(self.gate_proj(hidden_states))
        )


def _modulate(hidden_states: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(mx.float32)
    hidden_states = hidden_states * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return hidden_states.astype(dtype)


class ErnieImageTransformerBlock(nn.Module):
    def __init__(self, config: ErnieImageTransformerConfig) -> None:
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(config.hidden_size, eps=config.eps)
        self.self_attention = ErnieImageAttention(config)
        self.adaLN_mlp_ln = nn.RMSNorm(config.hidden_size, eps=config.eps)
        self.mlp = ErnieImageMLP(config.hidden_size, config.ffn_hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        modulation: tuple[mx.array, ...],
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        attended = self.self_attention(
            _modulate(self.adaLN_sa_ln(hidden_states), shift_msa, scale_msa),
            cos=cos,
            sin=sin,
            mask=mask,
        )
        hidden_states = hidden_states + (
            gate_msa[:, None, :].astype(mx.float32) * attended.astype(mx.float32)
        ).astype(hidden_states.dtype)
        mlp_output = self.mlp(
            _modulate(self.adaLN_mlp_ln(hidden_states), shift_mlp, scale_mlp)
        )
        return hidden_states + (
            gate_mlp[:, None, :].astype(mx.float32) * mlp_output.astype(mx.float32)
        ).astype(hidden_states.dtype)


class ErnieImageFinalNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, affine=False)
        self.linear = nn.Linear(hidden_size, 2 * hidden_size)

    def __call__(self, hidden_states: mx.array, conditioning: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        hidden_states = self.norm(hidden_states)
        return (hidden_states * (1.0 + scale[:, None, :]) + shift[:, None, :]).astype(
            hidden_states.dtype
        )


class ErnieImageTransformer(nn.Module):
    def __init__(self, config: ErnieImageTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ErnieImageTransformerConfig()
        config = self.config
        self.x_embedder = ErnieImagePatchEmbed(
            config.in_channels, config.hidden_size, config.patch_size
        )
        self.text_proj = nn.Linear(config.text_in_dim, config.hidden_size, bias=False)
        self.time_embedding = ErnieImageTimestepEmbedding(config.hidden_size)
        self.adaln_modulation = nn.Linear(config.hidden_size, 6 * config.hidden_size)
        self.layers = [
            ErnieImageTransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.final_norm = ErnieImageFinalNorm(config.hidden_size, config.eps)
        self.final_linear = nn.Linear(config.hidden_size, config.out_channels)

    def _conditioning(self, timestep: mx.array) -> mx.array:
        embedding = timestep_embedding(timestep, self.config.hidden_size).astype(
            timestep.dtype
        )
        return self.time_embedding(embedding)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        timestep: mx.array,
        text_hidden_states: mx.array,
        text_lengths: mx.array,
    ) -> mx.array:
        batch, _, latent_height, latent_width = hidden_states.shape
        image_hidden_states = self.x_embedder(hidden_states)
        image_tokens = image_hidden_states.shape[1]
        text_hidden_states = self.text_proj(text_hidden_states)
        sequence = mx.concatenate([image_hidden_states, text_hidden_states], axis=1)

        y = mx.arange(latent_height, dtype=mx.int32)[:, None]
        x = mx.arange(latent_width, dtype=mx.int32)[None, :]
        grid_y = mx.broadcast_to(y, (latent_height, latent_width)).reshape(-1)
        grid_x = mx.broadcast_to(x, (latent_height, latent_width)).reshape(-1)
        image_positions = mx.stack(
            [
                mx.broadcast_to(text_lengths[:, None], (batch, image_tokens)),
                mx.broadcast_to(grid_y[None, :], (batch, image_tokens)),
                mx.broadcast_to(grid_x[None, :], (batch, image_tokens)),
            ],
            axis=-1,
        )
        text_tokens = text_hidden_states.shape[1]
        text_index = mx.broadcast_to(
            mx.arange(text_tokens, dtype=mx.int32)[None, :],
            (batch, text_tokens),
        )
        text_positions = mx.stack(
            [text_index, mx.zeros_like(text_index), mx.zeros_like(text_index)],
            axis=-1,
        )
        position_ids = mx.concatenate([image_positions, text_positions], axis=1)
        cos, sin = rope_frequencies(
            position_ids,
            axes_dim=self.config.rope_axes_dim,
            theta=self.config.rope_theta,
        )

        valid_text = (
            mx.arange(text_tokens, dtype=mx.int32)[None, :] < text_lengths[:, None]
        )
        valid = mx.concatenate(
            [mx.ones((batch, image_tokens), dtype=mx.bool_), valid_text], axis=1
        )
        mask = mx.where(
            valid[:, None, None, :],
            mx.zeros((batch, 1, 1, sequence.shape[1]), dtype=mx.float32),
            mx.full(
                (batch, 1, 1, sequence.shape[1]),
                -float("inf"),
                dtype=mx.float32,
            ),
        )

        conditioning = self._conditioning(timestep)
        modulation = mx.split(
            self.adaln_modulation(nn.silu(conditioning)).astype(mx.float32),
            6,
            axis=-1,
        )
        for layer in self.layers:
            sequence = layer(
                sequence,
                modulation=modulation,
                cos=cos.astype(sequence.dtype),
                sin=sin.astype(sequence.dtype),
                mask=mask.astype(sequence.dtype),
            )
        sequence = self.final_norm(sequence, conditioning)
        image = self.final_linear(sequence[:, :image_tokens])
        image = image.reshape(
            batch, latent_height, latent_width, self.config.out_channels
        )
        return image.transpose(0, 3, 1, 2)


__all__ = [
    "ErnieImageAttention",
    "ErnieImageTransformer",
    "ErnieImageTransformerBlock",
    "rope_frequencies",
    "rotate_half",
    "timestep_embedding",
]
