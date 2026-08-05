from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import MiniMaxH3TransformerConfig
from .constants import MINIMAX_H3_MODALITY_NUM


@dataclass(frozen=True, slots=True)
class MiniMaxH3TransformerOutput:
    sample: mx.array
    audio_sample: mx.array


@dataclass(frozen=True, slots=True)
class MiniMaxH3AdaLNCache:
    """Materialized AdaLN outputs for one exact per-step timestep table."""

    _owner_token: object
    timestep_values: tuple[float, ...]
    block_modulations: tuple[tuple[mx.array, ...], ...]
    output_modulation: tuple[mx.array, mx.array]

    @property
    def nbytes(self) -> int:
        return sum(
            value.nbytes
            for modulation in self.block_modulations
            for value in modulation
        ) + sum(value.nbytes for value in self.output_modulation)

    def validate(
        self,
        owner_token: object,
        timesteps: mx.array,
        num_blocks: int,
    ) -> None:
        if self._owner_token is not owner_token:
            raise ValueError("AdaLN cache belongs to a different transformer")
        values = tuple(float(value) for value in timesteps.tolist())
        if values != self.timestep_values:
            raise ValueError(
                "AdaLN cache timestep values do not match this transformer call"
            )
        if len(self.block_modulations) != num_blocks:
            raise ValueError("AdaLN cache block count does not match this transformer")


def timestep_embedding(timesteps: mx.array, embedding_dim: int) -> mx.array:
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be one-dimensional, got {timesteps.shape}")
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / half_dim
    frequencies = mx.exp(exponent)
    arguments = timesteps[:, None].astype(mx.float32) * frequencies[None, :]
    embedding = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
    if embedding_dim % 2:
        embedding = mx.pad(embedding, ((0, 0), (0, 1)))
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_dim, bias=True)
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class MiniMaxH3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        self.rope_theta = float(rope_theta)

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # Packed positions are retained in float64 on MLX's CPU stream. Cast
        # and materialize before moving into the Metal transformer graph.
        if position_ids.dtype == mx.float64:
            with mx.stream(mx.cpu):
                position_ids = position_ids.astype(mx.float32)
                mx.eval(position_ids)
        else:
            position_ids = position_ids.astype(mx.float32)
        exponent = mx.arange(
            0,
            2 * self.rope_freq_dim,
            2,
            dtype=mx.float32,
        ) / (2 * self.rope_freq_dim)
        inv_freq = 1.0 / mx.power(self.rope_theta, exponent)
        frequencies = position_ids[..., None] * inv_freq.reshape(1, 1, -1)
        frequencies = mx.concatenate(
            [frequencies[:, 0], frequencies[:, 1], frequencies[:, 2]],
            axis=-1,
        )
        frequencies = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(frequencies), mx.sin(frequencies)


def apply_rotary_emb(
    hidden_states: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.astype(hidden_states.dtype)[None, :, None, :]
    sin = sin.astype(hidden_states.dtype)[None, :, None, :]
    first, second = mx.split(hidden_states_rotary, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    hidden_states_rotary = hidden_states_rotary * cos + rotated * sin
    return mx.concatenate([hidden_states_rotary, hidden_states_pass], axis=-1)


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
        )

    def __call__(self, temb: mx.array) -> tuple[mx.array, ...]:
        temb = self.linear(nn.silu(temb).astype(self.linear.weight.dtype))
        temb = temb.reshape(-1, 6 * self.hidden_size)
        return tuple(mx.split(temb, 6, axis=-1))


class MiniMaxH3AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def project(self, temb: mx.array) -> tuple[mx.array, mx.array]:
        modulation = self.linear(nn.silu(temb).astype(self.linear.weight.dtype))
        return tuple(mx.split(modulation, 2, axis=-1))

    def apply(
        self,
        hidden_states: mx.array,
        modulation: tuple[mx.array, mx.array],
        timestep_indices: mx.array,
    ) -> mx.array:
        shift, scale = modulation
        shift = mx.take(shift, timestep_indices, axis=0)
        scale = mx.take(scale, timestep_indices, axis=0)
        return self.norm(hidden_states) * (1.0 + scale) + shift

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        timestep_indices: mx.array,
    ) -> mx.array:
        return self.apply(hidden_states, self.project(temb), timestep_indices)


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = [
            nn.Linear(self.inner_dim, hidden_size, bias=False),
            nn.Dropout(0.0),
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dim,
        )
        key = self.to_k(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dim,
        )
        value = self.to_v(hidden_states).reshape(
            batch_size,
            sequence_length,
            self.heads,
            self.head_dim,
        )
        query = self.norm_q(query)
        key = self.norm_k(key)
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        if attention_mask is not None:
            attention_mask = mx.where(
                attention_mask,
                mx.array(0.0, dtype=query.dtype),
                mx.array(-mx.inf, dtype=query.dtype),
            )[None, None]
        output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            self.inner_dim,
        )
        return self.to_out[1](self.to_out[0](output))


class _SwiGLUProjection(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states, gate = mx.split(self.proj(hidden_states), 2, axis=-1)
        return hidden_states * nn.silu(gate)


class _Identity(nn.Module):
    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states


class MiniMaxH3FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int) -> None:
        super().__init__()
        self.net = [
            _SwiGLUProjection(hidden_size, ffn_dim),
            _Identity(),
            nn.Linear(ffn_dim, hidden_size, bias=False),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig) -> None:
        super().__init__()
        self.refiner_blocks = [
            MiniMaxH3TokenRefinerBlock(config) for _ in range(config.num_refiner_layers)
        ]
        self.final_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.final_norm_eps,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = MiniMaxH3Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = MiniMaxH3FeedForward(config.hidden_size, config.ffn_dim)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            config.time_embed_dim,
            config.hidden_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        modulation: tuple[mx.array, ...],
        adaln_indices: mx.array,
        rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        shift_msa = mx.take(shift_msa, adaln_indices, axis=0)
        scale_msa = mx.take(scale_msa, adaln_indices, axis=0)
        gate_msa = mx.take(gate_msa, adaln_indices, axis=0)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * self.attn(
            norm_hidden_states,
            rotary_emb,
            attention_mask,
        )

        shift_mlp = mx.take(shift_mlp, adaln_indices, axis=0)
        scale_mlp = mx.take(scale_mlp, adaln_indices, axis=0)
        gate_mlp = mx.take(gate_mlp, adaln_indices, axis=0)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp) + shift_mlp
        return hidden_states + gate_mlp * self.ff(norm_hidden_states)


class MiniMaxH3Transformer(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3TransformerConfig | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise ValueError("pass a config or keyword fields, not both")
        self.config = config or MiniMaxH3TransformerConfig(**config_kwargs)
        config = self.config
        self.proj_in = nn.Linear(
            config.video_patch_dim,
            config.hidden_size,
            bias=True,
        )
        self.audio_proj_in = nn.Linear(
            config.audio_in_channels,
            config.hidden_size,
            bias=True,
        )
        self.context_embedder = nn.Linear(
            config.text_dim,
            config.hidden_size,
            bias=True,
        )
        self.time_embedder = TimestepEmbedding(
            config.freq_dim,
            config.time_embed_hidden_dim,
            config.time_embed_dim,
        )
        self.rope = MiniMaxH3RotaryPosEmbed(
            config.rope_freq_dim,
            config.rope_theta,
        )
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.transformer_blocks = [
            MiniMaxH3TransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            config.hidden_size,
            config.time_embed_dim,
            config.final_norm_eps,
        )
        self.proj_out = nn.Linear(
            config.hidden_size,
            config.video_patch_dim,
            bias=True,
        )
        self.audio_proj_out = nn.Linear(
            config.hidden_size,
            config.audio_in_channels,
            bias=True,
        )
        self._adaln_cache_token = object()

    @property
    def adaln_weights_available(self) -> bool:
        linears = [block.adaln_proj.linear for block in self.transformer_blocks]
        linears.append(self.norm_out.linear)
        return all(hasattr(linear, "weight") for linear in linears)

    def _embed_timesteps(self, timesteps: mx.array) -> mx.array:
        temb = timestep_embedding(timesteps, self.config.freq_dim)
        return self.time_embedder(temb.astype(self.time_embedder.linear_1.weight.dtype))

    def build_adaln_cache(
        self,
        timesteps: mx.array,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> MiniMaxH3AdaLNCache:
        """Project and materialize AdaLN for one unchanged timestep tensor."""
        if not self.adaln_weights_available:
            raise RuntimeError(
                "AdaLN projection weights were dropped; this timestep table "
                "cannot be cached without reloading the transformer"
            )
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must be one-dimensional, got {timesteps.shape}"
            )

        # Preserve the live path's shape, order, and dtypes. Evaluating each
        # table severs its lazy graph from the projection before those weights
        # can be released.
        temb = self._embed_timesteps(timesteps)
        mx.eval(temb)
        block_modulations = []
        total = len(self.transformer_blocks) + 1
        if progress_callback is not None:
            progress_callback(0, total)
        for block_index, block in enumerate(self.transformer_blocks, start=1):
            modulation = block.adaln_proj(temb)
            mx.eval(modulation)
            block_modulations.append(modulation)
            if progress_callback is not None:
                progress_callback(block_index, total)
        output_modulation = self.norm_out.project(temb)
        mx.eval(output_modulation)
        if progress_callback is not None:
            progress_callback(total, total)
        return MiniMaxH3AdaLNCache(
            _owner_token=self._adaln_cache_token,
            timestep_values=tuple(float(value) for value in timesteps.tolist()),
            block_modulations=tuple(block_modulations),
            output_modulation=output_modulation,
        )

    def drop_adaln_weights(self) -> int:
        """Drop projections only after every required cache is materialized."""
        freed = 0
        linears = [block.adaln_proj.linear for block in self.transformer_blocks]
        linears.append(self.norm_out.linear)
        for linear in linears:
            for name in ("weight", "bias", "scales", "biases"):
                value = getattr(linear, name, None)
                if isinstance(value, mx.array):
                    freed += value.nbytes
                    delattr(linear, name)
        return freed

    def __call__(
        self,
        hidden_states: mx.array,
        audio_hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        timestep_indices: mx.array,
        token_tags: mx.array,
        position_ids: mx.array,
        video_indices: mx.array,
        audio_indices: mx.array,
        text_indices: mx.array,
        adaln_cache: MiniMaxH3AdaLNCache | None = None,
    ) -> MiniMaxH3TransformerOutput:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"position_ids must have shape (sequence_length, 3), got {position_ids.shape}"
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (
            sequence_length,
        ):
            raise ValueError(
                "token_tags and timestep_indices must match the packed sequence length"
            )
        if adaln_cache is None:
            if not self.adaln_weights_available:
                raise RuntimeError(
                    "AdaLN projection weights were dropped; pass the matching "
                    "materialized AdaLN cache"
                )
            temb = self._embed_timesteps(timestep)
        else:
            adaln_cache.validate(
                self._adaln_cache_token,
                timestep,
                len(self.transformer_blocks),
            )
            temb = None
        rotary_emb = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.astype(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(
            audio_hidden_states.astype(self.audio_proj_in.weight.dtype)
        )
        text_embeds = self.context_embedder(
            encoder_hidden_states.astype(self.context_embedder.weight.dtype)
        )
        text_embeds = self.token_refiner(text_embeds)
        packed = mx.zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1]),
            dtype=text_embeds.dtype,
        )
        packed = packed.at[:, text_indices, :].add(text_embeds)
        packed = packed.at[:, video_indices, :].add(video_embeds.astype(packed.dtype))
        packed = packed.at[:, audio_indices, :].add(audio_embeds.astype(packed.dtype))

        adaln_indices = (
            timestep_indices * MINIMAX_H3_MODALITY_NUM + mx.maximum(token_tags, 0)
        ).astype(mx.int32)
        is_padding = token_tags < 0
        attention_mask = None
        if bool(mx.any(is_padding).item()):
            attention_mask = is_padding[None, :] == is_padding[:, None]

        for block_index, block in enumerate(self.transformer_blocks):
            modulation = (
                block.adaln_proj(temb)
                if adaln_cache is None
                else adaln_cache.block_modulations[block_index]
            )
            packed = block(
                packed,
                modulation,
                adaln_indices,
                rotary_emb,
                attention_mask,
            )

        output_modulation = (
            self.norm_out.project(temb)
            if adaln_cache is None
            else adaln_cache.output_modulation
        )
        packed = self.norm_out.apply(
            packed,
            output_modulation,
            timestep_indices.astype(mx.int32),
        )
        packed = packed.astype(self.proj_out.weight.dtype)
        video_output = mx.take(self.proj_out(packed), video_indices, axis=1)
        audio_output = mx.take(
            self.audio_proj_out(packed),
            audio_indices,
            axis=1,
        )
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )


__all__ = [
    "MiniMaxH3AdaLNCache",
    "MiniMaxH3AdaLayerNormModulation",
    "MiniMaxH3AdaLayerNormOut",
    "MiniMaxH3Attention",
    "MiniMaxH3FeedForward",
    "MiniMaxH3RotaryPosEmbed",
    "MiniMaxH3TokenRefiner",
    "MiniMaxH3Transformer",
    "MiniMaxH3TransformerBlock",
    "MiniMaxH3TransformerOutput",
    "apply_rotary_emb",
    "timestep_embedding",
]
