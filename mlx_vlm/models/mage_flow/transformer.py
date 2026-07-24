from __future__ import annotations

import math
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn


def timestep_embedding(
    timesteps: mx.array,
    dim: int = 256,
    *,
    max_period: int = 10000,
    scale: float = 1000.0,
    dtype=mx.bfloat16,
) -> mx.array:
    """Mage-Flow's training-compatible, bf16-rounded timestep embedding."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    frequencies = mx.exp(exponent).astype(dtype).astype(mx.float32)
    args = timesteps.reshape(-1, 1).astype(mx.float32) * frequencies.reshape(1, -1)
    args = scale * args
    # The reference requests flip_sin_to_cos=True.
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.pad(embedding, ((0, 0), (0, 1)))
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, timesteps: mx.array, dtype) -> mx.array:
        x = timestep_embedding(timesteps, dtype=dtype).astype(dtype)
        return self.linear_2(nn.silu(self.linear_1(x)))


class TimeTextEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(hidden_size)

    def __call__(self, timesteps: mx.array, dtype) -> mx.array:
        return self.timestep_embedder(timesteps, dtype)


class Modulation(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.silu(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.linear_in = nn.Linear(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(nn.gelu_approx(self.linear_in(x)))


def _axis_frequencies(positions: mx.array, dim: int, theta: float) -> mx.array:
    exponent = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    inverse = 1.0 / mx.power(mx.array(theta, dtype=mx.float32), exponent)
    return positions.reshape(-1, 1).astype(mx.float32) * inverse.reshape(1, -1)


def image_rope_frequencies(
    image_shapes: Sequence[tuple[int, int, int]],
    *,
    axes_dim: tuple[int, int, int] = (16, 56, 56),
    theta: float = 10000.0,
) -> tuple[mx.array, mx.array]:
    """Build centered 2D multi-scale RoPE for target/reference latent grids."""
    all_frequencies = []
    for frame_index, (frames, height, width) in enumerate(image_shapes):
        frame_pos = mx.arange(frame_index, frame_index + frames, dtype=mx.float32)
        height_pos = mx.arange(height, dtype=mx.float32) - (height - height // 2)
        width_pos = mx.arange(width, dtype=mx.float32) - (width - width // 2)

        frame_freq = _axis_frequencies(frame_pos, axes_dim[0], theta)
        height_freq = _axis_frequencies(height_pos, axes_dim[1], theta)
        width_freq = _axis_frequencies(width_pos, axes_dim[2], theta)
        frame_freq = mx.broadcast_to(
            frame_freq[:, None, None, :],
            (frames, height, width, frame_freq.shape[-1]),
        )
        height_freq = mx.broadcast_to(
            height_freq[None, :, None, :],
            (frames, height, width, height_freq.shape[-1]),
        )
        width_freq = mx.broadcast_to(
            width_freq[None, None, :, :],
            (frames, height, width, width_freq.shape[-1]),
        )
        all_frequencies.append(
            mx.concatenate([frame_freq, height_freq, width_freq], axis=-1).reshape(
                -1, sum(axes_dim) // 2
            )
        )
    frequencies = mx.concatenate(all_frequencies, axis=0)
    return mx.cos(frequencies), mx.sin(frequencies)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply adjacent-pair complex RoPE to ``[B, L, H, D]``."""
    even = x[..., 0::2].astype(mx.float32)
    odd = x[..., 1::2].astype(mx.float32)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    real = even * cos - odd * sin
    imag = even * sin + odd * cos
    return mx.stack([real, imag], axis=-1).reshape(x.shape).astype(x.dtype)


class JointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.Linear(dim, dim)
        self.to_add_out = nn.Linear(dim, dim)

    def _reshape(self, x: mx.array) -> mx.array:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    def __call__(
        self,
        image: mx.array,
        text: mx.array,
        rope: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        img_q = self.norm_q(self._reshape(self.to_q(image)))
        img_k = self.norm_k(self._reshape(self.to_k(image)))
        img_v = self._reshape(self.to_v(image))
        txt_q = self.norm_added_q(self._reshape(self.add_q_proj(text)))
        txt_k = self.norm_added_k(self._reshape(self.add_k_proj(text)))
        txt_v = self._reshape(self.add_v_proj(text))

        img_q = apply_rotary(img_q, *rope)
        img_k = apply_rotary(img_k, *rope)
        text_length = text.shape[1]
        query = mx.concatenate([txt_q, img_q], axis=1).transpose(0, 2, 1, 3)
        key = mx.concatenate([txt_k, img_k], axis=1).transpose(0, 2, 1, 3)
        value = mx.concatenate([txt_v, img_v], axis=1).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            image.shape[0], text.shape[1] + image.shape[1], -1
        )
        txt_output = self.to_add_out(output[:, :text_length])
        img_output = self.to_out(output[:, text_length:])
        return img_output, txt_output


def _modulate(x: mx.array, parameters: mx.array) -> tuple[mx.array, mx.array]:
    shift, scale, gate = mx.split(parameters, 3, axis=-1)
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]


class MageFlowTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.img_mod = Modulation(dim)
        self.txt_mod = Modulation(dim)
        self.img_norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.txt_norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = JointAttention(dim, num_heads, head_dim)
        self.img_norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.img_mlp = FeedForward(dim)
        self.txt_mlp = FeedForward(dim)

    def __call__(
        self,
        image: mx.array,
        text: mx.array,
        temb: mx.array,
        rope: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, mx.array]:
        img_mod1, img_mod2 = mx.split(self.img_mod(temb), 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(self.txt_mod(temb), 2, axis=-1)
        img_norm, img_gate1 = _modulate(self.img_norm1(image), img_mod1)
        txt_norm, txt_gate1 = _modulate(self.txt_norm1(text), txt_mod1)
        img_attn, txt_attn = self.attn(img_norm, txt_norm, rope)
        image = image + img_gate1 * img_attn
        text = text + txt_gate1 * txt_attn
        img_norm, img_gate2 = _modulate(self.img_norm2(image), img_mod2)
        txt_norm, txt_gate2 = _modulate(self.txt_norm2(text), txt_mod2)
        image = image + img_gate2 * self.img_mlp(img_norm)
        text = text + txt_gate2 * self.txt_mlp(txt_norm)
        return image, text


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        # The reference implementation names these chunks scale, then shift.
        scale, shift = mx.split(self.linear(nn.silu(temb)), 2, axis=-1)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


class MageFlowTransformer(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 128,
        out_channels: int = 128,
        context_in_dim: int = 2560,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 12,
        axes_dim: tuple[int, int, int] = (16, 56, 56),
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        if sum(axes_dim) != head_dim:
            raise ValueError(
                f"axes_dim must sum to head_dim ({head_dim}), got {axes_dim}"
            )
        self.axes_dim = axes_dim
        self.theta = theta
        self.img_in = nn.Linear(in_channels, hidden_size)
        self.txt_norm = nn.RMSNorm(context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.time_text_embed = TimeTextEmbedding(hidden_size)
        self.transformer_blocks = [
            MageFlowTransformerBlock(hidden_size, num_heads, head_dim)
            for _ in range(depth)
        ]
        self.norm_out = AdaptiveLayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, out_channels)

    def __call__(
        self,
        *,
        img: mx.array,
        txt: mx.array,
        timesteps: mx.array,
        img_shapes: Sequence[tuple[int, int, int]],
    ) -> mx.array:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("img and txt must both have shape [B, L, D]")
        rope = image_rope_frequencies(
            img_shapes, axes_dim=self.axes_dim, theta=self.theta
        )
        image = self.img_in(img)
        text = self.txt_in(self.txt_norm(txt))
        temb = self.time_text_embed(timesteps.astype(image.dtype), image.dtype)
        for block in self.transformer_blocks:
            image, text = block(image, text, temb, rope)
        return self.proj_out(self.norm_out(image, temb))


__all__ = [
    "MageFlowTransformer",
    "apply_rotary",
    "image_rope_frequencies",
    "timestep_embedding",
]
