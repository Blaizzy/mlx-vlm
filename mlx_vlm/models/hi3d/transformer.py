"""Sparse latent flow transformer of Hi3D (Sparc3D), adapted from TRELLIS.

Features are ``(B, N, C)``: every batch row shares one voxel grid, which is
how the guided and unguided passes of a sampling step run together.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .sparse import Grid, SparseConv3d, downsample_mean


def layer_norm32(x, weight=None, bias=None, eps=1e-6):
    """Layer norm with float32 statistics, returned in the input dtype."""
    return mx.fast.layer_norm(x.astype(mx.float32), weight, bias, eps).astype(x.dtype)


class LayerNorm32(nn.Module):
    def __init__(self, channels, affine, eps=1e-6):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = mx.ones((channels,))
            self.bias = mx.zeros((channels,))

    def __call__(self, x):
        return layer_norm32(
            x, getattr(self, "weight", None), getattr(self, "bias", None), self.eps
        )


def rope_tables(coords, head_dim):
    """Cosine and sine of the spatial rotary phases, ``(N, head_dim // 2)``.

    Each of the three axes owns ``head_dim // 6`` frequencies. Grids reaching
    coordinate 64 stretch the frequencies the way the released model was
    trained for longer inference grids.
    """
    freq_dim = head_dim // 3 // 2
    freqs = 1.0 / (10000 ** (mx.arange(freq_dim, dtype=mx.float32) / freq_dim))
    max_length = int(mx.max(coords).item())
    if max_length >= 64:
        train_length, infer_length = 64, (64 + max_length) // 2
        wavelengths = freqs * train_length / (2 * math.pi)
        ratio = (wavelengths - 1) / 23
        ratio = ratio + (1 - ratio) * train_length / infer_length
        ratio = mx.where(wavelengths > 24, 1.0, ratio)
        ratio = mx.where(wavelengths < 1, train_length / infer_length, ratio)
        freqs = freqs * ratio
    phases = coords.astype(mx.float32)[..., None] * freqs
    phases = phases.reshape(coords.shape[0], -1)
    missing = head_dim // 2 - phases.shape[-1]
    if missing > 0:
        phases = mx.concatenate([phases, mx.zeros((phases.shape[0], missing))], -1)
    return mx.cos(phases), mx.sin(phases)


def apply_rope(x, rope):
    """Rotate interleaved channel pairs of ``x`` ``(B, N, H, D)`` in float32."""
    cos, sin = (t[None, :, None] for t in rope)
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x):
        y = x.astype(mx.float32)
        norm = mx.maximum(mx.linalg.norm(y, axis=-1, keepdims=True), 1e-12)
        return (y / norm * self.gamma * math.sqrt(x.shape[-1])).astype(x.dtype)


def attention(q, k, v):
    """Scaled dot-product attention over ``(B, N, H, D)`` tensors."""
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
    return out.transpose(0, 2, 1, 3)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x, rope):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # RoPE first, then per-head RMS normalization, as the checkpoint was trained.
        q = self.q_rms_norm(apply_rope(q, rope))
        k = self.k_rms_norm(apply_rope(k, rope))
        return self.to_out(attention(q, k, v).reshape(B, N, C))


class CrossAttention(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(ctx_channels, channels * 2)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x, context):
        B, N, C = x.shape
        q = self.to_q(x).reshape(B, N, self.num_heads, -1)
        kv = self.to_kv(context).reshape(B, context.shape[1], 2, self.num_heads, -1)
        out = attention(q, kv[:, :, 0], kv[:, :, 1])
        return self.to_out(out.reshape(B, N, C))


class FeedForward(nn.Module):
    def __init__(self, channels, mlp_ratio):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = [nn.Linear(channels, hidden), nn.GELU(approx="tanh")]
        self.mlp.append(nn.Linear(hidden, channels))

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


def _fold(value, dtype):
    """Round a float32 modulation row to the feature dtype, broadcast over voxels.

    The reference multiplies fp16 features by float32 modulation through a
    sparse broadcast that first copies the modulation into an fp16 buffer.
    """
    return value.astype(dtype)[:, None]


class Block(nn.Module):
    """Self-attention, image cross-attention and MLP under shared modulation."""

    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = LayerNorm32(channels, affine=False)
        self.norm2 = LayerNorm32(channels, affine=True)
        self.norm3 = LayerNorm32(channels, affine=False)
        self.self_attn = SelfAttention(channels, num_heads)
        self.cross_attn = CrossAttention(channels, ctx_channels, num_heads)
        self.mlp = FeedForward(channels, mlp_ratio)
        self.modulation = mx.zeros((1, 6 * channels))

    def __call__(self, x, mod, context, rope):
        dtype = x.dtype
        m = self.modulation.astype(mx.float32) + mod.astype(mx.float32)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            m, 6, axis=-1
        )
        h = self.norm1(x) * _fold(1 + scale_msa, dtype) + _fold(shift_msa, dtype)
        x = x + self.self_attn(h, rope) * _fold(gate_msa, dtype)
        x = x + self.cross_attn(self.norm2(x), context)
        h = self.norm3(x) * _fold(1 + scale_mlp, dtype) + _fold(shift_mlp, dtype)
        return x + self.mlp(h) * _fold(gate_mlp, dtype)


class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SparseConv3d(in_channels, out_channels)

    def __call__(self, x, grid):
        return self.conv(x, grid)


class ResBlock(nn.Module):
    """Timestep-modulated sparse residual block with optional 2x resampling."""

    def __init__(
        self,
        channels,
        emb_channels,
        out_channels=None,
        downsample=False,
        upsample=False,
    ):
        super().__init__()
        out_channels = out_channels or channels
        hidden = out_channels // 2
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, affine=True)
        self.norm2 = LayerNorm32(hidden, affine=False)
        self.conv1 = SparseConv(channels, hidden)
        self.conv2 = SparseConv(hidden, hidden)
        self.linear = nn.Linear(hidden, out_channels)
        self.emb_layers = [nn.SiLU(), nn.Linear(emb_channels, 2 * hidden)]
        if channels != out_channels:
            self.skip_connection = nn.Linear(channels, out_channels)

    def __call__(self, x, grid, emb, fine=None):
        scale, shift = mx.split(self.emb_layers[1](nn.silu(emb)), 2, axis=-1)
        scale, shift = scale.astype(x.dtype)[:, None], shift.astype(x.dtype)[:, None]
        if self.downsample:
            x, grid = downsample_mean(x, grid)
        elif self.upsample:
            x, grid = x[:, fine.downsample()[1]], fine
        h = self.conv1(nn.silu(self.norm1(x)), grid)
        h = self.norm2(h) * (1 + scale) + shift
        h = nn.silu(self.conv2(nn.silu(h), grid))
        skip = self.skip_connection(x) if "skip_connection" in self else x
        return self.linear(h) + skip, grid


class TimestepEmbedder(nn.Module):
    def __init__(self, channels, frequency_size=256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = [nn.Linear(frequency_size, channels), nn.SiLU()]
        self.mlp.append(nn.Linear(channels, channels))

    def __call__(self, t):
        half = self.frequency_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        x = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        for layer in self.mlp:
            x = layer(x)
        return x


class Hi3DTransformer(nn.Module):
    """Predicts the flow velocity of sparse latents under image conditioning."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config.model_channels
        channels = list(config.io_block_channels)
        self.t_embedder = TimestepEmbedder(C)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(C, 6 * C)]
        self.input_layer = nn.Linear(config.in_channels, channels[0])
        self.input_blocks = []
        for current, following in zip(channels, channels[1:] + [C]):
            for _ in range(config.num_io_res_blocks - 1):
                self.input_blocks.append(ResBlock(current, C))
            self.input_blocks.append(ResBlock(current, C, following, downsample=True))
        self.blocks = [
            Block(C, config.cond_channels, config.num_heads, config.mlp_ratio)
            for _ in range(config.num_blocks)
        ]
        self.out_blocks = []
        previous_channels = [C] + list(reversed(channels[1:]))
        for current, previous in zip(reversed(channels), previous_channels):
            self.out_blocks.append(ResBlock(previous * 2, C, current, upsample=True))
            for _ in range(config.num_io_res_blocks - 1):
                self.out_blocks.append(ResBlock(current * 2, C, current))
        self.out_layer = nn.Linear(channels[0], config.out_channels)

    @property
    def compute_dtype(self):
        return self.blocks[0].self_attn.to_qkv.weight.dtype

    def __call__(self, x, t, cond, grid: Grid):
        """Velocity for latents ``x`` ``(B, N, in_channels)`` at times ``t``
        ``(B,)`` (``1000 * sigma``) given conditions ``cond`` ``(B, L, C)``."""
        dtype = self.compute_dtype
        emb = self.t_embedder(t)
        conv_emb = emb.astype(dtype)
        h = self.input_layer(x).astype(dtype)
        skips, fine_grids = [], []
        for block in self.input_blocks:
            if block.downsample:
                fine_grids.append(grid)
            h, grid = block(h, grid, conv_emb)
            skips.append(h)
        mod = self.adaLN_modulation[1](nn.silu(emb)).astype(dtype)
        context = cond.astype(dtype)
        rope = rope_tables(
            grid.coords, self.config.model_channels // self.config.num_heads
        )
        for block in self.blocks:
            h = block(h, mod, context, rope)
        for block, skip in zip(self.out_blocks, reversed(skips)):
            fine = fine_grids.pop() if block.upsample else None
            h, grid = block(mx.concatenate([h, skip], axis=-1), grid, conv_emb, fine)
        h = mx.fast.layer_norm(h, None, None, 1e-5)
        return self.out_layer(h.astype(x.dtype))
