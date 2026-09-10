from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """2D RoPE tables, one row per patch."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    hpos = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    wpos = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    freqs = (
        mx.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(mx.float32) * inv_freq
    )
    freqs = freqs.reshape(n_h * n_w, -1)
    return (
        mx.cos(freqs)[:, None, :],
        mx.sin(freqs)[:, None, :],
    )


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotary embedding over the last dim, split in halves."""
    dtype = x.dtype
    x = x.astype(mx.float32)
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(
        dtype
    )


class PatchEmbed(nn.Module):
    """Flatten each patch and project to the vision dim."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(
            3 * config.vision_patch_size**2, config.vision_hidden_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x.reshape(x.shape[0], -1))


class VisionAttention(nn.Module):
    """Full bidirectional self-attention with 2D RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.vision_hidden_size
        self.n_heads = config.vision_num_heads
        self.head_dim = dim // config.vision_num_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        q, k, v = (
            t.reshape(n, self.n_heads, self.head_dim)
            for t in mx.split(self.wqkv(x), 3, axis=-1)
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        scores = mx.matmul(
            q.transpose(1, 0, 2), k.transpose(1, 0, 2).transpose(0, 2, 1)
        )
        o = mx.matmul(mx.softmax(scores * self.scale, axis=-1), v.transpose(1, 0, 2))
        return self.wo(o.transpose(1, 0, 2).reshape(n, -1))


class VisionMLP(nn.Module):
    """SwiGLU feed-forward over the vision dim."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(
            config.vision_hidden_size, 2 * config.vision_intermediate_size, bias=False
        )
        self.w2 = nn.Linear(
            config.vision_intermediate_size, config.vision_hidden_size, bias=False
        )
        self.act = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(self.act(gate) * up)


class VisionBlock(nn.Module):
    """Pre-norm attention + MLP residual block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.vision_hidden_size)
        self.attn = VisionAttention(config)
        self.norm2 = nn.RMSNorm(config.vision_hidden_size)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rope_dim = config.vision_hidden_size // config.vision_num_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(config)
        self.blocks = [VisionBlock(config) for _ in range(config.vision_num_layers)]
        self.norm = nn.RMSNorm(config.vision_hidden_size)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """Space-to-depth downsample of the ViT grid into language-dim embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        in_dim = config.vision_hidden_size * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.GELU(approx="precise")

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        r = self.downsample_ratio
        h2 = n_h - n_h % r
        w2 = n_w - n_w % r
        x = x.reshape(n_h, n_w, -1).transpose(2, 0, 1)[:, :h2, :w2]
        d = x.shape[0]
        x = (
            x.reshape(d, h2 // r, r, w2 // r, r)
            .transpose(1, 3, 2, 4, 0)
            .reshape(-1, d * r * r)
        )
        return self.w2(self.act(self.w1(x)))
