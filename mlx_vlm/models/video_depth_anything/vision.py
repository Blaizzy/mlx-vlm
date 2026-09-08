"""DINOv2 vision backbone for Video Depth Anything (channel-last MLX port)."""

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


def _cubic_weight(t, a=-0.75):
    at = mx.abs(t)
    at2 = at * at
    at3 = at2 * at
    w1 = (a + 2) * at3 - (a + 3) * at2 + 1.0
    w2 = a * at3 - 5 * a * at2 + 8 * a * at - 4 * a
    return mx.where(at <= 1.0, w1, mx.where(at < 2.0, w2, mx.zeros_like(t)))


def bicubic_scale_factor(x, scale_h, scale_w):
    """Match ``F.interpolate(scale_factor=..., mode='bicubic')`` exactly.

    Differs from the generic helper: PyTorch samples with the given
    (fractional) scale, floors the output size, and clamps border taps
    without renormalizing the weights.
    """
    B, C, in_h, in_w = x.shape
    out_h = int(in_h * scale_h)
    out_w = int(in_w * scale_w)

    def axis_weights(in_size, out_size, scale):
        src = (mx.arange(out_size, dtype=mx.float32) + 0.5) / scale - 0.5
        base = mx.floor(src)
        taps = base[:, None] + mx.arange(-1, 3, dtype=mx.float32)[None, :]
        w = _cubic_weight(src[:, None] - taps)
        idx = mx.clip(taps.astype(mx.int32), 0, in_size - 1)
        return idx, w

    iy, wy = axis_weights(in_h, out_h, scale_h)
    ix, wx = axis_weights(in_w, out_w, scale_w)

    g = x[:, :, iy.reshape(-1), :].reshape(B, C, out_h, 4, in_w)
    t = mx.sum(g * wy[None, None, :, :, None], axis=3)
    g = t[:, :, :, ix.reshape(-1)].reshape(B, C, out_h, out_w, 4)
    return mx.sum(g * wx[None, None, None, :, :], axis=4)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B, H, W, C) -> (B, N, D)."""

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x)  # (B, H/14, W/14, D)
        return x.reshape(B, -1, x.shape[-1])


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.embed_dim
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.attn = Attention(dim, config.num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = Mlp(dim, int(dim * config.mlp_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DINOv2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.patch_size = config.patch_size
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, 3, config.embed_dim
        )
        self.cls_token = mx.zeros((1, 1, config.embed_dim))
        self.pos_embed = mx.zeros(
            (1, self.patch_embed.num_patches + 1, config.embed_dim)
        )
        self.mask_token = mx.zeros((1, config.embed_dim))
        self.blocks = [Block(config) for _ in range(config.depth)]
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def interpolate_pos_encoding(self, x: mx.array, h: int, w: int) -> mx.array:
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        dim = x.shape[-1]
        class_pos_embed = self.pos_embed[:, :1]
        patch_pos_embed = self.pos_embed[:, 1:]
        # Small offset to avoid floating point error in the interpolation
        w0 = w // self.patch_size + self.config.interpolate_offset
        h0 = h // self.patch_size + self.config.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        # (B, C, H, W) for the interpolation; (sy, sx) because the reference
        # derives w0 from the pixel height and applies it to the W axis
        patch_pos_embed = patch_pos_embed.reshape(
            1, int(sqrt_N), int(sqrt_N), dim
        ).transpose(0, 3, 1, 2)
        patch_pos_embed = bicubic_scale_factor(
            patch_pos_embed.astype(mx.float32), sy, sx
        )
        patch_pos_embed = patch_pos_embed.transpose(0, 2, 3, 1).reshape(1, -1, dim)
        return mx.concatenate([class_pos_embed, patch_pos_embed], axis=1).astype(
            x.dtype
        )

    def prepare_tokens(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape
        x = self.patch_embed(x)
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.embed_dim))
        x = mx.concatenate([cls, x], axis=1)
        return x + self.interpolate_pos_encoding(x, H, W)

    def get_intermediate_layers(
        self, x: mx.array, indices: List[int]
    ) -> List[Tuple[mx.array, mx.array]]:
        """Run the backbone and return (patch_tokens, cls_token) per index."""
        x = self.prepare_tokens(x)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in indices:
                out = self.norm(x)
                outputs.append((out[:, 1:], out[:, 0]))
        assert len(outputs) == len(indices)
        return outputs
