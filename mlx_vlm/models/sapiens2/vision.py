"""Sapiens2 ViT backbone (patch embed, 2D RoPE, GQA + SwiGLU blocks, final RMSNorm).

Ported from sapiens/backbones/sapiens2.py. The backbone shared across pose / seg /
normal / pointmap heads is driven entirely by BackboneConfig (arch, image size,
patch size, rope params). Output is always a featmap — (B, H, W, C) channel-last —
since every downstream head consumes a 2-D feature grid.
"""

import math
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import BackboneConfig


class RopePositionEmbedding(nn.Module):
    """2-D axial rotary position embedding with "separate" coordinate normalization.

    The PyTorch reference (`sapiens.backbones.sapiens2.RopePositionEmbedding`) stores
    a `periods` buffer of length `D_head//4` as **bfloat16** (its default
    `pos_embed_rope_dtype="bf16"`) and computes sin/cos in the same dtype.  We
    match that: periods arrive from the checkpoint at bf16, and the downstream
    `apply_rope` in `GroupedQueryAttention` casts Q/K to bf16 for the rotation
    and casts them back.  This is numerically load-bearing — running rope in
    fp32 produces a ~3e-2 sin/cos drift versus PT.
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0,
                 normalize_coords: str = "separate"):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.normalize_coords = normalize_coords
        self.base = base
        # Initial values are overwritten from the checkpoint; kept for rare cases
        # where a user instantiates the model without loading weights.
        exponents = 2 * mx.arange(self.head_dim // 4) / (self.head_dim // 2)
        self.periods = mx.power(mx.array(base, dtype=mx.float32), exponents).astype(mx.bfloat16)

    def __call__(self, H: int, W: int):
        if self.normalize_coords == "max":
            denom_h = denom_w = max(H, W)
        elif self.normalize_coords == "min":
            denom_h = denom_w = min(H, W)
        else:  # "separate"
            denom_h, denom_w = H, W

        dtype = self.periods.dtype  # bf16 from checkpoint
        coords_h = ((mx.arange(H, dtype=mx.float32) + 0.5) / denom_h).astype(dtype)
        coords_w = ((mx.arange(W, dtype=mx.float32) + 0.5) / denom_w).astype(dtype)
        grid_h = mx.broadcast_to(coords_h[:, None], (H, W))
        grid_w = mx.broadcast_to(coords_w[None, :], (H, W))
        coords = mx.stack([grid_h, grid_w], axis=-1)  # (H, W, 2)
        coords = coords.reshape(H * W, 2)
        coords = 2.0 * coords - 1.0  # [-1, +1]

        two_pi = mx.array(2 * math.pi, dtype=dtype)
        angles = two_pi * coords[:, :, None] / self.periods[None, None, :]  # (HW, 2, D//4)
        angles = angles.reshape(H * W, -1)  # (HW, D//2)
        angles = mx.concatenate([angles, angles], axis=-1)  # (HW, D)
        return mx.sin(angles), mx.cos(angles)


def _rotate_half(x: mx.array) -> mx.array:
    # [..., D] → split last dim in half, return [-x2, x1]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _rope_apply(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    return (x * cos) + (_rotate_half(x) * sin)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.weight = mx.full((dim,), init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.weight


class GroupedQueryAttention(nn.Module):
    """Multi-head QKV with optional per-layer KV-head reduction.

    The backbone uses full MHA for the first `mhsa_early` and last `mhsa_late`
    layers (num_kv_heads == num_heads) and halves KV heads for the middle layers,
    so K / V projections vary in output width across blocks — the checkpoint's
    wk.weight / wv.weight shapes drive the module config.
    """

    def __init__(self, embed_dims: int, num_heads: int,
                 num_kv_heads: Optional[int] = None,
                 layer_scale_init_value: float = 0.0):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(embed_dims, embed_dims, bias=True)
        self.wk = nn.Linear(embed_dims, self.num_kv_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(embed_dims, self.num_kv_heads * self.head_dim, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=True)
        self.gamma = LayerScale(embed_dims, layer_scale_init_value) \
            if layer_scale_init_value > 0 else None

    def _apply_rope(self, q: mx.array, k: mx.array, sin: mx.array, cos: mx.array):
        # rope applies only to the trailing `sin.shape[-2]` tokens; cls + storage
        # tokens at the front are passed through.  PT casts Q/K to sin.dtype
        # (bf16) for the rotation and back to Q/K's original dtype afterwards.
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        rope_dtype = sin.dtype
        q_dtype, k_dtype = q.dtype, k.dtype

        q_body = q[:, :, prefix:, :].astype(rope_dtype)
        k_body = k[:, :, prefix:, :].astype(rope_dtype)
        q_rot = _rope_apply(q_body, sin, cos).astype(q_dtype)
        k_rot = _rope_apply(k_body, sin, cos).astype(k_dtype)
        q = mx.concatenate([q[:, :, :prefix, :], q_rot], axis=-2)
        k = mx.concatenate([k[:, :, :prefix, :], k_rot], axis=-2)
        return q, k

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        B, N, _ = x.shape
        q = self.wq(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, factor, axis=1)
            v = mx.repeat(v, factor, axis=1)

        q, k = self._apply_rope(q, k, sin, cos)

        # MLX fused SDPA supports head dim 64 (Sapiens2's head_dim for 0.4b); it
        # also handles larger dims for bigger archs.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.embed_dims)
        out = self.proj(out)
        if self.gamma is not None:
            out = self.gamma(out)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dims: int, feedforward_channels: int):
        super().__init__()
        self.w12 = nn.Linear(embed_dims, 2 * feedforward_channels, bias=True)
        self.w3 = nn.Linear(feedforward_channels, embed_dims, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x12 = self.w12(x)
        half = x12.shape[-1] // 2
        x1, x2 = x12[..., :half], x12[..., half:]
        return self.w3(nn.silu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dims: int, num_heads: int, num_kv_heads: Optional[int],
                 feedforward_channels: int, layer_scale_init_value: float):
        super().__init__()
        self.ln1 = nn.RMSNorm(embed_dims, eps=1e-6)
        self.attn = GroupedQueryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.ln2 = nn.RMSNorm(embed_dims, eps=1e-6)
        self.ffn = SwiGLUFFN(embed_dims, feedforward_channels)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x), sin, cos)
        x = x + self.ffn(self.ln2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dims: int, patch_size: int):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array):
        # x: (B, H, W, 3) → (B, H', W', C)
        y = self.projection(x)
        B, H, W, C = y.shape
        return y.reshape(B, H * W, C), (H, W)


class Sapiens2Backbone(nn.Module):
    """Top-level ViT backbone — always emits (B, H, W, C) featmap output."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.embed_dims = config.embed_dims

        self.patch_embed = PatchEmbed(3, config.embed_dims, config.patch_size)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=config.embed_dims,
            num_heads=config.num_heads,
            base=config.rope_base,
            normalize_coords=config.rope_normalize_coords,
        )
        self.cls_token = mx.zeros((1, 1, config.embed_dims))
        self.storage_tokens = mx.zeros((1, config.n_storage_tokens, config.embed_dims))

        blocks = []
        for i in range(config.num_layers):
            if i < config.mhsa_early or i >= config.num_layers - config.mhsa_late:
                num_kv_heads = None  # full MHA
            else:
                num_kv_heads = config.num_heads // 2  # GQA
            blocks.append(TransformerBlock(
                embed_dims=config.embed_dims,
                num_heads=config.num_heads,
                num_kv_heads=num_kv_heads,
                feedforward_channels=config.feedforward_channels,
                layer_scale_init_value=config.layer_scale_init_value,
            ))
        self.blocks = blocks
        self.ln1 = nn.RMSNorm(config.embed_dims, eps=1e-6) if config.final_norm else None

    @property
    def num_extra_tokens(self) -> int:
        return 1 + int(self.config.n_storage_tokens)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        """Args: pixel_values (B, H, W, 3).  Returns featmap (B, H', W', C)."""
        B = pixel_values.shape[0]
        x, (h, w) = self.patch_embed(pixel_values)

        cls = mx.broadcast_to(self.cls_token, (B,) + self.cls_token.shape[1:])
        storage = mx.broadcast_to(
            self.storage_tokens, (B,) + self.storage_tokens.shape[1:]
        )
        x = mx.concatenate([cls, storage, x], axis=1)

        sin, cos = self.rope_embed(h, w)
        for blk in self.blocks:
            x = blk(x, sin, cos)
        if self.ln1 is not None:
            x = self.ln1(x)

        patch_tokens = x[:, self.num_extra_tokens:, :]  # (B, h*w, C)
        return patch_tokens.reshape(B, h, w, self.embed_dims)


class VisionModel(nn.Module):
    """Thin wrapper around Sapiens2Backbone for framework symmetry."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.backbone = Sapiens2Backbone(config)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.backbone(pixel_values)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return weights
