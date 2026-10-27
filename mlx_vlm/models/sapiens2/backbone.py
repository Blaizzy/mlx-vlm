"""Sapiens2 vision backbone: channel-last images (B, H, W, C) in, token
sequence (B, prefix + h*w, D) and patch grid (h, w) out.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig


class LayerScale(nn.Module):
    """Per-channel scale loaded from the checkpoint."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.weight


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dims: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=patch_size, stride=patch_size
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        x = self.projection(x)  # (B, h, w, D)
        B, h, w, _ = x.shape
        return x.reshape(B, h * w, -1), (h, w)


class RopePositionEmbedding(nn.Module):
    """2D RoPE; frequencies stored as ``periods``.

    Tables are returned packed for ``_rope_apply``: ``sin`` (N, 2, D/2) as
    ``[-sin, sin]`` and ``cos`` (N, 1, D/2), with ``prefix`` identity rows
    so cls/register tokens are not rotated. Cached per grid.
    """

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        self.D_head = embed_dim // num_heads
        periods = base ** (
            2 * mx.arange(self.D_head // 4, dtype=mx.float32) / (self.D_head // 2)
        )
        self.periods = periods
        self._tables = {}

    def __call__(self, H: int, W: int, prefix: int = 0) -> Tuple[mx.array, mx.array]:
        key = (H, W, prefix)
        if key not in self._tables:
            # Patch-center coordinates in [-1, +1], normalized per axis.
            coords_h = mx.arange(0.5, H, dtype=mx.float32) / H
            coords_w = mx.arange(0.5, W, dtype=mx.float32) / W
            coords = mx.stack(mx.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)
            coords = 2.0 * coords.reshape(-1, 2) - 1.0

            periods = self.periods.astype(mx.float32)
            angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
            angles = angles.reshape(angles.shape[0], -1)  # [HW, D_head / 2]
            sin, cos = mx.sin(angles), mx.cos(angles)
            if prefix > 0:
                half = self.D_head // 2
                sin = mx.concatenate([mx.zeros((prefix, half)), sin])
                cos = mx.concatenate([mx.ones((prefix, half)), cos])
            self._tables[key] = (mx.stack([-sin, sin], axis=1), cos[:, None, :])
        return self._tables[key]


@mx.compile
def _rope_apply(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Rotate ``x`` (B, heads, N, D) by the packed tables, in float32; the
    reversed pair-axis view supplies (x2, x1)."""
    B, H, N, D = x.shape
    xf = x.astype(mx.float32).reshape(B, H, N, 2, D // 2)
    out = xf * cos + xf[..., ::-1, :] * sin
    return out.reshape(B, H, N, D).astype(x.dtype)


class GroupedQueryAttention(nn.Module):
    """MHSA / GQA with QK-RMSNorm and LayerScale.

    The checkpoint's q/k/v projections are merged into one ``wqkv`` matmul
    by ``Model.sanitize``; the attention kernel handles the KV grouping.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        use_qk_norm: bool = True,
        use_layer_scale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim**-0.5

        self.kv_size = self.num_kv_heads * self.head_dim
        self.wqkv = nn.Linear(embed_dims, embed_dims + 2 * self.kv_size, bias=qkv_bias)

        self.q_norm = (
            nn.RMSNorm(self.head_dim, eps=eps) if use_qk_norm else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, eps=eps) if use_qk_norm else nn.Identity()
        )

        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.gamma = LayerScale(embed_dims) if use_layer_scale else nn.Identity()

    def __call__(self, x: mx.array, rope=None) -> mx.array:
        B, N, _ = x.shape
        q, k, v = mx.split(
            self.wqkv(x), [self.embed_dims, self.embed_dims + self.kv_size], axis=-1
        )
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            sin, cos = rope
            q, k = _rope_apply(q, sin, cos), _rope_apply(k, sin, cos)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.embed_dims)
        return self.gamma(self.proj(out))


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dims: int, feedforward_channels: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(embed_dims, 2 * feedforward_channels, bias=bias)
        self.w3 = nn.Linear(feedforward_channels, embed_dims, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(self.w12(x), 2, axis=-1)
        return self.w3(nn.silu(x1) * x2)


class TransformerEncoderLayer2(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        feedforward_channels: Optional[int] = None,
        qkv_bias: bool = True,
        mlp_bias: bool = True,
        use_qk_norm: bool = True,
        use_layer_scale: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(embed_dims, eps=eps)
        self.attn = GroupedQueryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            use_qk_norm=use_qk_norm,
            use_layer_scale=use_layer_scale,
            eps=eps,
        )
        self.ln2 = nn.RMSNorm(embed_dims, eps=eps)
        self.ffn = SwiGLUFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels or embed_dims,
            bias=mlp_bias,
        )

    def __call__(self, x: mx.array, rope=None) -> mx.array:
        x = x + self.attn(self.ln1(x), rope=rope)
        x = x + self.ffn(self.ln2(x))
        return x


class Tokenizer(nn.Module):
    """Window self-attention tokenizer of the 4K variant."""

    def __init__(
        self,
        embed_dims: int,
        window_size: int = 4,
        num_heads: int = 4,
        num_tokenizer_layers: int = 1,
        qkv_bias: bool = True,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.ws = window_size
        self.chunk_size = chunk_size
        self.local_pos_embed = mx.zeros((1, 1 + window_size * window_size, embed_dims))
        self.blocks = [
            TransformerEncoderLayer2(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=embed_dims * 4,
                qkv_bias=qkv_bias,
                use_qk_norm=False,
                use_layer_scale=False,
            )
            for _ in range(num_tokenizer_layers)
        ]
        self.w_cls = mx.zeros((1, 1, embed_dims))

    def __call__(self, x: mx.array, hw: Tuple[int, int]):
        B, N, C = x.shape
        H, W = hw
        ws = self.ws
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"Grid {H}x{W} not divisible by window {ws}")
        ph, pw = H // ws, W // ws

        # Tokens -> non-overlapping windows: (B*ph*pw, ws*ws, C)
        x = x.reshape(B, ph, ws, pw, ws, C).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(B * ph * pw, ws * ws, C)

        total = x.shape[0]
        outs = []
        for i in range(0, total, self.chunk_size):
            chunk = x[i : i + self.chunk_size]
            m = chunk.shape[0]
            cls = mx.broadcast_to(self.w_cls, (m, 1, C))
            chunk = mx.concatenate([cls, chunk], axis=1)
            chunk = chunk + self.local_pos_embed
            for blk in self.blocks:
                chunk = blk(chunk)
            outs.append(chunk[:, 0])
        return mx.concatenate(outs, axis=0).reshape(B, ph * pw, C), (ph, pw)


class Sapiens2Backbone(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_dims = config.hidden_size
        self.patch_size = config.patch_size
        self.num_prefix_tokens = 1 + config.num_register_tokens

        self.patch_embed = PatchEmbed(
            in_channels=config.num_channels,
            embed_dims=config.hidden_size,
            patch_size=config.patch_size,
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            base=config.rope_theta,
        )

        if config.use_tokenizer:
            self.tokenizer = Tokenizer(
                embed_dims=config.hidden_size,
                window_size=config.tokenizer_window_size,
                num_heads=config.num_attention_heads,
                num_tokenizer_layers=config.num_tokenizer_layers,
            )
        else:
            self.tokenizer = None

        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.storage_tokens = (
            mx.zeros((1, config.num_register_tokens, config.hidden_size))
            if config.num_register_tokens > 0
            else None
        )

        kv_heads = config.kv_heads_per_layer
        self.blocks = [
            TransformerEncoderLayer2(
                embed_dims=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=kv_heads[i],
                feedforward_channels=config.intermediate_size,
                qkv_bias=config.query_bias and config.key_bias and config.value_bias,
                mlp_bias=config.mlp_bias,
                use_qk_norm=config.use_qk_norm,
                eps=config.rms_norm_eps,
            )
            for i in range(config.num_hidden_layers)
        ]

        self.ln1 = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.normalize_backbone_outputs
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        """The input is cast to the parameter dtype, so float32 pixel
        values run fully in bf16 on a bf16 model."""
        B = x.shape[0]
        x = x.astype(self.patch_embed.projection.weight.dtype)
        x, hw = self.patch_embed(x)
        if self.tokenizer is not None:
            x, hw = self.tokenizer(x, hw)

        prepend = [mx.broadcast_to(self.cls_token, (B, 1, self.embed_dims))]
        if self.storage_tokens is not None:
            prepend.append(
                mx.broadcast_to(
                    self.storage_tokens,
                    (B, self.storage_tokens.shape[1], self.embed_dims),
                )
            )
        x = mx.concatenate(prepend + [x], axis=1)

        rope = self.rope_embed(hw[0], hw[1], prefix=self.num_prefix_tokens)
        for block in self.blocks:
            x = block(x, rope=rope)
        x = self.ln1(x)
        return x, hw

    def feature_map(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) -> patch feature map (B, h, w, D), prefix removed."""
        tokens, (h, w) = self(x)
        patch = tokens[:, self.num_prefix_tokens :, :]
        return patch.reshape(x.shape[0], h, w, self.embed_dims)
