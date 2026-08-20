"""DINOv3-H+ ViT backbone for SAM 3D Body, implemented in MLX."""

import mlx.core as mx
import mlx.nn as nn

from .config import SAM3DConfig
from .layers import LayerScale, SwiGLU
from .rope import DINOv3RoPE, apply_rope


class Attention(nn.Module):
    """Multi-head attention with masked K bias and RoPE.

    DINOv3 mask_k_bias=True: K projection bias is zeroed out.
    Q and V biases are used normally.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # K bias masked to 0
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(
        self, x: mx.array, rope: tuple[mx.array, mx.array] | None = None
    ) -> mx.array:
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if rope is not None:
            sin, cos = rope
            prefix = N - sin.shape[0]
            q, k = apply_rope(q, k, sin, cos, prefix)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with LayerScale."""

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attention = Attention(embed_dim, num_heads, head_dim)
        self.ls1 = LayerScale(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp = SwiGLU(embed_dim, hidden_dim)
        self.ls2 = LayerScale(embed_dim)

    def __call__(
        self, x: mx.array, rope: tuple[mx.array, mx.array] | None = None
    ) -> mx.array:
        x = x + self.ls1(self.attention(self.norm1(x), rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Convolutional patch embedding (NHWC)."""

    def __init__(self, patch_size: int, embed_dim: int, in_channels: int = 3):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(x)


class DINOv3Backbone(nn.Module):
    """DINOv3-H+ ViT backbone.

    Input: (B, H, W, 3) NHWC image
    Output: (B, H_patches, W_patches, embed_dim) patch features
    """

    def __init__(self, config: SAM3DConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim
        hidden_dim = int(embed_dim * config.ffn_ratio)
        self.patch_size = config.patch_size

        self.cls_token = mx.zeros((1, 1, embed_dim))
        self.storage_tokens = mx.zeros((1, config.num_storage_tokens, embed_dim))

        self.patch_embed = PatchEmbed(config.patch_size, embed_dim)
        self.rope_embed = DINOv3RoPE(config.head_dim)

        self.blocks = [
            TransformerBlock(embed_dim, config.num_heads, config.head_dim, hidden_dim)
            for _ in range(config.depth)
        ]
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: (B, H, W, 3) NHWC image

        Returns:
            (B, H_patches, W_patches, embed_dim) patch features
        """
        B, H, W, _ = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size

        # patch embed -> flatten spatial dims
        x = self.patch_embed(x)  # (B, H_p, W_p, C)
        x = x.reshape(B, H_p * W_p, self.config.embed_dim)

        # prepend CLS + storage tokens
        cls = mx.broadcast_to(self.cls_token, (B, 1, self.config.embed_dim))
        stor = mx.broadcast_to(
            self.storage_tokens,
            (B, self.config.num_storage_tokens, self.config.embed_dim),
        )
        x = mx.concatenate([cls, stor, x], axis=1)

        # compute RoPE for patch grid
        rope = self.rope_embed(H_p, W_p)

        # transformer blocks
        for block in self.blocks:
            x = block(x, rope)

        # final norm, return spatial patch tokens
        x = self.norm(x)
        prefix = 1 + self.config.num_storage_tokens
        patch_tokens = x[:, prefix:]
        return patch_tokens.reshape(B, H_p, W_p, self.config.embed_dim)
