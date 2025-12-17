"""Vision encoder for Jina VLM in MLX."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


class PatchEmbedding(nn.Module):
    """Patch embedding using linear projection."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        # Linear projection for patches - named to match weights
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(patch_dim, config.hidden_size, bias=config.use_bias)

    def __call__(self, x: mx.array) -> Tuple[mx.array, Tuple[int, int]]:
        if x.ndim == 3:
            # Already patchified: (B, n_patches, patch_dim)
            B, n_patches, _ = x.shape
            nH = nW = int(n_patches**0.5)
            x = self.proj(x)
        else:
            # Image format: (B, C, H, W)
            B, C, H, W = x.shape
            pH, pW = self.patch_size, self.patch_size
            nH, nW = H // pH, W // pW
            x = x.reshape(B, C, nH, pH, nW, pW)
            x = x.transpose(0, 2, 4, 1, 3, 5)
            x = x.reshape(B, nH * nW, C * pH * pW)
            x = self.proj(x)
        return x, (nH, nW)


class VisionMLP(nn.Module):
    """MLP for vision transformer - matches weight naming: ffn.up, ffn.down"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        # Named to match weights: ffn.up, ffn.down
        self.up = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.use_bias
        )
        self.down = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )
        # Use built-in GELU with tanh approximation
        if config.activation == "gelu_pytorch_tanh":
            self.gelu = nn.GELU(approx="tanh")
        else:
            self.gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.up(x)
        x = self.gelu(x)
        x = self.down(x)
        return x


class VisionAttention(nn.Module):
    """Multi-head self-attention - matches weight naming: attn.qkv, attn.out"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # Fused QKV projection - named to match weights
        self.qkv = nn.Linear(
            config.hidden_size,
            3 * config.num_attention_heads * config.head_dim,
            bias=config.use_bias,
        )
        self.out = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = attn @ v

        x = x.transpose(0, 2, 1, 3).reshape(B, L, -1)
        x = self.out(x)
        return x


class VisionEncoderLayer(nn.Module):
    """Transformer block - matches weight naming: attn_norm, ffn_norm"""

    def __init__(self, config: VisionConfig):
        super().__init__()
        # Named to match weights: attn_norm, ffn_norm
        self.attn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.use_bias
        )
        self.attn = VisionAttention(config)
        self.ffn_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.use_bias
        )
        self.ffn = VisionMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VisionModel(nn.Module):
    """Vision encoder (SigLIP-style ViT)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.hidden_size = config.hidden_size
        self.vit_layers = config.vit_layers

        # Named to match weights: patch_embed.proj
        self.patch_embed = PatchEmbedding(config)

        # Named to match weights: pos_embed (saved as 2D, not 3D)
        num_patches = (config.image_size // config.patch_size) ** 2
        if config.use_cls_token:
            num_patches += 1
            self.cls_token = mx.zeros((1, 1, config.hidden_size))
        else:
            self.cls_token = None
        self.pos_embed = mx.zeros((num_patches, config.hidden_size))

        # Transformer blocks
        self.layers = [
            VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Named to match weights: post_norm
        if config.post_layer_norm:
            self.post_norm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps, bias=config.use_bias
            )
        else:
            self.post_norm = None

    def __call__(self, x: mx.array) -> Tuple[mx.array, List[mx.array]]:
        x, shape = self.patch_embed(x)

        if self.cls_token is not None:
            B = x.shape[0]
            cls = mx.broadcast_to(self.cls_token, (B, 1, self.hidden_size))
            x = mx.concatenate([cls, x], axis=1)

        # pos_embed is (num_patches, hidden_size), add batch dim for broadcast
        x = x + self.pos_embed[None, :, :]

        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)

        if self.post_norm is not None:
            x = self.post_norm(x)
            hidden_states.append(x)

        return x, hidden_states

    def get_features(self, images: mx.array) -> mx.array:
        """Extract features from specific ViT layers.

        Note: hidden_states includes all layer outputs plus the post_norm output.
        vit_layers indices (e.g., [-4, -10]) are applied to this full list.
        For 27 layers with post_norm, hidden_states has 28 elements:
        - indices 0-26: layer 0-26 outputs
        - index 27: post_norm output
        So vit_layers=[-4, -10] extracts layers 24 and 18 (not 23 and 17).
        """
        _, hidden_states = self(images)
        # Use full hidden_states including post_norm output for correct indexing

        features = []
        for layer_idx in self.vit_layers:
            feats = hidden_states[layer_idx]
            if self.cls_token is not None:
                feats = feats[:, 1:]
            features.append(feats)

        return mx.concatenate(features, axis=-1)
