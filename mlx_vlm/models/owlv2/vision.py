"""OWLv2 Vision Transformer encoder."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def quick_gelu(x):
    return x * mx.sigmoid(1.702 * x)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.class_embedding = mx.zeros((config.hidden_size,))
        num_positions = config.num_patches + 1
        self.position_embedding = nn.Embedding(num_positions, config.hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        # pixel_values: (B, H, W, C) in MLX convention
        patch_embeds = self.patch_embedding(pixel_values)  # (B, h, w, D)
        B = patch_embeds.shape[0]
        patch_embeds = patch_embeds.reshape(B, -1, patch_embeds.shape[-1])  # (B, N, D)

        cls_embeds = mx.broadcast_to(
            self.class_embedding, (B, 1, patch_embeds.shape[-1])
        )
        embeddings = mx.concatenate([cls_embeds, patch_embeds], axis=1)

        position_ids = mx.arange(embeddings.shape[1])[None]
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(quick_gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask=mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        x = self.embeddings(pixel_values)
        x = self.pre_layernorm(x)
        x = self.encoder(x)
        x = self.post_layernorm(x)
        return x


class VisionModel(nn.Module):
    """Wrapper for framework compatibility."""

    def __init__(self, config=None):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return None
