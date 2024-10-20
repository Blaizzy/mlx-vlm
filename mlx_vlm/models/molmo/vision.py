import inspect
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 24
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "gelu"
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    d_model: int = 768

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class ViTMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.image_emb_dim, config.image_mlp_dim)
        self.w2 = nn.Linear(config.image_mlp_dim, config.image_emb_dim)
        self.act = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.image_emb_dim
        self.num_heads = config.image_num_heads
        self.head_dim = config.image_head_dim
        self.num_key_value_heads = config.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.wq = nn.Linear(self.embed_dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            self.embed_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.wv = nn.Linear(
            self.embed_dim, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape

        q = (
            self.wq(x)
            .reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.wk(x)
            .reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.wv(x)
            .reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        if self.num_key_value_heads != self.num_heads:
            k = k.repeat(self.num_key_value_groups, axis=1)
            v = v.repeat(self.num_key_value_groups, axis=1)

        attn = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attn = mx.softmax(attn, axis=-1)

        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        out = self.wo(out)
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadDotProductAttention(config)
        self.mlp = ViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim, eps=config.image_norm_eps
        )
        self.ffn_norm = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.class_embedding = mx.zeros((1, config.image_emb_dim))
        self.positional_embedding = mx.zeros(
            (config.image_num_pos, config.image_emb_dim)
        )
        self.patch_embedding = nn.Linear(
            config.image_patch_size * config.image_patch_size * 3,
            config.image_emb_dim,
            bias=False,
        )
        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)
        self.transformer = nn.Sequential(
            *[ResidualAttentionBlock(config) for _ in range(config.image_num_layers)]
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, num_patch, _ = x.shape
        x = self.patch_embedding(x)
        cls_embedding = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, self.config.image_emb_dim)
        )
        x = mx.concatenate([cls_embedding, x], axis=1)
        x = x + self.positional_embedding[: x.shape[1]]
        x = self.pre_ln(x)
        return self.transformer(x)


class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_vit = VisionTransformer(config)

        self.image_pooling_2d = MultiHeadDotProductAttention(config)
        self.image_projector = nn.Sequential(
            nn.Linear(config.image_emb_dim, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, config.d_model),
        )

    def __call__(
        self, images: mx.array, image_masks: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        batch_size, num_image, num_patch, _ = images.shape
        image_features = self.image_vit(
            images.reshape(batch_size * num_image, num_patch, -1)
        )

        cls_embed = image_features[:, 0]
        image_features = image_features[:, 1:]

        image_features = image_features.reshape(batch_size, num_image, num_patch, -1)
        cls_embed = cls_embed.reshape(batch_size, num_image, -1)

        h, w = self.config.vision_backbone.image_num_patch
        image_features = image_features.reshape(batch_size, num_image, h, w, -1)

        image_features = image_features.reshape(
            batch_size * num_image * h * w // 4, 4, -1
        )
        image_features = self.image_pooling_2d(image_features)
        image_features = image_features.reshape(batch_size, num_image, h * w // 4, -1)

        image_features = self.image_projector(image_features)

        return image_features, cls_embed

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embed.proj.weight" in k:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 4, 1)
            else:
                sanitized_weights[k] = v

        return sanitized_weights
