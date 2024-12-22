import copy
import inspect
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str
    layers: int = 27
    width: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    mlp_ratio: float = 3.7362
    cls: str = None
    params: dict = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class MLPConfig:
    width: int
    intermediate_size: int


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class AttentionPoolLatent(nn.Module):
    """Attention pooling w/ latent query"""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[nn.Module] = None,
        drop: float = 0.0,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pool = pool_type

        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = mx.zeros((self.latent_len, embed_dim))[None, :]

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

        if pos_embed == "abs":
            spatial_len = self.feat_size
            self.pos_embed = mx.zeros((spatial_len, in_features))
        else:
            self.pos_embed = None

        self.norm = nn.LayerNorm(out_features)
        config = MLPConfig(
            width=embed_dim, intermediate_size=int(embed_dim * mlp_ratio)
        )
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        B, N, C = x.shape

        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = mx.array(self.latent)

        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .transpose(2, 0, 3, 1, 4)
        )
        k, v = mx.split(kv, 2, axis=0)

        q, k = self.q_norm(q), self.k_norm(k)

        x = mx.fast.scaled_dot_product_attention(
            q, k[0], v[0], scale=(1.0 / sqrt(q.shape[-1])), mask=None
        )

        x = x.transpose(0, 2, 1, 3).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        self.num_heads = num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dims, dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x, mask=None):
        qkv = self.qkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, S, num_heads, -1).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, config: Union[VisionConfig, Dict], bias: bool = True):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.width, config.intermediate_size, bias=bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.width, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.width
        self.attn = Attention(config.width, config.num_attention_heads, qkv_bias=True)
        self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.norm1(x)
        y = self.attn(y, mask)
        x = x + y
        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig, norm_layer: bool = False):
        super().__init__()
        self.config = config
        self.embed_dim = config.width
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.norm = (
            nn.LayerNorm(config.width, eps=config.layer_norm_eps)
            if norm_layer
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        patch_embeddings = self.proj(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        return self.norm(patch_embeddings)


class SigLipVisionModel(nn.Module):
    def __init__(
        self,
        config: VisionConfig,
        ignore_head: bool,
        pre_norm: bool = False,
        no_embed_class: bool = True,
    ):
        super().__init__()
        self.num_prefix_tokens = 1
        self.no_embed_class = False
        self.dynamic_img_size = False
        self.ignore_head = ignore_head
        self.cls_token = None
        self.reg_token = None
        self.patch_embed = VisionEmbeddings(config)
        self.norm_pre = nn.LayerNorm(config.width) if pre_norm else nn.Identity()
        self.blocks = [EncoderLayer(config) for _ in range(config.layers)]
        self.norm = nn.LayerNorm(config.width)
        num_patches = self.patch_embed.num_patches
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = mx.random.normal((embed_len, config.width))[None, :]

        norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.attn_pool = AttentionPoolLatent(
            config.width,
            num_heads=config.num_attention_heads,
            norm_layer=norm_layer,
            mlp_ratio=config.mlp_ratio,
        )

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.patch_embed(x)
        x += self.pos_embed
        x = self.norm_pre(x)

        encoder_states = (x,) if output_hidden_states else None
        for l in self.blocks:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        pooler_output = self.norm(x)

        if not self.ignore_head:
            pooler_output = self.attn_pool(pooler_output)
        return pooler_output, x, encoder_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig, ignore_head: bool = True):
        super().__init__()

        self.model_type = config.model_type
        self.config = config
        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_tower = SigLipVisionModel(config, ignore_head)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_tower(x, output_hidden_states)

    def sanitize(self, weights):
        sanitized_weights = {}
        weight_keys = {
            "neck.0.weight",
            "neck.2.weight",
            "neck_hd.0.weight",
            "neck_hd.2.weight",
            "downsamples.0.weight",
            "downsamples.1.weight",
            "patch_embed.proj.weight",
        }
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue

            elif ".".join(k.split(".")[-3:]) in weight_keys:
                # PyTorch conv2d weight tensors have shape:
                #   [out_channels, in_channels, kH, KW]
                # MLX conv2d expects the weight be of shape:
                #   [out_channels, kH, KW, in_channels]
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)

            else:
                sanitized_weights[k] = v

        return sanitized_weights
