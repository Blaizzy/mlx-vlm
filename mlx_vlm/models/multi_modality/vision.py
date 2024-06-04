import inspect
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str
    num_hidden_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    image_size: int = 384
    patch_size: int = 16
    projection_dim: int = 768
    vocab_size: int = 32000
    num_channels: int = 3
    layer_norm_eps: float = 1e-5

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
    hidden_size: int
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


def interpolate(arr, new_size, mode="bicubic", antialias=True):
    # Simple implementation of interpolation using NumPy
    old_size = arr.shape[-2:]
    scale_factors = [ns / os for ns, os in zip(new_size, old_size)]
    new_arr = np.zeros((arr.shape[0], arr.shape[1], *new_size))

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            old_i, old_j = i / scale_factors[0], j / scale_factors[1]
            old_i_floor, old_j_floor = int(old_i), int(old_j)
            old_i_ceil, old_j_ceil = min(old_i_floor + 1, old_size[0] - 1), min(
                old_j_floor + 1, old_size[1] - 1
            )

            # Perform interpolation (assuming bicubic)
            t, u = old_i - old_i_floor, old_j - old_j_floor
            new_arr[:, :, i, j] = (
                arr[:, :, old_i_floor, old_j_floor] * (1 - t) * (1 - u)
                + arr[:, :, old_i_ceil, old_j_floor] * t * (1 - u)
                + arr[:, :, old_i_floor, old_j_ceil] * (1 - t) * u
                + arr[:, :, old_i_ceil, old_j_ceil] * t * u
            )

    return new_arr


def resample_abs_pos_embed(
    posemb,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb
    if old_size is None:
        hw = int(np.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw
    if num_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb
    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.astype(np.float32)  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).transpose(0, 3, 1, 2)
    posemb = interpolate(posemb, new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.transpose(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.astype(orig_dtype)
    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = np.concatenate([posemb_prefix, posemb], axis=1)
    if verbose:
        print(f"Resized position embedding: {old_size} to {new_size}.")
    return posemb


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * mx.ones((dim,)))

    def forward(self, x: mx.array):
        return x @ self.gamma if self.inplace else x * self.gamma


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
            hidden_size=embed_dim, intermediate_size=int(embed_dim * mlp_ratio)
        )
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
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
        qkv_bias: bool = False,
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
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn = Attention(
            config.hidden_size, config.num_attention_heads, qkv_bias=True
        )
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
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.norm = nn.LayerNorm(config.hidden_size) if norm_layer else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        patch_embeddings = self.proj(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        return self.norm(patch_embeddings)


class SiglipVisionModel(nn.Module):
    def __init__(
        self, config: VisionConfig, pre_norm: bool = False, no_embed_class: bool = True
    ):
        super().__init__()
        self.num_prefix_tokens = 1
        self.no_embed_class = False
        self.dynamic_img_size = False
        self.cls_token = None
        self.reg_token = None
        self.patch_embed = VisionEmbeddings(config)
        self.norm_pre = nn.LayerNorm(config.hidden_size) if pre_norm else nn.Identity()
        self.blocks = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)
        num_patches = self.patch_embed.num_patches
        embed_len = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = (
            mx.random.normal((embed_len, config.hidden_size))[None, :] * 0.02
        )

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_pool = AttentionPoolLatent(
            config.hidden_size,
            num_heads=config.num_attention_heads,
            norm_layer=norm_layer,
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
        return self.attn_pool(pooler_output), x, encoder_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.model_type = config.model_type
        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.vision_model = SiglipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states)

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
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)

            else:
                sanitized_weights[k] = v

        return sanitized_weights
