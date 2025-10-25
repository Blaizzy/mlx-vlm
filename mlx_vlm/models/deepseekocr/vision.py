import math
from math import sqrt
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import interpolate
from .config import VisionConfig


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

        self.qkv_proj = nn.Linear(dims, dims * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dims, dims, bias=True)

    def __call__(self, x, mask=None):
        qkv = self.qkv_proj(x)
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

        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config: Union[VisionConfig, Dict], bias: bool = True):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
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
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, qkv_bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = 224
        self.patch_size = config.patch_size

        self.class_embedding = mx.random.normal((self.embed_dim,))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def _get_abs_pos(self, abs_pos, tgt_size):
        """
        Resize absolute positional embeddings

        Args:
            abs_pos: Tensor of shape (L, C) - absolute position embeddings
            tgt_size: int - target size M

        Returns:
            Tensor of shape (M, C) - resized position embeddings
        """
        dim = abs_pos.shape[-1]
        abs_pos_new = mx.squeeze(abs_pos, axis=0)
        cls_token, old_pos_embed = abs_pos_new[:1], abs_pos_new[1:]
        src_size = int(math.sqrt(abs_pos_new.shape[0] - 1))
        tgt_size_2d = int(math.sqrt(tgt_size))
        dtype = abs_pos.dtype

        if src_size != tgt_size_2d:
            # Reshape to (1, src_size, src_size, dim) then transpose to (1, dim, src_size, src_size)
            old_pos_embed = mx.reshape(old_pos_embed, (1, src_size, src_size, dim))
            old_pos_embed = mx.transpose(old_pos_embed, (0, 3, 1, 2))
            old_pos_embed = old_pos_embed.astype(mx.float32)

            new_pos_embed = interpolate(old_pos_embed, (tgt_size_2d, tgt_size_2d))

            new_pos_embed = new_pos_embed.astype(dtype)
            new_pos_embed = mx.transpose(new_pos_embed, (0, 2, 3, 1))
            new_pos_embed = mx.reshape(new_pos_embed, (tgt_size_2d * tgt_size_2d, dim))
            vision_pos_embed = mx.concatenate([cls_token, new_pos_embed], axis=0)
            vision_pos_embed = mx.reshape(
                vision_pos_embed, (1, tgt_size_2d * tgt_size_2d + 1, dim)
            )
            return vision_pos_embed
        else:
            return abs_pos

    def __call__(
        self, x: mx.array, patch_embeds: Optional[mx.array] = None
    ) -> mx.array:
        batch_size, height, width, _ = x.shape
        target_dtype = self.position_embedding.weight.dtype

        if patch_embeds is not None:
            patch_embeddings = patch_embeds
        else:
            patch_embeddings = self.patch_embedding(x)

        # Flatten patch embeddings properly
        patch_embeds = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)

        # Broadcast class embedding
        class_embeds = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, self.embed_dim)
        ).astype(target_dtype)

        # Concatenate class and patch embeddings
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)

        # Create position IDs
        position_ids = mx.array(np.arange(self.num_positions)[None, :])

        # Add positional embeddings
        embeddings = embeddings + self._get_abs_pos(
            self.position_embedding(position_ids), embeddings.shape[1]
        ).astype(target_dtype)

        return embeddings


class NoTPTransformer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_layers = config.layers
        self.layers = [EncoderLayer(config) for _ in range(config.layers)]

    def __call__(
        self,
        x: mx.array,
    ) -> mx.array:
        for l in self.layers:
            x = l(x, mask=None)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()

        self.model_type = config.model_type
        self.config = config
        if self.model_type != "vision":
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = VisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.transformer = NoTPTransformer(config)

    def __call__(
        self,
        x: mx.array,
        patch_embeds: mx.array = None,
    ) -> mx.array:
        x = self.embeddings(x, patch_embeds)
        x = self.pre_layrnorm(x)
        return self.transformer(x)

    def sanitize(self, weights):
        sanitized_weights = {}
        weight_keys = {
            "neck.0.weight",
            "neck.2.weight",
            "neck_hd.0.weight",
            "neck_hd.2.weight",
            "sam_model.net_2.weight",
            "sam_model.net_3.weight",
            "downsamples.0.weight",
            "downsamples.1.weight",
            "patch_embed.proj.weight",
            "embeddings.patch_embedding.weight",
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
