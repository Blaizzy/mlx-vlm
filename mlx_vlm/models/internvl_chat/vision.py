import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import interpolate


@dataclass
class VisionConfig:
    model_type: str
    hidden_size: int = 1024
    num_attention_heads: int = 16
    patch_size: int = 14
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    image_size: int = 448
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    drop_path_rate: float = 0.1
    qkv_bias: bool = True
    qk_normalization: bool = False
    norm_type: str = "layer_norm"

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


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
    def __init__(self, config: VisionConfig):
        super().__init__()

        if (config.hidden_size % config.num_attention_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({config.hidden_size} % {config.num_attention_heads}) != 0"
            )

        self.dims = dims = config.hidden_size

        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.qkv_bias = config.qkv_bias

        self.qkv = nn.Linear(dims, 3 * dims, bias=config.qkv_bias)
        self.proj = nn.Linear(dims, dims)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = nn.RMSNorm(dims, eps=config.layer_norm_eps)
            self.k_norm = nn.RMSNorm(dims, eps=config.layer_norm_eps)

    def __call__(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        queries, keys, values = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # Each has shape (B, groups, N, C//groups)

        if self.qk_normalization:
            B_, H_, N_, D_ = queries.shape
            queries = (
                self.q_norm(queries.transpose(0, 2, 1, 3).flatten(-2, -1))
                .reshape(B_, N_, H_, D_)
                .transpose(0, 2, 1, 3)
            )
            keys = (
                self.k_norm(keys.transpose(0, 2, 1, 3).flatten(-2, -1))
                .reshape(B_, N_, H_, D_)
                .transpose(0, 2, 1, 3)
            )

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.proj(output)


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="precise")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = getattr(config, "norm_type", "layer_norm")

        self.attn = Attention(config)
        self.mlp = MLP(config)

        if self.norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        elif self.norm_type == "rms_norm":
            self.norm1 = nn.RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.norm2 = nn.RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")

        self.ls1 = mx.ones((self.embed_dim,))
        self.ls2 = mx.ones((self.embed_dim,))

        self.drop_path1 = (
            nn.Dropout(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            nn.Dropout(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        dtype = x.dtype
        x = x + self.drop_path1(self.attn(self.norm1(x).astype(dtype)) * self.ls1)

        x = x + self.drop_path2(self.mlp(self.norm2(x).astype(dtype)) * self.ls2)

        return x.astype(dtype)


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dpr = [
            mx.array(x)
            for x in np.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]
        self.layers = [
            EncoderLayer(config, dpr[i]) for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        encoder_states = (x,) if output_hidden_states else None
        h = x
        for l in self.layers:
            x = l(x, mask=mask)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

            h = x

        return (h, encoder_states)


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = mx.random.normal((1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = mx.random.normal(
            (1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.reshape(
            1,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            -1,
        ).transpose(0, 3, 1, 2)
        pos_embed = interpolate(pos_embed, (H, W))
        pos_embed = (
            pos_embed.reshape(1, -1, H * W).transpose(0, 2, 1).astype(target_dtype)
        )
        return pos_embed

    def __call__(self, x: mx.array) -> mx.array:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(x).transpose(
            0, 3, 1, 2
        )  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = mx.flatten(patch_embeds, start_axis=2).transpose(0, 2, 1)
        class_embeds = mx.broadcast_to(
            self.class_embedding, (batch_size, 1, self.embed_dim)
        ).astype(target_dtype)
        embeddings = mx.concatenate([class_embeds, patch_embeds], axis=1)
        position_embedding = mx.concatenate(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            axis=1,
        )
        embeddings = embeddings + position_embedding.astype(target_dtype)

        return embeddings


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model", "intern_vit_6b"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.embeddings(x)
        last_hidden_state, encoder_outputs = self.encoder(
            x=x, output_hidden_states=output_hidden_states, mask=None
        )
        pooler_output = last_hidden_state[:, 0, :]
        return last_hidden_state, pooler_output, encoder_outputs[1:]

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "patch_embedding.weight" in k:
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
