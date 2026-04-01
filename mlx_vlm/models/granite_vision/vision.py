from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def check_array_shape(arr):
    shape = arr.shape
    if len(shape) != 4:
        return False
    out_channels, kH, KW, _ = shape
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    return False


class Attention(nn.Module):
    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries, keys, values, mask=None):
        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

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


class MHA(nn.Module):
    """Multi-head attention with fused QKV projection (for pooling head)."""

    def __init__(
        self,
        dims: int,
        num_heads: int,
        bias: bool = True,
    ):
        super().__init__()
        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )
        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5
        self.in_proj = nn.Linear(dims, dims * 3, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, queries: mx.array, kv: mx.array, mask=None):
        B, L, D = queries.shape

        qkv = self.in_proj(queries)
        _, keys, values = mx.split(qkv, 3, axis=-1)

        num_heads = self.num_heads
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
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU(approx="fast")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Attention(
            config.hidden_size, config.num_attention_heads, bias=True
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        y = self.layer_norm1(x)
        y = self.self_attn(y, y, y, mask)
        x = x + y
        y = self.layer_norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]


class VisionEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        position_ids = mx.array(np.arange(self.num_positions)[None, :])
        embeddings = patch_embeddings + self.position_embedding(position_ids)
        return embeddings


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.probe = mx.ones((1, 1, config.hidden_size))
        self.attention = MHA(
            config.hidden_size, num_heads=config.num_attention_heads, bias=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array):
        x = self.attention(self.probe, x)[0]
        residual = x
        x = self.layernorm(x)
        x = residual + self.mlp(x)
        return x[:, 0]


class SigLipVisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def __call__(
        self,
        x: mx.array,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:
        x = self.embeddings(x)

        encoder_states = (x,) if output_hidden_states else None

        for l in self.encoder.layers:
            x = l(x, mask=None)
            if output_hidden_states:
                encoder_states = encoder_states + (x,)

        pooler_output = self.post_layernorm(x)
        return pooler_output, x, encoder_states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.vision_model = SigLipVisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_model(x, output_hidden_states)

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            elif "patch_embedding.weight" in k:
                if check_array_shape(v):
                    sanitized_weights[k] = v
                else:
                    sanitized_weights[k] = v.transpose(0, 2, 3, 1)
            elif "head.attention.in_proj_weight" in k:
                # PyTorch nn.MultiheadAttention stores as in_proj_weight parameter
                sanitized_weights[k.replace("in_proj_weight", "in_proj.weight")] = v
            elif "head.attention.in_proj_bias" in k:
                sanitized_weights[k.replace("in_proj_bias", "in_proj.bias")] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights
