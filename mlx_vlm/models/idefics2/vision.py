import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    layer_norm_eps: float = 1e-6
    num_channels: int = 3

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
    def __init__(
        self,
        dims: int,
        num_heads: int,
        query_input_dims: Optional[int] = None,
        key_input_dims: Optional[int] = None,
        value_input_dims: Optional[int] = None,
        value_dims: Optional[int] = None,
        value_output_dims: Optional[int] = None,
    ):
        super().__init__()

        if (dims % num_heads) != 0:
            raise ValueError(
                "The input feature dimensions should be divisible by the "
                f"number of heads ({dims} % {num_heads}) != 0"
            )

        query_input_dims = query_input_dims or dims
        key_input_dims = key_input_dims or dims
        value_input_dims = value_input_dims or key_input_dims
        value_dims = value_dims or dims
        value_output_dims = value_output_dims or dims

        self.num_heads = num_heads
        head_dim = dims // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(query_input_dims, dims, bias=True)
        self.k_proj = nn.Linear(key_input_dims, dims, bias=True)
        self.v_proj = nn.Linear(value_input_dims, value_dims, bias=True)
        self.out_proj = nn.Linear(value_dims, value_output_dims, bias=True)

    def __call__(self, x: mx.array, mask=None):
        B, L, _ = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        num_heads = self.num_heads

        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        if mask is not None:
            mask = mask[:, :, mask.shape[-2] :, :]

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
        self.self_attn = Attention(config.hidden_size, config.num_attention_heads)
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


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

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

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = self.image_size // self.patch_size
        self.num_positions = self.num_patches**2
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, H, W, C = x.shape
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        max_nb_patches_h, max_nb_patches_w = (
            H // self.patch_size,
            W // self.patch_size,
        )
        boundaries = np.linspace(
            1 / self.num_patches, 1.0, self.num_patches, endpoint=False
        )
        position_ids = np.zeros((B, max_nb_patches_h * max_nb_patches_w), dtype=int)

        for batch_idx, p_attn_mask in enumerate(mask):
            p_attn_mask = np.array(p_attn_mask)
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0, :].sum()

            fractional_coords_h = np.linspace(0, 1, nb_patches_h, endpoint=False)
            fractional_coords_w = np.linspace(0, 1, nb_patches_w, endpoint=False)

            bucket_coords_h = (
                np.digitize(fractional_coords_h, boundaries, right=True) - 1
            )
            bucket_coords_w = (
                np.digitize(fractional_coords_w, boundaries, right=True) - 1
            )

            pos_ids = (
                bucket_coords_h[:, None] * self.num_patches + bucket_coords_w
            ).flatten()
            position_ids[batch_idx][p_attn_mask.reshape(-1)] = pos_ids

        embeddings = patch_embeddings
        embeddings += self.position_embedding(mx.array(position_ids))
        return embeddings


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if self.model_type not in ["idefics2", "idefics2_vision"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        self.embeddings = VisionEmbeddings(config)
        self.encoder = Encoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> mx.array:

        B, L, D, C = x.shape
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = mx.ones(
                (
                    B,
                    L // patch_size,
                    D // patch_size,
                ),
                dtype=mx.bool_,
            )

        x = self.embeddings(x, mask=patch_attention_mask)
        encoder_outputs = self.encoder(x=x, output_hidden_states=output_hidden_states)

        pooler_output = self.post_layernorm(encoder_outputs[0])

        return pooler_output, x, encoder_outputs[-1]

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "patch_embedding.weight" in k:
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
