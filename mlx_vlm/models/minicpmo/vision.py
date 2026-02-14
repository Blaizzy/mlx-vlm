from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def check_array_shape(arr):
    shape = arr.shape
    if len(shape) != 4:
        return False
    out_channels, k_h, k_w, _ = shape
    return (out_channels >= k_h) and (out_channels >= k_w) and (k_h == k_w)


class SiglipAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scale,
            mask=attention_mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class SiglipMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = nn.GELU(approx="precise")

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.mlp = SiglipMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = [
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class SiglipVisionEmbeddings(nn.Module):
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

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def __call__(
        self,
        pixel_values: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        tgt_sizes: Optional[mx.array] = None,
    ) -> mx.array:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = mx.flatten(patch_embeds, start_axis=1, end_axis=2)

        seq_len = embeddings.shape[1]
        max_nb_patches_h = int(pixel_values.shape[1] // self.patch_size)
        max_nb_patches_w = int(pixel_values.shape[2] // self.patch_size)

        if patch_attention_mask is None:
            patch_attention_mask = mx.ones(
                (batch_size, max_nb_patches_h, max_nb_patches_w), dtype=mx.bool_
            )
        elif patch_attention_mask.ndim == 2:
            patch_attention_mask = patch_attention_mask[:, None, :]

        boundaries = mx.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side
        )
        position_ids = np.zeros((batch_size, seq_len), dtype=np.int32)

        for batch_idx in range(batch_size):
            if tgt_sizes is not None:
                nb_patches_h = max(int(tgt_sizes[batch_idx, 0]), 1)
                nb_patches_w = max(int(tgt_sizes[batch_idx, 1]), 1)
            else:
                cur_mask = patch_attention_mask[batch_idx]
                nb_patches_h = max(int(mx.sum(cur_mask[:, 0]).item()), 1)
                nb_patches_w = max(int(mx.sum(cur_mask[0]).item()), 1)

            fractional_h = mx.clip(
                mx.arange(nb_patches_h, dtype=mx.float32) / nb_patches_h,
                a_min=0.0,
                a_max=1.0 - 1e-6,
            )
            fractional_w = mx.clip(
                mx.arange(nb_patches_w, dtype=mx.float32) / nb_patches_w,
                a_min=0.0,
                a_max=1.0 - 1e-6,
            )
            bucket_h = mx.sum(fractional_h[:, None] >= boundaries[None, :], axis=1)
            bucket_w = mx.sum(fractional_w[:, None] >= boundaries[None, :], axis=1)

            pos_ids = (
                bucket_h[:, None] * self.num_patches_per_side + bucket_w[None, :]
            ).reshape(-1)
            pos_ids = np.array(pos_ids, dtype=np.int32)

            flat_mask = np.array(patch_attention_mask[batch_idx]).reshape(-1)[:seq_len]
            valid_indices = np.where(flat_mask)[0]
            valid_len = min(len(valid_indices), len(pos_ids))
            if valid_len > 0:
                position_ids[batch_idx, valid_indices[:valid_len]] = pos_ids[:valid_len]

        position_ids = mx.array(position_ids, dtype=mx.int32)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        if self.model_type not in ["siglip_vision_model", "siglip", "minicpmo"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self,
        pixel_values: mx.array,
        patch_attention_mask: Optional[mx.array] = None,
        tgt_sizes: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )
        hidden_states = hidden_states.astype(
            self.embeddings.patch_embedding.weight.dtype
        )
        hidden_states = self.encoder(hidden_states=hidden_states, attention_mask=None)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if "position_ids" in key:
                continue
            if key.endswith("patch_embedding.weight"):
                if not check_array_shape(value):
                    value = value.transpose(0, 2, 3, 1)
            sanitized_weights[key] = value
        return sanitized_weights
