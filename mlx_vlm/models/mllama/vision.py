import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    image_size: int = 560
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    max_num_tiles: int = 4
    max_aspect_ratio_id: int = 8
    num_global_layers: int = 8
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    vision_output_dim: int = 7680
    intermediate_layers_indices: List[int] = field(
        default_factory=lambda: [3, 7, 15, 23, 30]
    )
    supported_aspect_ratios: Tuple[List[int]] = (
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 1],
        [2, 2],
        [3, 1],
        [4, 1],
    )

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


class MllamaVisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.embed_dim, bias=False
        )

    def __call__(
        self,
        hidden_state: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.reshape(
            batch_size, q_seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = key.reshape(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value = value.reshape(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, : key.shape[-2], :]

        attn_output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=attention_mask
        )

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        return self.o_proj(attn_output)


class MllamaVisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.gelu = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig, is_gated: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.is_gated = is_gated

        self.self_attn = MllamaVisionAttention(config)
        self.mlp = MllamaVisionMLP(config)

        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=config.norm_eps
        )

        if is_gated:
            self.gate_attn = mx.zeros(1)
            self.gate_ffn = mx.zeros(1)

    def __call__(
        self,
        hidden_state: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Self Attention
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = mx.tanh(self.gate_attn) * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = mx.tanh(self.gate_ffn) * hidden_state
        hidden_state = residual + hidden_state

        return hidden_state


class MllamaVisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig, num_layers=32, is_gated=False):
        super().__init__()
        self.layers = [
            MllamaVisionEncoderLayer(config, is_gated) for _ in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, List[mx.array]]:
        encoder_states = ()
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            encoder_states = encoder_states + (hidden_states,)
        return hidden_states, encoder_states


class MllamaPrecomputedAspectRatioEmbedding(nn.Module):
    def __init__(self, config: VisionConfig, is_gated: bool = True):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.is_gated = is_gated

        self.embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1, self.max_num_tiles * self.hidden_size
        )
        if is_gated:
            self.gate = mx.zeros(1)

    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array:
        embeddings = self.embedding(aspect_ratio_ids)
        embeddings = embeddings.reshape(-1, self.max_num_tiles, 1, self.hidden_size)

        if self.is_gated:
            embeddings = embeddings * mx.tanh(self.gate)

        hidden_state = hidden_state + embeddings
        return hidden_state


class MllamaPrecomputedPositionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.max_num_tiles = config.max_num_tiles
        self.max_aspect_ratio_id = config.max_aspect_ratio_id
        self.num_patches = (config.image_size // config.patch_size) ** 2 + 1
        self.hidden_size = config.hidden_size
        self.scale = config.hidden_size**-0.5

        self.gate = mx.zeros(1)

        # position embedding
        self.embedding = (
            mx.random.normal((self.num_patches, self.hidden_size)) * self.scale
        )

        # tile position embedding
        self.tile_embedding = nn.Embedding(
            self.max_aspect_ratio_id + 1,
            self.max_num_tiles * self.num_patches * self.hidden_size,
        )

    def __call__(self, hidden_state: mx.array, aspect_ratio_ids: mx.array) -> mx.array:
        # position embeddings
        gated_position_embedding = (1 - mx.tanh(self.gate)) * self.embedding
        hidden_state = hidden_state + gated_position_embedding.reshape(
            1, 1, self.num_patches, self.hidden_size
        )

        # precomputed tile position embeddings
        tile_position_embedding = self.tile_embedding(aspect_ratio_ids)
        batch_size = hidden_state.shape[0]
        tile_position_embedding = tile_position_embedding.reshape(
            batch_size, self.max_num_tiles, self.num_patches, self.hidden_size
        )
        gated_tile_position_embedding = mx.tanh(self.gate) * tile_position_embedding
        hidden_state = hidden_state + gated_tile_position_embedding

        return hidden_state


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.max_num_tiles = config.max_num_tiles
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.intermediate_layers_indices = config.intermediate_layers_indices

        self.num_patches = (self.image_size // self.patch_size) ** 2 + 1
        self.scale = config.hidden_size**-0.5

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.class_embedding = mx.random.normal((self.hidden_size,)) * self.scale
        self.gated_positional_embedding = MllamaPrecomputedPositionEmbedding(config)

        self.pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True
        )
        self.post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding(
            config, is_gated=True
        )

        # layer norms
        self.layernorm_pre = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)
        self.layernorm_post = nn.LayerNorm(self.hidden_size, eps=config.norm_eps)

        # encoders
        self.transformer = MllamaVisionEncoder(
            config, config.num_hidden_layers, is_gated=False
        )
        self.global_transformer = MllamaVisionEncoder(
            config, config.num_global_layers, is_gated=True
        )

    def __call__(
        self,
        pixel_values: mx.array,
        aspect_ratio_ids: mx.array,
        aspect_ratio_mask: mx.array,
    ) -> mx.array:
        batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
            pixel_values.shape
        )
        aspect_ratio_ids = aspect_ratio_ids.reshape(
            batch_size * num_concurrent_media, -1
        )

        pixel_values = pixel_values.reshape(
            batch_size * num_concurrent_media * num_tiles, num_channels, height, width
        )
        # Patch embedding
        patch_embeds = self.patch_embedding(pixel_values.moveaxis(1, 3)).moveaxis(3, 1)

        hidden_state = patch_embeds.reshape(
            patch_embeds.shape[0], patch_embeds.shape[1], -1
        ).transpose(0, 2, 1)

        # Tile embeddings
        _, num_patches, dim = hidden_state.shape
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, -1, dim
        )
        hidden_state = self.pre_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )

        # Add cls token
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media * num_tiles, num_patches, dim
        )
        class_embedding = mx.broadcast_to(
            self.class_embedding,
            (batch_size * num_concurrent_media * num_tiles, 1, dim),
        )
        hidden_state = mx.concatenate([class_embedding, hidden_state], axis=1)
        num_patches += 1

        # Position embeddings
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, num_tiles, num_patches, dim
        )
        hidden_state = self.gated_positional_embedding(hidden_state, aspect_ratio_ids)

        hidden_state = self.layernorm_pre(hidden_state)

        # Compute the number of tokens to pad
        num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8

        # Pad the tensor
        padding = [(0, 0), (0, 0), (0, num_padding_patches), (0, 0)]
        hidden_state = mx.pad(hidden_state, padding)
        slice_index = -num_padding_patches if num_padding_patches > 0 else None

        # Prepare attention mask
        attention_mask = aspect_ratio_mask.reshape(
            batch_size * num_concurrent_media, -1
        )
        attention_mask = _prepare_aspect_ratio_attention_mask(
            aspect_ratio_mask=attention_mask,
            num_patches=self.num_patches,
            target_length=hidden_state.shape[2],
        )

        # Apply encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media, -1, self.hidden_size
        )
        output = self.transformer(hidden_state, attention_mask=attention_mask)

        hidden_state = output[0]

        hidden_state = self.layernorm_post(hidden_state)

        # Apply global encoder
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            self.hidden_size,
        )
        hidden_state = self.post_tile_positional_embedding(
            hidden_state, aspect_ratio_ids
        )
        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            self.hidden_size,
        )
        global_output = self.global_transformer(
            hidden_state, attention_mask=attention_mask
        )

        hidden_state = global_output[0]

        hidden_state = hidden_state.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        )

        hidden_state = hidden_state[:, :, :slice_index]
        hidden_state = hidden_state.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, dim
        )

        # Collect intermediate layer outputs from encoder output
        all_intermediate_hidden_states = output[1]
        intermediate_hidden_states = mx.stack(all_intermediate_hidden_states, axis=-1)
        intermediate_hidden_states = intermediate_hidden_states[
            ..., self.intermediate_layers_indices
        ]

        # Remove padding from intermediate hidden states
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            -1,
        )
        intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
        intermediate_hidden_states = intermediate_hidden_states.reshape(
            batch_size, num_concurrent_media, num_tiles, num_patches, -1
        )

        # Concatenate final hidden state and intermediate hidden states
        hidden_state = mx.concatenate(
            [hidden_state, intermediate_hidden_states], axis=-1
        )

        return hidden_state

    @staticmethod
    def sanitize(weights):
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


def _prepare_aspect_ratio_attention_mask(
    aspect_ratio_mask: mx.array,
    num_patches: int,
    target_length: int,
) -> mx.array:
    dtype = mx.float32
    aspect_ratio_mask = aspect_ratio_mask.astype(dtype)

    # Expand aspect ratio mask to target_length
    batch_size, max_num_tiles = aspect_ratio_mask.shape
    attention_mask = aspect_ratio_mask.reshape(batch_size, max_num_tiles, 1, 1).astype(
        dtype
    )
    attention_mask = mx.tile(attention_mask, (1, 1, target_length, 1))

    # Mask padding patches
    pad_patches = target_length - num_patches
    attention_mask[:, :, -pad_patches:] = 0

    # Invert the mask (0 -> 1, 1 -> 0)
    attention_mask = 1 - attention_mask

    # Reshape to 2D and create 4D attention mask
    # (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
    attention_mask = attention_mask.reshape(
        batch_size, max_num_tiles * target_length, 1
    )

    min_value = -1e9
    attention_mask = attention_mask @ attention_mask.transpose(0, 2, 1) * min_value
    attention_mask = attention_mask[:, None, :, :]

    return attention_mask
