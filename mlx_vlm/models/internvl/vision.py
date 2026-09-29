import mlx.core as mx
import mlx.nn as nn

from ..base import interpolate
from ..internvl_chat.vision import MLP, check_array_shape
from .config import VisionConfig


def _normalization(config):
    if config.norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class PatchEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, pixel_values):
        return self.projection(pixel_values).reshape(
            pixel_values.shape[0], -1, self.projection.weight.shape[0]
        )


class Embeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.position_embeddings = mx.zeros(
            (1, (config.image_size // config.patch_size) ** 2 + 1, config.hidden_size)
        )
        self.image_size = config.image_size
        self.patch_size = config.patch_size

    def __call__(self, pixel_values):
        patches = self.patch_embeddings(pixel_values)
        height = pixel_values.shape[1] // self.patch_size
        width = pixel_values.shape[2] // self.patch_size
        cls_token = mx.broadcast_to(
            self.cls_token, (pixel_values.shape[0], 1, self.cls_token.shape[-1])
        )
        position_embeddings = self.position_embeddings
        if patches.shape[1] + 1 != position_embeddings.shape[1]:
            positions = position_embeddings[:, 1:].reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            positions = interpolate(
                positions.transpose(0, 3, 1, 2), (height, width)
            ).transpose(0, 2, 3, 1)
            position_embeddings = mx.concatenate(
                [
                    position_embeddings[:, :1],
                    positions.reshape(1, -1, positions.shape[-1]),
                ],
                axis=1,
            )
        return mx.concatenate([cls_token, patches], axis=1) + position_embeddings


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.projection_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )

    def __call__(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        queries = self.q_norm(self.q_proj(hidden_states))
        keys = self.k_norm(self.k_proj(hidden_states))
        values = self.v_proj(hidden_states)
        queries = queries.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.projection_layer(output)


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.layernorm_before = _normalization(config)
        self.layernorm_after = _normalization(config)
        self.lambda_1 = mx.full((config.hidden_size,), config.layer_scale_init_value)
        self.lambda_2 = mx.full((config.hidden_size,), config.layer_scale_init_value)

    def __call__(self, hidden_states):
        hidden_states = (
            hidden_states
            + self.attention(self.layernorm_before(hidden_states)) * self.lambda_1
        )
        return (
            hidden_states
            + self.mlp(self.layernorm_after(hidden_states)) * self.lambda_2
        )


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, hidden_states, output_hidden_states=False):
        states = (hidden_states,) if output_hidden_states else None
        for layer in self.layer:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                states += (hidden_states,)
        return hidden_states, states


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def __call__(self, pixel_values, output_hidden_states=False):
        hidden_states = self.embeddings(pixel_values)
        return self.encoder(hidden_states, output_hidden_states)

    def sanitize(self, weights):
        for key in list(weights):
            if key.endswith("patch_embeddings.projection.weight"):
                weight = weights[key]
                if not check_array_shape(weight):
                    weights[key] = weight.transpose(0, 2, 3, 1)
        return weights
