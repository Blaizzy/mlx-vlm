from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig


def check_conv1d_weight_shape(arr):
    shape = arr.shape
    if len(shape) != 3:
        return False
    out_channels, kernel_size, in_channels = shape
    return out_channels >= kernel_size and in_channels >= 1


class AudioAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
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

        mask = None
        if attention_mask is not None:
            # attention_mask is boolean [B, L] with True for valid tokens.
            invalid = mx.logical_not(attention_mask)
            mask = mx.where(
                invalid[:, None, None, :],
                mx.array(-1e9, dtype=query_states.dtype),
                mx.array(0.0, dtype=query_states.dtype),
            )

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scale,
            mask=mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class AudioMLP(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)
        self.activation_fn = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(
            config.d_model, eps=config.layer_norm_eps
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attn = AudioAttention(config)
        self.mlp = AudioMLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class AudioProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def __call__(self, audio_features: mx.array) -> mx.array:
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.conv1 = nn.Conv1d(
            in_channels=config.num_mel_bins,
            out_channels=config.d_model,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = AudioEncoder(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def __call__(
        self,
        input_features: mx.array,
        feature_lengths: Optional[mx.array] = None,
    ) -> mx.array:
        # input_features: [B, 80, frames]
        if input_features.ndim != 3:
            raise ValueError(
                f"Expected input_features with 3 dims [B, 80, T], got {input_features.shape}"
            )

        hidden_states = input_features.transpose(0, 2, 1)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        batch_size, seq_len, _ = hidden_states.shape

        position_ids = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_len))
        hidden_states = hidden_states + self.embed_positions(position_ids)

        attention_mask = None
        if feature_lengths is not None:
            conv_lengths = ((feature_lengths - 1) // 2) + 1
            token_ids = mx.arange(seq_len, dtype=mx.int32)[None, :]
            attention_mask = token_ids < conv_lengths[:, None]

        hidden_states = self.layers(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if key.endswith("conv1.weight") or key.endswith("conv2.weight"):
                # PyTorch Conv1d: [out, in, k], MLX Conv1d: [out, k, in]
                if not check_conv1d_weight_shape(value):
                    value = value.transpose(0, 2, 1)
            sanitized_weights[key] = value
        return sanitized_weights
