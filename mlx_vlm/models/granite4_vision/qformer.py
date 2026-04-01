"""Blip2 QFormer implementation in MLX for granite4_vision."""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class QFormerConfig:
    hidden_size: int = 1152
    num_attention_heads: int = 18
    intermediate_size: int = 3072
    encoder_hidden_size: int = 1152
    num_hidden_layers: int = 1


class Blip2QFormerSelfAttention(nn.Module):
    """Multi-head attention used for both self-attention and cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        encoder_hidden_size: int | None = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden_size, hidden_size)
        kv_input_size = (
            encoder_hidden_size if encoder_hidden_size is not None else hidden_size
        )
        self.key = nn.Linear(kv_input_size, hidden_size)
        self.value = nn.Linear(kv_input_size, hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        queries = self.query(hidden_states)

        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        keys = self.key(kv_input)
        values = self.value(kv_input)

        _, S, _ = keys.shape

        queries = queries.reshape(
            B, L, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, S, self.num_attention_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            B, S, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0 / self.scale
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return output


class Blip2QFormerSelfOutput(nn.Module):
    """Output projection + residual + LayerNorm for attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Capital 'L' and 'N' to match HuggingFace weight keys
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Blip2QFormerAttention(nn.Module):
    """Wraps self-attention (or cross-attention) + output projection."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        encoder_hidden_size: int | None = None,
    ):
        super().__init__()
        self.attention = Blip2QFormerSelfAttention(
            hidden_size, num_attention_heads, encoder_hidden_size
        )
        self.output = Blip2QFormerSelfOutput(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        attn_output = self.attention(hidden_states, encoder_hidden_states)
        output = self.output(attn_output, hidden_states)
        return output


class Blip2QFormerIntermediate(nn.Module):
    """FFN up-projection with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return hidden_states


class Blip2QFormerOutput(nn.Module):
    """FFN down-projection + residual + LayerNorm."""

    def __init__(self, intermediate_size: int, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        # Capital 'L' and 'N' to match HuggingFace weight keys
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def __call__(self, hidden_states: mx.array, residual: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class Blip2QFormerLayer(nn.Module):
    """Single QFormer encoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        # Self-attention
        self.attention = Blip2QFormerAttention(
            config.hidden_size, config.num_attention_heads
        )
        # Cross-attention (key/value from encoder)
        self.crossattention = Blip2QFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            encoder_hidden_size=config.encoder_hidden_size,
        )
        # FFN
        self.intermediate_query = Blip2QFormerIntermediate(
            config.hidden_size, config.intermediate_size
        )
        self.output_query = Blip2QFormerOutput(
            config.intermediate_size, config.hidden_size
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        # 1. Self-attention
        attention_output = self.attention(hidden_states)

        # 2. Cross-attention
        cross_output = self.crossattention(attention_output, encoder_hidden_states)

        # 3. FFN
        intermediate_output = self.intermediate_query(cross_output)
        layer_output = self.output_query(intermediate_output, cross_output)

        return layer_output


class Blip2QFormerEncoder(nn.Module):
    """Stack of QFormer layers."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.layer = [
            Blip2QFormerLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, encoder_hidden_states)
        return hidden_states


class Blip2QFormerModel(nn.Module):
    """Full QFormer: LayerNorm on queries, then encoder layers."""

    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.encoder = Blip2QFormerEncoder(config)

    def __call__(
        self,
        query_embeds: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        hidden_states = self.layernorm(query_embeds)
        hidden_states = self.encoder(hidden_states, encoder_hidden_states)
        return hidden_states
