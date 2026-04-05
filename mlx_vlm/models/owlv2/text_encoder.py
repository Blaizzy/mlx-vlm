"""OWLv2 Text Transformer encoder."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import TextConfig
from .vision import quick_gelu


class TextEmbeddings(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)[None]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class TextAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return self.out_proj(out)


class TextMLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(quick_gelu(self.fc1(x)))


class TextEncoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = TextAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TextMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask=mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.layers = [
            TextEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class TextTransformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embeddings = TextEmbeddings(config)
        self.encoder = TextEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        x = self.embeddings(input_ids)

        # Build causal mask
        seq_len = input_ids.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(
            x.dtype
        )
        if attention_mask is not None:
            # Expand attention_mask (B, L) -> (B, 1, 1, L) additive mask
            pad_mask = (1.0 - attention_mask[:, None, None, :].astype(x.dtype)) * -1e9
            causal_mask = causal_mask + pad_mask

        x = self.encoder(x, mask=causal_mask)
        x = self.final_layer_norm(x)
        return x
