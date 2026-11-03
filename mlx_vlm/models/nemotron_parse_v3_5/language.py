# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math

import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import TextConfig


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_heads = config.decoder_attention_heads
        self.num_kv_heads = config.self_attention_kv_heads
        self.head_dim = config.d_model // self.num_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

    def __call__(self, x, mask, cache=None):
        batch, length, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.head_dim**-0.5, mask=mask
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def __call__(self, x, mask, cache=None):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask, cache)
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig, max_sequence_length=16384):
        super().__init__()
        self.config = config
        self.max_sequence_length = max_sequence_length
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.layers = [DecoderLayer(config) for _ in range(config.decoder_layers)]
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def embed(self, ids):
        return self.embed_tokens(ids) * self.embed_scale

    def __call__(
        self, inputs=None, inputs_embeds=None, cache=None, logits_to_keep=None, **kwargs
    ):
        x = self.embed(inputs) if inputs_embeds is None else inputs_embeds
        past = cache[0].offset if cache else 0
        if past + x.shape[1] > self.max_sequence_length:
            raise ValueError("Visual prefix plus text exceeds the model context limit")
        # The visual prefix is causal too. There are no decoder position embeddings.
        mask = create_attention_mask(x, cache=cache[0] if cache else None)
        x = self.layernorm_embedding(x)
        for layer, c in zip(self.layers, cache or [None] * len(self.layers)):
            x = layer(x, mask, c)
        x = self.layer_norm(x)
        if logits_to_keep is not None:
            x = x[:, -logits_to_keep:]
        return LanguageModelOutput(logits=self.embed_tokens.as_linear(x))

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    @property
    def head_dim(self):
        return self.config.d_model // self.config.decoder_attention_heads

    @property
    def n_kv_heads(self):
        return self.config.self_attention_kv_heads
