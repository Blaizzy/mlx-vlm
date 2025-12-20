from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig


class Molmo2Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
    ):
        super().__init__()
        self.embedding = mx.zeros((num_embeddings, features))
        self.new_embedding = mx.zeros((num_new_embeddings, features))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([self.embedding, self.new_embedding], axis=0)[x]


class LanguageModelMLP(nn.Module):
    def __init__(self, input_dim: int, intermediate_size: int):
        super().__init__()
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ff_proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        x = nn.silu(gate) * x
        return self.ff_out(x)


class Molmo2Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )

        self.att_proj = nn.Linear(
            config.hidden_size,
            sum(self.fused_dims),
            bias=config.qkv_bias,
        )

        self.q_norm = nn.RMSNorm(dims=config.head_dim, eps=config.layer_norm_eps)
        self.k_norm = nn.RMSNorm(dims=config.head_dim, eps=config.layer_norm_eps)

        self.attn_out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            bias=False,
        )

        self.rotary_emb = nn.RoPE(self.head_dim, base=config.rope_theta)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.att_proj(hidden_states)
        q, k, v = mx.split(
            qkv,
            [self.fused_dims[0], self.fused_dims[0] + self.fused_dims[1]],
            axis=-1,
        )

        q = self.q_norm(q.reshape(batch_size, seq_len, self.num_heads, self.head_dim))
        k = self.k_norm(
            k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        )
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rotary_emb(q, offset=cache.offset)
            k = self.rotary_emb(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        att = scaled_dot_product_attention(q, k, v, cache, scale=self.scale, mask=mask)
        att = att.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.attn_out(att)


class Molmo2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Molmo2Attention(config)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = LanguageModelMLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, mask, cache)

        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Molmo2Transformer(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config

        self.wte = Molmo2Embedding(
            config.vocab_size, config.additional_vocab_size, config.hidden_size
        )
        self.blocks = [
            Molmo2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]
        self.ln_f = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.emb_drop = nn.Dropout(config.embedding_dropout)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            hidden_states = self.wte(input_ids)
        else:
            hidden_states = inputs_embeds

        if cache is None:
            cache = [None] * len(self.blocks)

        if mask is None:
            mask = create_attention_mask(hidden_states, cache)

        hidden_states = self.emb_drop(hidden_states)

        for block, c in zip(self.blocks, cache):
            hidden_states = block(hidden_states, mask, c)

        return self.ln_f(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Molmo2Transformer(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        hidden_states = self.model(input_ids, inputs_embeds, mask, cache)
        logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    @staticmethod
    def sanitize(weights):
        # Remove unused precomputed rotary freqs if present.
        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}

    @property
    def layers(self):
        return self.model.blocks
