# Copyright © 2023-2024 Apple Inc.

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from ..rope_utils import initialize_rope
from .bitlinear import BitLinear
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // self.n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = BitLinear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = BitLinear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = BitLinear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = BitLinear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )
        self.attn_sub_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(
            batch_size, sequence_length, self.n_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, sequence_length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(self.attn_sub_norm(output))


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.gate_proj = BitLinear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.down_proj = BitLinear(
            args.intermediate_size, args.hidden_size, bias=args.mlp_bias
        )
        self.up_proj = BitLinear(
            args.hidden_size, args.intermediate_size, bias=args.mlp_bias
        )
        self.ffn_sub_norm = nn.RMSNorm(args.intermediate_size, eps=args.rms_norm_eps)

    def __call__(self, x) -> mx.array:
        hidden_states = nn.relu2(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        hidden_states = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class BitNetModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(hidden_states, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask, cache=layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = BitNetModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        hidden_states = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
