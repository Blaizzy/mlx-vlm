from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        head_dim = self.hidden_size // config.num_attention_heads
        self.q_num_heads = config.num_attention_heads
        self.qk_dim = self.v_dim = head_dim
        self.k_num_heads = self.v_num_heads = int(
            np.ceil(self.q_num_heads / config.n_shared_head)
        )
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_num_heads * self.qk_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.k_num_heads * self.qk_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.v_num_heads * self.v_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.q_num_heads * self.v_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=1.0,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(
            batch_size, sequence_length, self.q_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.k_num_heads, self.qk_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.v_num_heads, self.v_dim
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rotary_emb(queries, offset=cache.offset)
            keys = self.rotary_emb(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rotary_emb(queries)
            keys = self.rotary_emb(keys)

        keys = mx.tile(keys, [1, self.config.n_shared_head, 1, 1])
        values = mx.tile(values, [1, self.config.n_shared_head, 1, 1])
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=attention_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class PlamoDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attention = self.self_attn(hidden_states, attention_mask, cache)
        mlp = self.mlp(hidden_states)
        return residual + attention + mlp


class PlamoDecoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.layers = [
            PlamoDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]


class PlamoModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = PlamoDecoder(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, layer_cache in zip(self.layers.layers, cache):
            h = layer(h, mask, cache=layer_cache)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig) -> None:
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = PlamoModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        hidden_states = self.model(inputs, cache, inputs_embeds)
        return LanguageModelOutput(logits=self.lm_head(hidden_states))

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]
