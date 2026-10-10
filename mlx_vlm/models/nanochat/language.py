import math
from functools import partial
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig


def rms_norm(x):
    return mx.fast.rms_norm(x, None, 1e-5)


def apply_rotary_emb(x, offset, base=10000.0, freqs=None):
    head_dim = x.shape[-1]
    if freqs is None:
        half_dim = head_dim // 2
        freqs = -mx.exp(
            mx.arange(0.0, half_dim, dtype=mx.float32) * (math.log(base) / half_dim)
        )

    return mx.fast.rope(
        x,
        dims=head_dim,
        traditional=False,
        base=None,
        freqs=freqs,
        scale=1.0,
        offset=offset,
    )


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.c_q = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.c_k = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_v = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        half_dim = self.head_dim // 2
        self._rope_freqs = -mx.exp(
            mx.arange(0.0, half_dim, dtype=mx.float32)
            * (math.log(args.rope_theta) / half_dim)
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape

        queries = self.c_q(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        keys = self.c_k(x).reshape(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )
        values = self.c_v(x).reshape(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        )

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        queries = apply_rotary_emb(queries, offset, freqs=self._rope_freqs)
        keys = apply_rotary_emb(keys, offset, freqs=self._rope_freqs)
        queries = rms_norm(queries)
        keys = rms_norm(keys)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.c_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(nn.relu2(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.attn = Attention(args)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attn(rms_norm(x), mask=mask, cache=cache)
        return h + self.mlp(rms_norm(h))


class NanoChatModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.wte(inputs) if inputs_embeds is None else inputs_embeds
        h = rms_norm(h)

        if cache is None:
            cache = [None] * len(self.h)
        mask = create_attention_mask(h, cache[0])

        for layer, layer_cache in zip(self.h, cache):
            h = layer(h, mask=mask, cache=layer_cache)

        return rms_norm(h)


@partial(mx.compile, shapeless=True)
def softcap(logits, cap=15.0):
    return cap * mx.tanh(logits / cap)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.transformer = NanoChatModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        hidden_states = self.transformer(inputs, cache, inputs_embeds)
        return LanguageModelOutput(logits=softcap(self.lm_head(hidden_states)))

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.transformer.h

    def make_cache(self):
        return [KVCache() for _ in self.layers]
