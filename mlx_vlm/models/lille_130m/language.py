from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig


class Lille130mAttention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_head = args.n_head
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.n_embd // args.n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            args.n_embd,
            (args.n_head + 2 * args.n_kv_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.n_head * self.head_dim, args.n_embd, bias=False)
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)
        self.rope = nn.RoPE(self.head_dim, True, args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_proj(self.norm(x))

        query_size = self.n_head * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        queries, keys, values = mx.split(
            qkv, [query_size, query_size + kv_size], axis=-1
        )
        queries = queries.reshape(
            batch_size, sequence_length, self.n_head, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size, sequence_length, self.n_kv_heads, self.head_dim
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
        return self.out_proj(output)


class Lille130mMLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        hidden_dim = 256 * round(int(8 * args.n_embd / 3) / 256)
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)
        self.gate_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.up_proj = nn.Linear(args.n_embd, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, args.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        return self.down_proj(swiglu(self.gate_proj(h), self.up_proj(h)))


class Lille130Block(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.attention = Lille130mAttention(args)
        self.feed_forward = Lille130mMLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attention(x, mask, cache)
        return h + self.feed_forward(h)


class Lille130(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.n_embd)
        self.layers = [Lille130Block(args) for _ in range(args.n_layer)]
        self.norm = nn.RMSNorm(args.n_embd, eps=args.layer_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.tok_embeddings(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, mask, cache=layer_cache)
        return self.tok_embeddings.as_linear(self.norm(h))


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.transformer = Lille130(args)

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
        logits = self.transformer(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return LanguageModelOutput(logits=logits)

    @property
    def layers(self):
        return self.transformer.layers

    def make_cache(self):
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights):
        return {key: value for key, value in weights.items() if "rotary_emb" not in key}
