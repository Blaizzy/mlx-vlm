from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..mlp import SwiGLUMLP as MLP
from .config import ModelConfig


class LayerNorm2D(nn.Module):

    def __init__(self, d1, d2, eps):
        super().__init__()
        self.weight = mx.zeros((d1, d2))
        self.eps = eps

    def __call__(self, x):
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // args.num_attention_heads
        self.scale = head_dim**-0.5

        attetion_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attetion_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attetion_bias)

        self.use_qk_norm = args.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = LayerNorm2D(self.n_heads, head_dim, eps=args.layer_norm_eps)
            self.k_norm = LayerNorm2D(
                self.n_kv_heads, head_dim, eps=args.layer_norm_eps
            )

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.use_qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads

        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class CohereModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
        )

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.model_type = args.model_type
        self.model = CohereModel(args)
        self.args = args

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        out = out * self.model.args.logit_scale
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.layers]
