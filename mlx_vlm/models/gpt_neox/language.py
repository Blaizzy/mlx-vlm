from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        assert (
            args.hidden_size % args.num_attention_heads == 0
        ), "hidden_size must be divisible by num_attention_heads"

        self.hidden_size = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.rotary_pct),
            traditional=False,
            base=args.rotary_emb_base,
        )

        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)

        new_qkv_shape = qkv.shape[:-1] + (self.num_attention_heads, 3 * self.head_dim)
        qkv = qkv.reshape(*new_qkv_shape)

        queries, keys, values = [x.transpose(0, 2, 1, 3) for x in qkv.split(3, -1)]

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
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.dense_h_to_4h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * self.hidden_size, self.hidden_size)

    def __call__(self, x) -> mx.array:
        return self.dense_4h_to_h(nn.gelu_approx(self.dense_h_to_4h(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.layer_norm_eps = args.layer_norm_eps
        self.attention = Attention(args)
        self.mlp = MLP(args)
        self.use_parallel_residual = args.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.use_parallel_residual:
            residual = x
            attn = self.attention(self.input_layernorm(x), mask, cache)
            ffn = self.mlp(self.post_attention_layernorm(x))
            out = attn + ffn + residual
            return out
        else:
            attn_output = self.attention(self.input_layernorm(x), mask, cache)
            x = x + attn_output
            ffn_output = self.mlp(self.post_attention_layernorm(x))
            x = x + ffn_output
            return x


class GPTNeoXModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_norm_eps = args.layer_norm_eps
        assert self.vocab_size > 0
        self.embed_in = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_out = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.h = [TransformerBlock(args=args) for _ in range(self.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def __call__(self, inputs: mx.array, cache=None, inputs_embeds=None):
        _, L = inputs.shape

        hidden_states = (
            self.embed_in(inputs) if inputs_embeds is None else inputs_embeds
        )

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        out = self.final_layer_norm(hidden_states)
        out = self.embed_out(out)

        return out


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = GPTNeoXModel(args)

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        new_weights = {}

        for w_key, w_value in weights.items():
            ignore_suffixes = [
                ".attention.bias",
                ".attention.masked_bias",
                ".attention.rotary_emb.inv_freq",
            ]

            skip_weight = False
            for ignored_suffix in ignore_suffixes:
                if w_key.endswith(ignored_suffix):
                    skip_weight = True
                    break

            if skip_weight:
                continue

            if not w_key.startswith("model."):
                w_key = f"model.{w_key}"

            w_key = w_key.replace(".gpt_neox.layers.", ".h.")
            w_key = w_key.replace(".gpt_neox.", ".")

            new_weights[w_key] = w_value

        return new_weights

    @property
    def layers(self):
        return self.model.h

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.layers]
