import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int = 8192
    head_dim: int = 128
    num_hidden_layers: int = 40
    intermediate_size: int = 14336
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    rope_theta: float = 50000.0
    vocab_size: int = 256000
    layer_norm_eps: float = 1e-05
    logit_scale: float = 0.0625
    attention_bias: bool = False
    layer_norm_bias: bool = False
    sliding_window: int = 4096
    sliding_window_pattern: int = 4
    max_position_embeddings: int = 4096

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim
        if (head_dim * n_heads) != dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {dim}"
                f" and `num_heads`: {n_heads})."
            )
        self.scale = head_dim**-0.5

        attetion_bias = config.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attetion_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attetion_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attetion_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=config.rope_theta)

        self.use_sliding_window = (layer_idx + 1) % config.sliding_window_pattern != 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE only if sliding window is enabled
        if self.use_sliding_window:
            if cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)
            else:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if self.use_sliding_window and mask is not None and isinstance(mask, mx.array):
            key_len = keys.shape[-2]
            if mask.shape[-1] != key_len:
                mask = mask[..., -key_len:]

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads

        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias
        )
        self.config = config

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class CohereModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=config.layer_norm_bias
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            mask = create_attention_mask(h, cache[j - 1 : j])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.model_type = config.model_type
        self.model = CohereModel(config)
        self.config = config

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: mx.array = None,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, inputs_embeds, mask, cache)
        out = self.model.embed_tokens.as_linear(out)
        out = out * self.model.config.logit_scale
        return LanguageModelOutput(logits=out)

    def make_cache(self):
        caches = []
        for i in range(self.config.num_hidden_layers):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(KVCache())
            else:
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.model.config.head_dim

    @property
    def n_kv_heads(self):
        return self.model.config.num_key_value_heads
