import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache, LanguageModelOutput, RotatingKVCache, create_attention_mask


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1.0e-6
    vocab_size: int = 262208
    num_key_value_heads: int = 4
    rope_global_base_freq: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    query_pre_attn_scalar: float = 0.0625
    sliding_window: int = 1024
    rope_scaling: Optional[Dict[str, Union[float, List[float]]]] = None
    mm_tokens_per_image: int = 256
    sliding_window_pattern: int = 6

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.repeats = n_heads // n_kv_heads
        self.head_dim = head_dim = config.head_dim
        self.layer_idx = layer_idx

        self.scale = config.query_pre_attn_scalar**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(dims=head_dim, eps=config.rms_norm_eps)
        self.is_sliding = (layer_idx + 1) % config.sliding_window_pattern != 0

        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=(
                config.rope_local_base_freq
                if self.is_sliding
                else config.rope_global_base_freq
            ),
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)

        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Sliding window
        if mask is not None and mask.shape[-1] != keys.shape[-2]:
            mask = mask[..., -keys.shape[-2] :]

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        # This should not be GELU approx, jax.nn.gelu
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        out = h + self.post_feedforward_layernorm(r)
        return out


class Gemma3Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        h *= mx.array(self.config.hidden_size**0.5, mx.bfloat16).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_sliding = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )

            if mask is None and is_sliding:
                mask = sliding_window_mask
            elif mask is None:
                mask = full_mask

            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds=None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, inputs_embeds, mask, cache)
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if "lm_head.weight" not in weights:
            weights["language_model.lm_head.weight"] = weights[
                "language_model.model.embed_tokens.weight"
            ]
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    def make_cache(self):
        caches = []
        for i in range(self.config.num_hidden_layers):
            if (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            ):
                caches.append(
                    KVCache(
                        head_dim=self.config.head_dim,
                        n_kv_heads=self.config.num_key_value_heads,
                    )
                )
            else:
                caches.append(
                    RotatingKVCache(
                        max_size=self.config.sliding_window,
                        keep=0,
                    )
                )
        return caches
