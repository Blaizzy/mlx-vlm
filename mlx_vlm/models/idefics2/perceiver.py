import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .language import MLP, TextConfig
from .vision import VisionConfig


@dataclass
class PercieverConfig:
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_hidden_size: int
    mm_vision_tower: str
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, args: PercieverConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        kv: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        hidden_states = mx.concatenate([kv, x], axix=-2)

        queries, keys, values = (
            self.q_proj(x),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config: PercieverConfig):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Idefics2PerceiverAttention(config.text_config)
        self.post_attention_norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = MLP(self.hidden_size, self.hidden_size * 4)

    def __call__(
        self,
        x: mx.array,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(hidden_states)

        latents, cache = self.self_attn(latents, context, mask=mask, cache=cache)

        latents = x + latents
        r = latents

        latents = self.post_attention_norm(latents)
        latents = self.mlp(latents)
        latents = r + latents
        return latents


class Idefics2PerceiverResampler(nn.Module):
    def __init__(self, args: PercieverConfig):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.n_latents = args.n_latents

        self.latents = mx.ones(self.n_latents, self.args)
        self.layers = [Idefics2PerceiverLayer(args) for _ in range(args.depth)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ):

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        h = self.latents
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, x, mask=mask, cache=cache[e])

        return self.norm(h), cache
