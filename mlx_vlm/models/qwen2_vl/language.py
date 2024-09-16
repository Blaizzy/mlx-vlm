import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from ..base import KVCache, create_attention_mask


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = 32768
    rope_theta: float = 1000000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            self.rope_scaling = None
            # required_keys = {"factor", "type"}
            # if not all(key in self.rope_scaling for key in required_keys):
            #     raise ValueError(f"rope_scaling must contain keys {required_keys}")

            # if self.rope_scaling["type"] != "linear":
            #     raise ValueError("rope_scaling 'type' currently only supports 'linear'")

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Qwen2VLRotaryEmbedding:
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        rope_type="default",
        config=None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        if config is None:
            print(
                "Warning: `Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in future versions."
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = self._get_rope_init_function()

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, **self.rope_kwargs
        )
        self.inv_freq = mx.array(inv_freq)
        self.original_inv_freq = self.inv_freq

    def _get_rope_init_function(self):
        # Implement the ROPE_INIT_FUNCTIONS dictionary logic here
        # For simplicity, we'll just implement the default RoPE initialization
        def default_rope_init(config, **kwargs):
            dim = kwargs.get("dim", config.hidden_size // config.num_attention_heads)
            base = kwargs.get("base", 10000)
            max_position_embeddings = kwargs.get(
                "max_position_embeddings", config.max_position_embeddings
            )

            inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            return inv_freq, 1.0

        return default_rope_init

    def _dynamic_frequency_update(self, position_ids):
        seq_len = mx.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, seq_len=seq_len, **self.rope_kwargs
            )
            self.inv_freq = mx.array(inv_freq)
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    def __call__(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = mx.expand_dims(
            mx.expand_dims(mx.expand_dims(self.inv_freq, 0), 0), -1
        )
        inv_freq_expanded = np.tile(
            inv_freq_expanded, (3, position_ids.shape[0], inv_freq_expanded.shape[2], 1)
        )
        inv_freq_expanded = mx.array(inv_freq_expanded)
        position_ids = np.tile(position_ids, (3, 1, 1))
        position_ids = mx.array(position_ids)
        position_ids_expanded = mx.expand_dims(
            position_ids, 2
        )  # shape (3, bs, 1, positions)

        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded.astype(mx.float32))
        freqs = mx.transpose(freqs, (0, 1, 3, 2))
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.astype(x.dtype), sin.astype(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (mx.array): The query tensor.
        k (mx.array): The key tensor.
        cos (mx.array): The cosine part of the rotary embedding.
        sin (mx.array): The sine part of the rotary embedding.
        mrope_section (List[int]): Multimodal rope section for channel dimension of temporal, height and width.
        unsqueeze_dim (int, optional): Dimension to unsqueeze. Defaults to 1.

    Returns:
        tuple(mx.array): The rotated query and key tensors.
    """
    mrope_section = mx.array([m * 2 for m in mrope_section])[0]

    # Split and concatenate cos and sin
    cos_split = mx.split(cos, mrope_section, axis=-1)
    sin_split = mx.split(sin, mrope_section, axis=-1)

    print(cos.shape)

    cos = mx.concatenate([m[i % 3] for i, m in enumerate(cos_split)], axis=-1)
    sin = mx.concatenate([m[i % 3] for i, m in enumerate(sin_split)], axis=-1)

    # Unsqueeze
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)

    # Apply rotary embedding
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        # self.rope_scaling = args.rope_scaling

        # self.rotary_emb = Qwen2VLRotaryEmbedding(
        #     head_dim,
        #     max_position_embeddings=args.max_position_embeddings,
        #     base=args.rope_theta,
        #     config=args
        # )
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
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # kv_seq_len = keys.shape[-2]
        # position_ids = mx.arange(0, kv_seq_len, dtype=mx.int32)[None,:]

        # cos, sin = self.rotary_emb(values, position_ids)

        # queries, keys = apply_multimodal_rotary_pos_emb(
        #     queries, keys, cos, sin, position_ids, self.rope_scaling["mrope_section"]
        # )

        # keys, values = cache.update_and_fetch(keys, values)

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
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen2Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Remove unused precomputed rotary freqs
        # return {
        #     k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        # }
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
