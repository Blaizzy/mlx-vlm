from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..activations import swiglu
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import TextConfig


def _rope_cos_sin(position_ids: mx.array, head_dim: int, base: float) -> tuple:
    half = head_dim // 2
    inv_freq = base ** (-mx.arange(0, half, dtype=mx.float32) / half)
    freqs = position_ids[..., None].astype(mx.float32) * inv_freq
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb)[:, None, :, :], mx.sin(emb)[:, None, :, :]


def _apply_rope_with_cos_sin(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return (x * cos) + (rotated * sin)


def build_magi_block_mask(
    kv_len: int, q_len: int, block_size: int, dtype: mx.Dtype = mx.float32
) -> mx.array:
    q_global_start = kv_len - q_len
    window_start_k = kv_len - block_size
    blocked_k = window_start_k - 1

    q_idx = mx.arange(q_len).reshape(q_len, 1)
    k_idx = mx.arange(kv_len).reshape(1, kv_len)
    g_idx = q_idx + q_global_start

    in_window = q_idx >= (q_len - block_size)

    causal = (~in_window) & (k_idx <= g_idx)
    win_to_prefix = in_window & (k_idx < blocked_k)
    win_to_window = in_window & (k_idx >= window_start_k)

    allowed = causal | win_to_prefix | win_to_window
    neg = mx.array(float("-inf"), dtype=dtype)
    mask = mx.where(allowed, mx.array(0.0, dtype=dtype), neg)
    return mask[None, None]


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

        self.rope = nn.RoPE(
            head_dim, traditional=args.rope_traditional, base=args.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        head_dim = D // self.n_heads
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if position_ids is not None:
            cos, sin = _rope_cos_sin(position_ids, head_dim, self.rope.base)
            queries = _apply_rope_with_cos_sin(queries, cos, sin)
            keys = _apply_rope_with_cos_sin(keys, cos, sin)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
        elif cache is not None:
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


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen2Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen2DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
    ):
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, position_ids)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Qwen2Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        **kwargs,
    ):
        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
