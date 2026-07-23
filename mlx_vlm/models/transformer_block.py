"""Config-driven decoder block shared across text backbones.

A single ``TransformerBlock`` with a pluggable token mixer (attention now; MLA
and SSM/linear mixers slot into the same ``self_attn`` interface later), a
switchable gated MLP, a norm-variant selector, and a residual layout selector.
Behaviour-preserving: submodule names match the per-model implementations it
replaces so existing checkpoints load unchanged.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import scaled_dot_product_attention
from .mlp import GatedMLP


class Gemma1pRMSNorm(nn.Module):
    """RMSNorm with the Gemma zero-centered (1 + weight) gamma convention."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def make_norm(kind: str, dims: int, eps: float) -> nn.Module:
    """Build a norm by variant name: rmsnorm | gemma | layernorm."""
    if kind == "gemma":
        return Gemma1pRMSNorm(dims, eps)
    if kind == "layernorm":
        return nn.LayerNorm(dims, eps=eps)
    return nn.RMSNorm(dims, eps=eps)


@dataclass
class BlockSpec:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    scale: float
    intermediate_size: int
    rope: nn.Module
    norm_type: str = "rmsnorm"
    rms_norm_eps: float = 1e-6
    layout: str = "pre"
    attn_bias: bool = False
    qk_norm: bool = False
    attn_logit_softcapping: Optional[float] = None
    mlp_act: Callable = nn.silu
    mlp_bias: bool = False
    use_sliding: bool = False


class Attention(nn.Module):
    """Attention token-mixer: GQA/MHA with optional QK-norm, qkv bias, and an
    optional logit-softcap path (else fused scaled-dot-product attention)."""

    def __init__(self, spec: BlockSpec):
        super().__init__()
        dim = spec.hidden_size
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.scale = spec.scale
        self.softcap = spec.attn_logit_softcapping
        self.rope = spec.rope

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=spec.attn_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=spec.attn_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=spec.attn_bias
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=spec.attn_bias)

        if spec.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=spec.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=spec.rms_norm_eps)
        self.qk_norm = spec.qk_norm

    def _softcap_attention(self, q, k, v, mask):
        B, _, L, _ = q.shape
        q = q * self.scale
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            q = q.reshape(B, self.n_kv_heads, repeats, L, self.head_dim)
            k = mx.expand_dims(k, 2)
            v = mx.expand_dims(v, 2)
        scores = q @ k.swapaxes(-1, -2)
        scores = mx.tanh(scores / self.softcap) * self.softcap
        if mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(
                    mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype)
                )
            else:
                scores = scores + mask
        scores = mx.softmax(scores, precise=True, axis=-1)
        out = scores @ v
        if repeats > 1:
            out = out.reshape(B, self.n_heads, L, self.head_dim)
        return out

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.qk_norm:
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

        if self.softcap is None:
            out = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        else:
            out = self._softcap_attention(queries, keys, values, mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Decoder block with pre-norm or Gemma2 sandwich residual layout."""

    def __init__(self, spec: BlockSpec):
        super().__init__()
        self.hidden_size = spec.hidden_size
        self.num_attention_heads = spec.num_attention_heads
        self.use_sliding = spec.use_sliding
        self.layout = spec.layout

        self.self_attn = Attention(spec)
        self.mlp = GatedMLP(
            spec.hidden_size,
            spec.intermediate_size,
            act=spec.mlp_act,
            bias=spec.mlp_bias,
        )

        n, dim, eps = spec.norm_type, spec.hidden_size, spec.rms_norm_eps
        self.input_layernorm = make_norm(n, dim, eps)
        self.post_attention_layernorm = make_norm(n, dim, eps)
        if spec.layout == "sandwich":
            self.pre_feedforward_layernorm = make_norm(n, dim, eps)
            self.post_feedforward_layernorm = make_norm(n, dim, eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.layout == "sandwich":
            r = self.post_attention_layernorm(
                self.self_attn(self.input_layernorm(x), mask, cache)
            )
            h = x + r
            r = self.post_feedforward_layernorm(
                self.mlp(self.pre_feedforward_layernorm(h))
            )
            return h + r
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
