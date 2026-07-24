"""Config-driven decoder block shared across text backbones.

A single ``TransformerBlock`` with a pluggable token mixer (attention now; MLA
and SSM/linear mixers slot into the same ``self_attn`` interface later), a
switchable MLP (dense gated MLP or a config-driven mixture-of-experts), a
norm-variant selector, and a residual layout selector. Behaviour-preserving:
submodule names match the per-model implementations it replaces so existing
checkpoints load unchanged.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import scaled_dot_product_attention
from .mlp import GatedMLP, SwiGLUMLP
from .switch_layers import SwitchGLU


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
class MoESpec:
    """Router + expert configuration for a mixture-of-experts feed-forward.

    Covers the simple top-k router (softmax/sigmoid gate, unconditional
    normalisation) and the DeepSeek-style grouped router (sigmoid gate with an
    additive score-correction bias, group-limited selection, conditional
    normalisation, routed-scaling). Experts and shared experts reuse
    ``switch_layers.SwitchGLU`` and ``mlp.SwiGLUMLP``.
    """

    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    scoring: str = "softmax"
    scoring_precise: bool = False
    select_order: str = "desc"
    use_correction_bias: bool = False
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    norm_guard_topk: bool = False
    norm_denom: str = "max"
    norm_eps: float = 1e-12
    routed_scaling_factor: float = 1.0
    expert_attr: str = "switch_mlp"
    expert_bias: bool = False
    gate_bias: bool = False
    num_shared_experts: int = 0
    shared_intermediate_size: Optional[int] = None
    shared_bias: bool = False


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
    attn_out_bias: Optional[bool] = None
    qk_norm: bool = False
    attn_logit_softcapping: Optional[float] = None
    mlp_act: Callable = nn.silu
    mlp_bias: bool = False
    use_sliding: bool = False
    use_rope: bool = True
    qk_norm_full: bool = False
    qk_norm_post_rope: bool = False
    qk_norm_names: tuple = ("q_norm", "k_norm")
    moe: Optional[MoESpec] = None


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
        self.use_rope = spec.use_rope
        self.rope = spec.rope if spec.use_rope else None

        out_bias = spec.attn_bias if spec.attn_out_bias is None else spec.attn_out_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=spec.attn_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=spec.attn_bias
        )
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=spec.attn_bias
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=out_bias)

        self.qk_norm = spec.qk_norm
        self.qk_norm_full = spec.qk_norm_full
        self.qk_norm_post_rope = spec.qk_norm_post_rope
        self._qname, self._kname = spec.qk_norm_names
        if spec.qk_norm:
            if spec.qk_norm_full:
                qdim, kdim = (
                    self.n_heads * self.head_dim,
                    self.n_kv_heads * self.head_dim,
                )
            else:
                qdim = kdim = self.head_dim
            setattr(self, self._qname, nn.RMSNorm(qdim, eps=spec.rms_norm_eps))
            setattr(self, self._kname, nn.RMSNorm(kdim, eps=spec.rms_norm_eps))

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

        if self.qk_norm and self.qk_norm_full:
            queries = getattr(self, self._qname)(queries)
            keys = getattr(self, self._kname)(keys)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        if self.qk_norm and not self.qk_norm_full and not self.qk_norm_post_rope:
            queries = getattr(self, self._qname)(queries)
            keys = getattr(self, self._kname)(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if self.use_rope:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
            if self.qk_norm and self.qk_norm_post_rope:
                queries = getattr(self, self._qname)(queries)
                keys = getattr(self, self._kname)(keys)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            if self.use_rope:
                queries = self.rope(queries)
                keys = self.rope(keys)
            if self.qk_norm and self.qk_norm_post_rope:
                queries = getattr(self, self._qname)(queries)
                keys = getattr(self, self._kname)(keys)

        if self.softcap is None:
            out = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        else:
            out = self._softcap_attention(queries, keys, values, mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class _MoEGate(nn.Module):
    """Linear router weight (+ optional bias and score-correction bias)."""

    def __init__(self, spec: MoESpec):
        super().__init__()
        self.weight = mx.zeros((spec.num_experts, spec.hidden_size))
        if spec.gate_bias:
            self.bias = mx.zeros((spec.num_experts,))
        if spec.use_correction_bias:
            self.e_score_correction_bias = mx.zeros((spec.num_experts,))

    def __call__(self, x: mx.array) -> mx.array:
        g = x @ self.weight.T
        if "bias" in self:
            g = g + self.bias
        return g


class MoEMLP(nn.Module):
    """Config-driven mixture-of-experts feed-forward.

    Reuses ``SwitchGLU`` for the routed experts and ``SwiGLUMLP`` for the
    optional shared experts; the router matches either the simple top-k or the
    DeepSeek-style grouped selection described on ``MoESpec``.
    """

    def __init__(self, spec: MoESpec):
        super().__init__()
        self.spec = spec
        self.gate = _MoEGate(spec)
        experts = SwitchGLU(
            spec.hidden_size,
            spec.moe_intermediate_size,
            spec.num_experts,
            bias=spec.expert_bias,
        )
        setattr(self, spec.expert_attr, experts)
        if spec.num_shared_experts > 0:
            self.shared_experts = SwiGLUMLP(
                spec.hidden_size,
                spec.shared_intermediate_size,
                spec.shared_bias,
            )
        else:
            self.shared_experts = None

    def _route(self, gates: mx.array):
        spec = self.spec
        if spec.scoring == "sigmoid":
            scores = mx.sigmoid(gates.astype(mx.float32))
        else:
            scores = mx.softmax(
                gates.astype(mx.float32), axis=-1, precise=spec.scoring_precise
            )
        orig = scores

        if spec.use_correction_bias:
            scores = scores + self.gate.e_score_correction_bias

        if spec.n_group > 1:
            scores = mx.unflatten(scores, axis=-1, shape=(spec.n_group, -1))
            group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
            k = spec.n_group - spec.topk_group
            group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
            scores = mx.put_along_axis(
                scores, mx.stop_gradient(group_idx), mx.array(0.0), axis=-2
            )
            scores = mx.flatten(scores, -2, -1)

        k = spec.num_experts_per_tok
        if spec.select_order == "asc":
            inds = mx.stop_gradient(mx.argpartition(scores, kth=-k, axis=-1)[..., -k:])
        else:
            inds = mx.stop_gradient(
                mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            )
        weights = mx.take_along_axis(orig, inds, axis=-1)

        if spec.norm_topk_prob and (not spec.norm_guard_topk or k > 1):
            denom = weights.sum(axis=-1, keepdims=True)
            if spec.norm_denom == "add":
                weights = weights / (denom + spec.norm_eps)
            else:
                weights = weights / mx.maximum(denom, spec.norm_eps)

        weights = weights * spec.routed_scaling_factor
        return inds, weights

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = self._route(self.gate(x))
        experts = getattr(self, self.spec.expert_attr)
        y = experts(x, inds)
        y = (y * weights[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class TransformerBlock(nn.Module):
    """Decoder block with pre-norm or Gemma2 sandwich residual layout."""

    def __init__(self, spec: BlockSpec):
        super().__init__()
        self.hidden_size = spec.hidden_size
        self.num_attention_heads = spec.num_attention_heads
        self.use_sliding = spec.use_sliding
        self.layout = spec.layout

        self.self_attn = Attention(spec)
        if spec.moe is not None:
            self.mlp = MoEMLP(spec.moe)
        else:
            self.mlp = GatedMLP(
                spec.hidden_size,
                spec.intermediate_size,
                act=spec.mlp_act,
                bias=spec.mlp_bias,
            )

        n, dim, eps = spec.norm_type, spec.hidden_size, spec.rms_norm_eps
        if spec.layout == "post":
            self.post_attention_layernorm = make_norm(n, dim, eps)
            self.post_feedforward_layernorm = make_norm(n, dim, eps)
        else:
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
        if self.layout == "post":
            h = x + self.post_attention_layernorm(self.self_attn(x, mask, cache))
            return h + self.post_feedforward_layernorm(self.mlp(h))
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r
