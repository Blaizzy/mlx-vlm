"""Z1T: attention-free (AFT-conv), DyT-normed, fixed-sparsity decoder LM.

Port of Extropic's `z1t` (github.com/extropic-ai/sparse-transformers) into the
mlx-vlm text-model interface. The block mixes tokens with a causal depthwise
conv plus a causal cumulative pool (no softmax / no q.k), so it does not fit the
key/value `KVCache`; `Z1TCache` instead accumulates the embedded prefix and the
backbone is re-run over it each step (exact, O(L) with L <= max_position).
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from .config import ModelConfig


def _sparse_count(fan_in: Optional[int], dim: int) -> int:
    if fan_in is None:
        return dim
    return max(1, min(fan_in, dim))


class SparseLinear(nn.Module):
    """y = sum_k weight * x[indices] + bias. `indices` is a fixed, non-trained
    (out, k) fan-in table stored as an int param so it loads from the checkpoint."""

    def __init__(self, in_features: int, out_features: int, k: int):
        super().__init__()
        self.weight = mx.zeros((out_features, k))
        self.bias = mx.zeros((out_features,))
        self.indices = mx.zeros((out_features, k), dtype=mx.int32)

    def __call__(self, x: mx.array) -> mx.array:
        idx = self.indices
        if idx.dtype not in (mx.int32, mx.int64, mx.uint32):
            idx = idx.astype(mx.int32)
        g = mx.take(x, idx, axis=-1)
        return (g * self.weight).sum(axis=-1) + self.bias


def _make_linear(in_features: int, out_features: int, fan_in: Optional[int]):
    k = _sparse_count(fan_in, in_features)
    if k >= in_features:
        return nn.Linear(in_features, out_features)
    return SparseLinear(in_features, out_features, k)


class DyT(nn.Module):
    """Dynamic Tanh: weight * tanh(alpha * x) + bias over the last axis."""

    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = mx.array([alpha_init])
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * mx.tanh(self.alpha * x) + self.bias


def _shift_time(x: mx.array, d: int) -> mx.array:
    """Shift along the time axis (1) by d, zero-filling the front (causal)."""
    if d == 0:
        return x
    L = x.shape[1]
    if d >= L:
        return mx.zeros_like(x)
    pad = mx.zeros((x.shape[0], d, *x.shape[2:]), dtype=x.dtype)
    return mx.concatenate([pad, x[:, : L - d]], axis=1)


class AFTConv(nn.Module):
    """AFT-conv (causal): depthwise conv stencil + causal cumulative pool."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim, heads = config.hidden_size, config.aft_heads
        assert dim % heads == 0, "hidden_size must be divisible by aft_heads"
        self.dim = dim
        self.heads = heads
        self.ksize = config.aft_ksize
        self.tanh_linear = config.tanh_linear
        fan = config.fan_in("attn")
        self.qkv_proj = _make_linear(dim, 2 * dim + heads, fan)
        self.out_proj = _make_linear(dim, dim, fan)
        self.w_conv = mx.zeros((heads, self.ksize))

    def __call__(self, x: mx.array) -> mx.array:
        B, L, dim = x.shape
        h, hd, s = self.heads, dim // self.heads, self.ksize

        qkv = self.qkv_proj(x)
        if self.tanh_linear:
            qkv = mx.tanh(qkv)
        Q = qkv[..., :dim]
        V = qkv[..., dim : 2 * dim]
        K = qkv[..., 2 * dim :]
        Qh = Q.reshape(B, L, h, hd)
        Vh = V.reshape(B, L, h, hd)

        eK = mx.exp(mx.tanh(K))  # (B, L, h)
        kernel = mx.exp(self.w_conv) - 1.0  # (h, s)
        eKV = eK[..., None] * Vh  # (B, L, h, hd)

        num_conv = mx.zeros_like(eKV)
        den_conv = mx.zeros_like(eK)
        for j in range(s):
            d = s - 1 - j
            num_conv = num_conv + kernel[:, j].reshape(1, 1, h, 1) * _shift_time(eKV, d)
            den_conv = den_conv + kernel[:, j].reshape(1, 1, h) * _shift_time(eK, d)

        num = num_conv + mx.cumsum(eKV, axis=1)
        den = den_conv + mx.cumsum(eK, axis=1)
        ctx = num / (den[..., None] + 1e-6)
        Y = (mx.tanh(Qh) * ctx).reshape(B, L, dim)

        out = self.out_proj(Y)
        if self.tanh_linear:
            out = mx.tanh(out)
        return out


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.hidden_size
        hidden = int(dim * 4.0)
        fan = config.fan_in("mlp")
        self.proj1 = _make_linear(dim, hidden, fan)
        self.proj2 = _make_linear(hidden, dim, fan)
        self.tanh_linear = config.tanh_linear
        self.tanh_mlp = config.tanh_mlp

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj1(x)
        if self.tanh_linear:
            h = mx.tanh(h)
        h = mx.tanh(h) if self.tanh_mlp else nn.silu(h)
        o = self.proj2(h)
        if self.tanh_linear:
            o = mx.tanh(o)
        return o


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = DyT(config.hidden_size, config.dyt_alpha)
        self.attn = AFTConv(config)
        self.norm2 = DyT(config.hidden_size, config.dyt_alpha)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Z1TCache:
    """Accumulates the embedded prefix; the backbone re-runs over it each step."""

    def __init__(self):
        self.offset = 0
        self._embeds = None

    def update(self, new_h: mx.array) -> mx.array:
        self._embeds = (
            new_h
            if self._embeds is None
            else mx.concatenate([self._embeds, new_h], axis=1)
        )
        mx.eval(self._embeds)  # realize so the decode graph doesn't grow each step
        self.offset = self._embeds.shape[1]
        return self._embeds


def _build_pe(dim: int, max_seq: int, omega: float = 1e4) -> mx.array:
    d = dim // 2
    freq = mx.exp(-math.log(omega) * mx.arange(d) / d)
    angle = mx.arange(max_seq)[:, None] * freq[None, :]
    pe = mx.stack([mx.sin(angle), mx.cos(angle)], axis=-1)  # (max_seq, d, 2)
    return pe.reshape(max_seq, dim)


class Z1TModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = DyT(config.hidden_size, config.dyt_alpha)
        # additive positional table, loaded from the checkpoint (Z1T-0 ships it ~0)
        self.pe = _build_pe(config.hidden_size, config.max_position_embeddings)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> mx.array:
        new_h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is not None and cache[0] is not None:
            full_h = cache[0].update(new_h)
        else:
            full_h = new_h
        L_new, L_full = new_h.shape[1], full_h.shape[1]

        x = full_h + self.pe[:L_full].astype(full_h.dtype)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -L_new:, :]
        return self.norm(x)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Z1TModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [Z1TCache() for _ in range(self.config.num_hidden_layers)]

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache=None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        out = self.model(inputs, inputs_embeds=inputs_embeds, cache=cache)
        return LanguageModelOutput(logits=self.lm_head(out))
