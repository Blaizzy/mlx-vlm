"""Z1T: attention-free (AFT-conv), DyT-normed, fixed-sparsity decoder LM.

Port of Extropic's `z1t` (github.com/extropic-ai/sparse-transformers) into the
mlx-vlm text-model interface. Each block mixes tokens with a causal depthwise
conv plus a causal cumulative pool (no softmax / no q.k). This has no key/value
tensors, so it does not use `KVCache`; instead each layer keeps a `Z1TCache`
holding its running pool sums and its `ksize-1` conv window, giving O(1)/step
decode that is exact vs the flat forward (window-prepend == same conv;
prior-sum + cumsum(chunk) == same pool).
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


# Cap the transient (B, chunk, out, k) gather so a long prefill stays in memory;
# the gather is split along the sequence axis, which is exact.
_SPARSE_CHUNK_ELEMS = 8_000_000


class SparseLinear(nn.Module):
    """y = sum_k weight * x[indices] + bias. `indices` is a fixed, non-trained
    (out, k) fan-in table stored as an int param so it loads from the checkpoint."""

    def __init__(self, in_features: int, out_features: int, k: int):
        super().__init__()
        self.weight = mx.zeros((out_features, k))
        self.bias = mx.zeros((out_features,))
        self.indices = mx.zeros((out_features, k), dtype=mx.int32)

    def _gather(self, x: mx.array, idx: mx.array) -> mx.array:
        return (mx.take(x, idx, axis=-1) * self.weight).sum(axis=-1) + self.bias

    def __call__(self, x: mx.array) -> mx.array:
        idx = self.indices
        if idx.dtype not in (mx.int32, mx.int64, mx.uint32):
            idx = idx.astype(mx.int32)
        out_features = self.weight.shape[0]
        if x.ndim == 3 and x.shape[1] * out_features > _SPARSE_CHUNK_ELEMS:
            step = max(1, _SPARSE_CHUNK_ELEMS // out_features)
            parts = [
                self._gather(x[:, s : s + step], idx)
                for s in range(0, x.shape[1], step)
            ]
            return mx.concatenate(parts, axis=1)
        return self._gather(x, idx)


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


class Z1TCache:
    """Per-layer streaming state for AFT-conv decode.

    Holds the running pool sums (`cum_eKV`, `cum_eK`) and the last `ksize-1`
    conv-window values (`win_eKV`, `win_eK`). Empty state == zeros == the flat
    forward's left-padding, so a fresh cache reproduces the full-sequence result.
    """

    def __init__(self):
        self.offset = 0
        self.cum_eKV = None
        self.cum_eK = None
        self.win_eKV = None
        self.win_eK = None


class AFTConv(nn.Module):
    """AFT-conv (causal): depthwise conv stencil + causal cumulative pool.

    Threads an optional `Z1TCache`: prepends the prior conv window and continues
    the pool from the prior running sums, so decoding one token is O(1) in the
    context length while matching the flat forward.
    """

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

    def __call__(self, x: mx.array, cache: Optional[Z1TCache] = None) -> mx.array:
        if cache is not None and x.shape[1] == 1:
            return self._decode_step(x, cache)

        B, Ln, dim = x.shape
        h, hd, s = self.heads, dim // self.heads, self.ksize

        qkv = self.qkv_proj(x)
        if self.tanh_linear:
            qkv = mx.tanh(qkv)
        Q = qkv[..., :dim]
        V = qkv[..., dim : 2 * dim]
        K = qkv[..., 2 * dim :]
        Qh = Q.reshape(B, Ln, h, hd)
        Vh = V.reshape(B, Ln, h, hd)

        eK = mx.exp(mx.tanh(K))  # (B, Ln, h)
        kernel = mx.exp(self.w_conv) - 1.0  # (h, s)
        eKV = eK[..., None] * Vh  # (B, Ln, h, hd)

        if cache is not None and cache.cum_eKV is not None:
            cum_eKV, cum_eK = cache.cum_eKV, cache.cum_eK
            win_eKV, win_eK = cache.win_eKV, cache.win_eK
        else:
            cum_eKV = mx.zeros((B, h, hd), x.dtype)
            cum_eK = mx.zeros((B, h), x.dtype)
            win_eKV = mx.zeros((B, s - 1, h, hd), x.dtype)
            win_eK = mx.zeros((B, s - 1, h), x.dtype)

        # depthwise causal conv over [window ++ new]: output[t] = sum_j k_j * ext[t+j]
        eKV_ext = mx.concatenate([win_eKV, eKV], axis=1)  # (B, s-1+Ln, h, hd)
        eK_ext = mx.concatenate([win_eK, eK], axis=1)  # (B, s-1+Ln, h)
        num_conv = mx.zeros((B, Ln, h, hd), x.dtype)
        den_conv = mx.zeros((B, Ln, h), x.dtype)
        for j in range(s):
            num_conv = (
                num_conv + kernel[:, j].reshape(1, 1, h, 1) * eKV_ext[:, j : j + Ln]
            )
            den_conv = den_conv + kernel[:, j].reshape(1, 1, h) * eK_ext[:, j : j + Ln]

        # global pool: prior running sum + inclusive cumsum within the new chunk
        run_eKV = cum_eKV[:, None] + mx.cumsum(eKV, axis=1)
        run_eK = cum_eK[:, None] + mx.cumsum(eK, axis=1)
        num = num_conv + run_eKV
        den = den_conv + run_eK
        ctx = num / (den[..., None] + 1e-6)
        Y = (mx.tanh(Qh) * ctx).reshape(B, Ln, dim)

        out = self.out_proj(Y)
        if self.tanh_linear:
            out = mx.tanh(out)

        if cache is not None:
            cache.cum_eKV = cum_eKV + eKV.sum(axis=1)
            cache.cum_eK = cum_eK + eK.sum(axis=1)
            if s > 1:
                cache.win_eKV = eKV_ext[:, -(s - 1) :]
                cache.win_eK = eK_ext[:, -(s - 1) :]
            cache.offset += Ln
            mx.eval(cache.cum_eKV, cache.cum_eK, cache.win_eKV, cache.win_eK)
        return out

    def _decode_step(self, x: mx.array, cache: Z1TCache) -> mx.array:
        """O(1) single-token decode. Works on time-axis-free (B, h, hd) tensors:
        MLX (0.31) miscompiles fused kernels over a length-1 sequence axis, so
        keeping no size-1 time dim here is load-bearing, not just an optimization.
        """
        B, _, dim = x.shape
        h, hd, s = self.heads, dim // self.heads, self.ksize

        qkv = self.qkv_proj(x[:, 0])
        if self.tanh_linear:
            qkv = mx.tanh(qkv)
        Qh = qkv[:, :dim].reshape(B, h, hd)
        Vh = qkv[:, dim : 2 * dim].reshape(B, h, hd)
        eK = mx.exp(mx.tanh(qkv[:, 2 * dim :]))  # (B, h)
        eKV = eK[..., None] * Vh  # (B, h, hd)
        kernel = mx.exp(self.w_conv) - 1.0  # (h, s)

        if cache.cum_eKV is not None:
            cum_eKV, cum_eK = cache.cum_eKV, cache.cum_eK
            win_eKV, win_eK = cache.win_eKV, cache.win_eK
        else:
            cum_eKV = mx.zeros((B, h, hd), x.dtype)
            cum_eK = mx.zeros((B, h), x.dtype)
            win_eKV = mx.zeros((B, s - 1, h, hd), x.dtype)
            win_eK = mx.zeros((B, s - 1, h), x.dtype)

        taps_eKV = [win_eKV[:, j] for j in range(s - 1)] + [eKV]
        taps_eK = [win_eK[:, j] for j in range(s - 1)] + [eK]
        num = cum_eKV + eKV
        den = cum_eK + eK
        for j in range(s):
            num = num + kernel[:, j].reshape(1, h, 1) * taps_eKV[j]
            den = den + kernel[:, j].reshape(1, h) * taps_eK[j]
        ctx = num / (den[..., None] + 1e-6)
        Y = (mx.tanh(Qh) * ctx).reshape(B, dim)
        out = self.out_proj(Y)
        if self.tanh_linear:
            out = mx.tanh(out)

        cache.cum_eKV = cum_eKV + eKV
        cache.cum_eK = cum_eK + eK
        if s > 1:
            cache.win_eKV = mx.concatenate([win_eKV[:, 1:], eKV[:, None]], axis=1)
            cache.win_eK = mx.concatenate([win_eK[:, 1:], eK[:, None]], axis=1)
        cache.offset += 1
        mx.eval(cache.cum_eKV, cache.cum_eK, cache.win_eKV, cache.win_eK)
        return out[:, None]


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

    def __call__(self, x: mx.array, cache: Optional[Z1TCache] = None) -> mx.array:
        x = x + self.attn(self.norm1(x), cache=cache)
        x = x + self.mlp(self.norm2(x))
        return x


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

    def _pos(self, offset: int, length: int, dtype) -> mx.array:
        pos = self.pe[offset : offset + length]
        if pos.shape[0] < length:  # past the table (out-of-distribution): zero-extend
            pad = mx.zeros((length - pos.shape[0], self.pe.shape[1]), self.pe.dtype)
            pos = mx.concatenate([pos, pad], axis=0)
        return pos.astype(dtype)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> mx.array:
        x = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        offset = cache[0].offset if (cache is not None and cache[0] is not None) else 0
        x = x + self._pos(offset, x.shape[1], x.dtype)

        caches = cache if cache is not None else [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, caches):
            x = layer(x, cache=layer_cache)
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
