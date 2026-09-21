"""Dense inference layers; all tensors and computation use MLX."""

import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn


class Sequential(nn.Module):
    """Keep the upstream numbered state-dict names."""

    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            self[str(i)] = layer

    def __call__(self, x):
        for layer in self.values():
            x = layer(x)
        return x

    def __getitem__(self, key):
        return super().__getitem__(str(key) if isinstance(key, int) else key)

    def __setitem__(self, key, value):
        super().__setitem__(str(key) if isinstance(key, int) else key, value)


def layer_norm(x, eps=1e-5):
    return mx.fast.layer_norm(x, None, None, eps)


@partial(mx.compile, shapeless=True)
def modulate(x, scale, shift):
    return x * (1 + scale) + shift


@partial(mx.compile, shapeless=True)
def gated_residual(x, h, gate):
    return x + h * gate


@mx.compile
def _head_norm(x, gamma):
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.maximum(mx.sum(xf * xf, axis=-1, keepdims=True), 1e-24))
    return (y * gamma * math.sqrt(x.shape[-1])).astype(x.dtype)


class LayerNorm(nn.LayerNorm):
    def __call__(self, x):
        y = super().__call__(x.astype(mx.float32))
        return y.astype(x.dtype)


def position_embedding(coords, channels):
    dim = channels // 6
    freq = mx.exp(-math.log(10000) * mx.arange(dim, dtype=mx.float32) / dim)
    phase = coords.astype(mx.float32)[..., None] * freq
    out = mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-1)
    out = out.reshape(coords.shape[0], -1)
    return mx.pad(out, [(0, 0), (0, channels - out.shape[-1])])


class TimestepEmbedder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp = Sequential(
            nn.Linear(256, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def __call__(self, t):
        phase = t.reshape(-1, 1).astype(mx.float32) * mx.exp(
            -math.log(10000) * mx.arange(128, dtype=mx.float32) / 128
        )
        x = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
        return self.mlp(x.astype(self.mlp["0"].weight.dtype))


class FeedForward(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.mlp = Sequential(
            nn.Linear(channels, int(channels * ratio)),
            nn.GELU(approx="tanh"),
            nn.Linear(int(channels * ratio), channels),
        )

    def __call__(self, x):
        return self.mlp(x)


class HeadNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x):
        return _head_norm(x, self.gamma)


def attend(q, k, v, mask=None):
    y = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        scale=q.shape[-1] ** -0.5,
        mask=mask,
    )
    return y.transpose(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], -1)


class Attention(nn.Module):
    def __init__(self, channels, heads, context_channels=None, qk_norm=False):
        super().__init__()
        self.heads = heads
        if context_channels is None:
            self.to_qkv = nn.Linear(channels, channels * 3)
        else:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(context_channels, channels * 2)
        if qk_norm:
            self.q_rms_norm = HeadNorm(heads, channels // heads)
            self.k_rms_norm = HeadNorm(heads, channels // heads)
        self.to_out = nn.Linear(channels, channels)

    def prepare(self, context):
        channels = self.to_out.weight.shape[1]
        if context is None:
            value = (
                self.to_kv.bias[channels:]
                if "bias" in self.to_kv
                else mx.zeros(channels, self.to_kv.weight.dtype)
            )
            return (self.to_out(value[None, None]),)
        kv = self.to_kv(context).reshape(*context.shape[:2], 2, self.heads, -1)
        return kv[:, :, 0], kv[:, :, 1]

    def __call__(self, x, context=None, mask=None):
        b, n, c = x.shape
        if context is None:
            qkv = self.to_qkv(x).reshape(b, n, 3, self.heads, -1)
            q, k, v = (qkv[:, :, i] for i in range(3))
        else:
            if isinstance(context, tuple) and len(context) == 1:
                return mx.broadcast_to(context[0], x.shape)
            q = self.to_q(x).reshape(b, n, self.heads, -1)
            k, v = context if isinstance(context, tuple) else self.prepare(context)
        if "q_rms_norm" in self:
            q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        return self.to_out(attend(q, k, v, mask))


class MOTAttention(nn.Module):
    def __init__(self, channels, heads, names):
        super().__init__()
        self.heads = heads
        self.to_qkv = {n: nn.Linear(channels, 3 * channels) for n in names}
        self.to_out = {n: nn.Linear(channels, channels) for n in names}
        self.q_rms_norm = {n: HeadNorm(heads, channels // heads) for n in names}
        self.k_rms_norm = {n: HeadNorm(heads, channels // heads) for n in names}

    def __call__(self, x):
        q, k, v = {}, {}, {}
        for name, value in x.items():
            qkv = self.to_qkv[name](value).reshape(*value.shape[:2], 3, self.heads, -1)
            q[name] = self.q_rms_norm[name](qkv[:, :, 0])
            k[name] = self.k_rms_norm[name](qkv[:, :, 1])
            v[name] = qkv[:, :, 2]
        out = {
            "shape": self.to_out["shape"](attend(q["shape"], k["shape"], v["shape"]))
        }
        others = [n for n in x if n != "shape"]
        keys = mx.concatenate([k[n] for n in others] + [k["shape"]], axis=1)
        vals = mx.concatenate([v[n] for n in others] + [v["shape"]], axis=1)
        for n in others:
            out[n] = self.to_out[n](attend(q[n], keys, vals))
        return out


class ModulatedBlock(nn.Module):
    def __init__(self, channels, heads, context_channels, names=None):
        super().__init__()
        self.adaLN_modulation = Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels))
        if names is None:
            self.norm2 = LayerNorm(channels, eps=1e-6)
            self.self_attn = Attention(channels, heads, qk_norm=True)
            self.cross_attn = Attention(channels, heads, context_channels)
            self.mlp = FeedForward(channels)
        else:
            self.norm2 = {n: LayerNorm(channels, eps=1e-6) for n in names}
            self.self_attn = MOTAttention(channels, heads, names)
            self.cross_attn = {
                n: Attention(channels, heads, context_channels) for n in names
            }
            self.mlp = {n: FeedForward(channels) for n in names}

    @staticmethod
    def _cross(attn, norm, x, context):
        if isinstance(context, tuple) and len(context) == 1:
            return x + context[0]
        return x + attn(norm(x), context)

    def __call__(self, x, time, context):
        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = [
            v[:, None] for v in mx.split(self.adaLN_modulation(time), 6, axis=-1)
        ]
        if isinstance(x, dict):
            contexts = (
                context if isinstance(context, dict) else dict.fromkeys(x, context)
            )
            h = self.self_attn(
                {n: modulate(layer_norm(v, 1e-6), scale, shift) for n, v in x.items()}
            )
            x = {n: gated_residual(v, h[n], gate) for n, v in x.items()}
            x = {
                n: self._cross(self.cross_attn[n], self.norm2[n], v, contexts[n])
                for n, v in x.items()
            }
            return {
                n: gated_residual(
                    v,
                    self.mlp[n](modulate(layer_norm(v, 1e-6), scale_mlp, shift_mlp)),
                    gate_mlp,
                )
                for n, v in x.items()
            }
        x = gated_residual(
            x, self.self_attn(modulate(layer_norm(x, 1e-6), scale, shift)), gate
        )
        x = self._cross(self.cross_attn, self.norm2, x, context)
        return gated_residual(
            x, self.mlp(modulate(layer_norm(x, 1e-6), scale_mlp, shift_mlp)), gate_mlp
        )
