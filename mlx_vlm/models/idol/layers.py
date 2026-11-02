"""MLX layers for Idol (adapted from speridlabs/idol-sdk, Apache-2.0)."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def sanitize_weights(model, weights, *, transformer_dtype=mx.float32):
    """Convert original PyTorch keys, convolution layouts, and inference dtypes."""
    names = {key: key for key, _ in tree_flatten(model.parameters())}
    transformer_keys, conv_keys = set(), set()
    # Strip only Sequential's extra component, preserving actual layers lists.
    # Work from the deepest module so nested Sequential paths also match.
    for path, module in sorted(
        model.named_modules(), key=lambda item: len(item[0]), reverse=True
    ):
        if isinstance(module, nn.Sequential):
            prefix = f"{path}.layers." if path else "layers."
            for key in names:
                if key.startswith(prefix):
                    names[key] = names[key].replace(prefix, f"{path}." if path else "")
        if isinstance(module, nn.Conv2d):
            conv_keys.add(f"{path}.weight")
        if isinstance(module, nn.Linear) and path.startswith(
            ("blocks.", "noise_refiner.", "context_refiner.")
        ):
            transformer_keys.update(f"{path}.{name}" for name in ("weight", "bias"))

    names = {source: target for target, source in names.items()}
    sanitized = {}
    for key, value in weights.items():
        if key.endswith(".num_batches_tracked"):
            continue
        key = names.get(key, key)
        if key in conv_keys:
            value = value.transpose(0, 2, 3, 1)
        if mx.issubdtype(value.dtype, mx.floating):
            # Embeddings, norms and learned RePo angles stay in float32.
            dtype = transformer_dtype if key in transformer_keys else mx.float32
            value = value.astype(dtype)
        sanitized[key] = value
    return sanitized


class LayerNorm32(nn.LayerNorm):
    def __call__(self, x):
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = mx.ones((heads, dim))
        self.scale = math.sqrt(dim)

    def __call__(self, x):
        # Upstream uses L2 normalization with a clamped norm, not RMSNorm eps.
        h = x.astype(mx.float32)
        h = h / mx.maximum(mx.sqrt(mx.sum(h * h, axis=-1, keepdims=True)), 1e-12)
        return (h * self.gamma * self.scale).astype(x.dtype)


class FeedForwardNet(nn.Module):
    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approx="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def __call__(self, x):
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, channels, inner_channels, channels_out, mlp_layer_num=2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.extend(
                [
                    nn.Linear(channels if i == 0 else inner_channels, inner_channels),
                    nn.GELU(approx="tanh"),
                ]
            )
        layers.append(nn.Linear(inner_channels if layers else channels, channels_out))
        self.mlp = nn.Sequential(*layers)

    def __call__(self, x):
        return self.mlp(x)


class PcdAbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels, in_channels=3, max_res=16):
        super().__init__()
        self.channels = channels
        dim = channels // in_channels // 2
        extra = max(0, dim - max_res)
        logs = mx.concatenate(
            [
                mx.arange(max_res, dtype=mx.float32),
                mx.arange(extra, dtype=mx.float32) / max(extra, 1) * max_res,
            ]
        )[:dim]
        self._freqs = 2**logs

    def __call__(self, x):
        angles = x[..., None] * self._freqs * (2 * math.pi)
        h = mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)
        h = h.reshape(*x.shape[:-1], -1)
        return mx.pad(h, [(0, 0)] * (h.ndim - 1) + [(0, self.channels - h.shape[-1])])


class PcdAbsolutePositionEmbedderv2(PcdAbsolutePositionEmbedder):
    def __init__(self, channels, in_channels=3, max_res=10):
        super().__init__(channels, in_channels, max_res)
        # Base class multiplies by 2*pi; decoder positions use pi.
        self._freqs = 2 ** mx.linspace(0, max_res, channels // in_channels // 2) / 2


class SinusoidalEmbedder(nn.Module):
    def __init__(
        self, hidden_size, frequency_embedding_size=256, max_period=10000, two_pi=False
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        half = frequency_embedding_size // 2
        self._freqs = mx.exp(
            -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
        )
        self.two_pi = two_pi

    def __call__(self, t):
        angles = t.astype(mx.float32)[:, None] * self._freqs
        if self.two_pi:
            angles = angles * (2 * math.pi)
        return self.mlp(mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1))


def apply_rotary_emb(x, freqs):
    cos, sin = freqs
    h = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    a, b = h[..., 0], h[..., 1]
    return (
        mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1)
        .reshape(x.shape)
        .astype(x.dtype)
    )


class RePo3DRotaryEmbedding(nn.Module):
    def __init__(
        self,
        model_channels,
        num_heads,
        head_dim,
        repo_hidden_ratio=0.125,
        max_freq=16.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, hidden, bias=False)
        self.content_map = nn.Linear(model_channels, hidden, bias=False)
        self.final_map = nn.Linear(hidden, 3 * num_heads, bias=False)
        d = 2 * (head_dim // 6)
        self.freqs_0 = mx.linspace(1, max_freq, d // 2)
        self.freqs_1 = mx.linspace(1, max_freq, d // 2)
        self.freqs_2 = mx.linspace(1, max_freq, (head_dim - 2 * d) // 2)

    def __call__(self, x):
        h = self.norm(x.astype(mx.float32))
        h = self.final_map(nn.silu(self.gate_map(h)) * self.content_map(h))
        pos = h.reshape(*h.shape[:2], self.num_heads, 3)
        angles = (
            mx.concatenate(
                [pos[..., i : i + 1] * getattr(self, f"freqs_{i}") for i in range(3)],
                axis=-1,
            )
            * math.pi
        )
        return mx.cos(angles), mx.sin(angles)


def attention(q, k, v):
    h = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        scale=q.shape[-1] ** -0.5,
    )
    return h.transpose(0, 2, 1, 3)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads, ctx_channels=None, cross=False):
        super().__init__()
        self.num_heads, self.cross = num_heads, cross
        if cross:
            self.to_q = nn.Linear(channels, channels)
            self.to_kv = nn.Linear(ctx_channels, channels * 2)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x, context=None):
        b, n, c = x.shape
        if self.cross:
            q = self.to_q(x).reshape(b, n, self.num_heads, -1)
            kv = self.to_kv(context).reshape(b, context.shape[1], 2, self.num_heads, -1)
            k, v = kv[:, :, 0], kv[:, :, 1]
        else:
            qkv = self.to_qkv(x).reshape(b, n, 3, self.num_heads, -1)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        h = attention(self.q_rms_norm(q), self.k_rms_norm(k), v)
        return self.to_out(h.reshape(b, n, c))


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.q_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.k_norm = MultiHeadRMSNorm(channels // num_heads, num_heads)
        self.out = nn.Linear(channels, channels)

    def __call__(self, x, rope_emb):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.q_norm(apply_rotary_emb(q, rope_emb))
        k = self.k_norm(apply_rotary_emb(k, rope_emb))
        return self.out(attention(q, k, v).reshape(b, n, c))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0, modulation=True):
        super().__init__()
        self.modulation = modulation
        self.norm1 = LayerNorm32(channels, affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(channels, num_heads)
        self.mlp = FeedForwardNet(channels, mlp_ratio)
        if modulation:
            self.shift_table = mx.zeros((1, 6 * channels))

    def __call__(self, x, mod, rotary_emb):
        if not self.modulation:
            x = x + self.attn(self.norm1(x), rotary_emb)
            return x + self.mlp(self.norm2(x))
        parts = mx.split(mod + self.shift_table.astype(mod.dtype), 6, axis=-1)
        shift, scale, gate, shift2, scale2, gate2 = [p[:, None] for p in parts]
        x = x + self.attn(self.norm1(x) * (1 + scale) + shift, rotary_emb) * gate
        return x + self.mlp(self.norm2(x) * (1 + scale2) + shift2) * gate2


class TransformerCrossBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, eps=1e-6)
        self.norm3 = LayerNorm32(channels, affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(channels, num_heads)
        self.cross_attn = MultiHeadAttention(
            channels, num_heads, ctx_channels, cross=True
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio)

    def __call__(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm32(channels, affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, affine=False, eps=1e-6)
        self.cross_attn = MultiHeadAttention(
            channels, num_heads, ctx_channels, cross=True
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio)

    def __call__(self, x, mod, context):
        shift, scale, gate, shift2, scale2, gate2 = [
            p[:, None] for p in mx.split(mod, 6, axis=-1)
        ]
        x = x + self.cross_attn(self.norm1(x) * (1 + scale) + shift, context) * gate
        return x + self.mlp(self.norm2(x) * (1 + scale2) + shift2) * gate2
