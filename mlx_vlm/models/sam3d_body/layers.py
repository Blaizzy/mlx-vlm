"""Shared layers: SwiGLU FFN, LayerScale, LayerNorm32."""

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.LayerNorm):
    """LayerNorm that upcasts to float32 before computing, matching PyTorch."""

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        return super().__call__(x.astype(mx.float32)).astype(orig_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate(silu) * up -> down."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim)  # gate
        self.w2 = nn.Linear(embed_dim, hidden_dim)  # up
        self.w3 = nn.Linear(hidden_dim, embed_dim)  # down

    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))


class LayerScale(nn.Module):
    """Learnable per-channel scaling."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma
