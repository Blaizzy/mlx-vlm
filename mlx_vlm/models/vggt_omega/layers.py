"""Transformer blocks for the VGGT-Omega encoder, aggregator and heads.

The blocks reuse the DINOv2 modules, which have the same checkpoint names.
Matmuls and attention run in the weight dtype; the residual stream and all
norms stay in float32, as under the reference CUDA autocast.
"""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dinov2.dinov2 import Attention as DINOv2Attention
from ..dinov2.dinov2 import LayerScale
from ..dinov2.dinov2 import Mlp as DINOv2Mlp
from ..sapiens2.backbone import _rope_apply
from .kernels import add_layer_norm, can_add_layer_norm, can_prepare_qk, prepare_qk

Rope = Optional[Tuple[mx.array, mx.array]]


def norm32(norm: nn.LayerNorm, x: mx.array) -> mx.array:
    """Apply ``norm`` in float32 (bf16 layer norms drift over deep stacks)."""
    return mx.fast.layer_norm(x.astype(mx.float32), norm.weight, norm.bias, norm.eps)


class Attention(DINOv2Attention):
    """DINOv2 attention with optional Q/K LayerNorm and packed 2D RoPE.

    The checkpoint's ``bias_mask`` is folded into ``qkv.bias`` on load.
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool, eps: float):
        super().__init__(dim, num_heads)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)

    def __call__(self, x: mx.array, rope: Rope = None) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        norms = (self.q_norm, self.k_norm) if "q_norm" in self else None
        if (norms or rope) and can_prepare_qk(self.head_dim):
            q, k = prepare_qk(qkv, norms, rope, v.dtype)
        else:
            q = qkv[:, :, 0].transpose(0, 2, 1, 3)
            k = qkv[:, :, 1].transpose(0, 2, 1, 3)
            if norms:
                q, k = norm32(self.q_norm, q), norm32(self.k_norm, k)
            if rope is not None:
                q, k = _rope_apply(q, *rope), _rope_apply(k, *rope)
            q, k = q.astype(v.dtype), k.astype(v.dtype)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(x.transpose(0, 2, 1, 3).reshape(B, N, C))


@mx.compile
def _gelu(x: mx.array) -> mx.array:
    """GELU computed in float32, rounded once to the input dtype. In bf16
    math, ``1 + erf`` cancels for negative inputs (45% off at x = -3), and
    the error compounds over the blocks."""
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


class Mlp(DINOv2Mlp):
    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm block with LayerScale, in DINOv2 checkpoint names."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, qk_norm, eps)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array, rope: Rope = None) -> mx.array:
        """x: (B, N, C) float32 residual stream."""
        dtype = self.attn.qkv.weight.dtype
        if not can_add_layer_norm(x):
            x = x + self.ls1(self.attn(norm32(self.norm1, x).astype(dtype), rope))
            return x + self.ls2(self.mlp(norm32(self.norm2, x).astype(dtype)))
        h = self.attn(add_layer_norm(x, self.norm1, dtype), rope)
        x, y = add_layer_norm(x, self.norm2, dtype, h, self.ls1.gamma)
        return _scale_add(x, self.mlp(y), self.ls2.gamma)


@mx.compile
def _scale_add(x: mx.array, h: mx.array, gamma: mx.array) -> mx.array:
    """LayerScale residual update in float32."""
    return x + h.astype(mx.float32) * gamma
