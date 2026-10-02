"""Transformer blocks for the image encoder and the multi-view transformer.

The blocks keep the DINOv2 checkpoint names. Matmuls and attention run in the
weight dtype; the residual stream, the norms and the activations run in
float32, as under the reference bf16 autocast.
"""

import mlx.core as mx
import mlx.nn as nn

from ..dinov2.dinov2 import Attention, LayerScale
from ..dinov2.dinov2 import Mlp as DINOv2Mlp
from ..dinov2.dinov2 import SwiGLUFFN as DINOv2SwiGLUFFN


def layer_norm(norm: nn.LayerNorm, x: mx.array, dtype=mx.float32) -> mx.array:
    """``norm`` computed in float32 and written as ``dtype``."""
    y = mx.fast.layer_norm(x.astype(mx.float32), norm.weight, norm.bias, norm.eps)
    return y.astype(dtype)


@mx.compile
def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


@mx.compile
def _swiglu(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return (nn.silu(x1) * x2).astype(x.dtype)


@mx.compile
def _scale_add(x: mx.array, h: mx.array, gamma: mx.array) -> mx.array:
    return x + h.astype(mx.float32) * gamma


class Mlp(DINOv2Mlp):
    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(_gelu(self.fc1(x)))


class SwiGLUFFN(DINOv2SwiGLUFFN):
    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(_swiglu(self.w12(x)))


class Block(nn.Module):
    """Pre-norm ViT block, with LayerScale when ``layer_scale`` is set."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        ffn: str = "mlp",
        layer_scale: bool = True,
        qkv_bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = (
            SwiGLUFFN(dim, hidden) if ffn.startswith("swiglu") else Mlp(dim, hidden)
        )
        if layer_scale:
            self.ls1 = LayerScale(dim)
            self.ls2 = LayerScale(dim)

    def _residual(self, x: mx.array, h: mx.array, name: str) -> mx.array:
        if name in self:
            return _scale_add(x, h, self[name].gamma)
        return x + h.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, N, C) float32 residual stream."""
        dtype = self.attn.qkv.weight.dtype
        x = self._residual(x, self.attn(layer_norm(self.norm1, x, dtype)), "ls1")
        return self._residual(x, self.mlp(layer_norm(self.norm2, x, dtype)), "ls2")
