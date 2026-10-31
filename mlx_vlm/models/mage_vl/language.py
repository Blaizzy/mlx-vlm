"""Qwen3 decoder with Mage's source RoPE and activation rounding."""

import mlx.core as mx
import mlx.nn as nn

from ..qwen3.language import LanguageModel as Qwen3LanguageModel
from ..qwen3.language import Qwen3Model, TransformerBlock
from .rope import compute_inv_frequencies


class RotaryEmbedding(nn.Module):
    def __init__(self, dims, base):
        super().__init__()
        # Keep the non-persistent source buffer in FP32 during model conversion.
        self._inv_freq = mx.array(compute_inv_frequencies(base, dims))
        mx.eval(self._inv_freq)

    def __call__(self, x, offset=0):
        # Batch caches supply one offset per row, including negative left padding.
        positions = (
            mx.arange(x.shape[-2], dtype=mx.float32)[None, :]
            + mx.array(offset, dtype=mx.float32)[..., None]
        )
        angles = positions[..., None] * self._inv_freq
        angles = mx.concatenate([angles, angles], axis=-1)[:, None]
        cos, sin = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
        left, right = mx.split(x, 2, axis=-1)
        # Torch rounds cos/sin and each product before adding the products.
        return x * cos + mx.concatenate([-right, left], axis=-1) * sin


class MLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj

    def __call__(self, x):
        gate = self.gate_proj(x)
        # The source SiLU kernel computes in FP32 and rounds once to BF16.
        gate = nn.silu(gate.astype(mx.float32)).astype(gate.dtype)
        return self.down_proj(gate * self.up_proj(x))


class LanguageModel(Qwen3LanguageModel):
    def __init__(self, args):
        super().__init__(args)
        scaling = args.rope_scaling or {}
        rope_type = scaling.get("type") or scaling.get("rope_type", "default")
        rope = (
            RotaryEmbedding(args.head_dim, args.rope_theta)
            if rope_type == "default"
            else None
        )
        for layer in self.model.layers:
            if rope is not None:
                layer.self_attn.rope = rope
            layer.mlp = MLP(layer.mlp)


__all__ = ["LanguageModel", "Qwen3Model", "TransformerBlock"]
