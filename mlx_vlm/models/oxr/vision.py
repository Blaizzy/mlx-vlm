"""SigLIP2-NaViT vision encoder used by SoMark OXR (Ovis2.5-style).

Differences from a plain SigLIP ViT:
* native-resolution patches with 2D rotary embeddings (half rotation, fp32);
* the 32x32 learned position embeddings are bicubically resized per image;
* patches are ordered in 2x2 merge blocks (``hidden_stride``) so that four
  consecutive tokens form one visual atom after the encoder;
* full attention per image (the checkpoint has no full/window layer split);
* no post-layernorm: OXR consumes the last encoder layer directly.
"""

from functools import lru_cache
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def _cubic_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    def near(x):  # |x| <= 1
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def far(x):  # 1 < |x| < 2
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], axis=-1)


@lru_cache(maxsize=64)
def bicubic_matrix(in_size: int, out_size: int) -> np.ndarray:
    """(out, in) matrix equal to torch F.interpolate(mode="bicubic", align_corners=False)."""
    scale = in_size / out_size
    src = (np.arange(out_size) + 0.5) * scale - 0.5
    x0 = np.floor(src).astype(np.int64)
    w = _cubic_weights(src - x0)
    m = np.zeros((out_size, in_size), dtype=np.float64)
    for k in range(4):
        idx = np.clip(x0 - 1 + k, 0, in_size - 1)
        np.add.at(m, (np.arange(out_size), idx), w[:, k])
    return m.astype(np.float32)


def _rotate_half(x: mx.array) -> mx.array:
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    xf = x.astype(mx.float32)
    out = xf * cos[:, :, None, :] + _rotate_half(xf) * sin[:, :, None, :]
    return out.astype(x.dtype)


class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, L, D = x.shape
        shape = (B, L, self.num_heads, self.head_dim)
        q = apply_rotary(self.q_proj(x).reshape(shape), cos, sin)
        k = apply_rotary(self.k_proj(x).reshape(shape), cos, sin)
        v = self.v_proj(x).reshape(shape)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Attention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), cos, sin)
        return x + self.mlp(self.layer_norm2(x))


class PatchEmbedding(nn.Module):
    """Conv2d(kernel=stride=patch) on pre-flattened (C, p, p) patches == a Linear."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        in_dim = config.num_channels * config.temporal_patch_size * config.patch_size**2
        self.weight = mx.zeros((config.hidden_size, in_dim))
        self.bias = mx.zeros((config.hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.addmm(self.bias, x.astype(self.weight.dtype), self.weight.T)


class Embeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(config)
        self.grid_side = config.image_size // config.patch_size
        self.position_embedding = nn.Embedding(self.grid_side**2, config.hidden_size)

    def position_embeddings(self, t: int, h: int, w: int) -> mx.array:
        s = self.config.hidden_stride
        side = self.grid_side
        weight = self.position_embedding.weight
        pe = weight.astype(mx.float32).reshape(side, side, -1)
        mh = mx.array(bicubic_matrix(side, h))
        mw = mx.array(bicubic_matrix(side, w))
        pe = mx.einsum("hi,ijc,wj->hwc", mh, pe, mw).astype(weight.dtype)
        pe = pe.reshape(h // s, s, w // s, s, -1).transpose(0, 2, 1, 3, 4)
        pe = pe.reshape(h * w, -1)
        return mx.tile(pe, (t, 1)) if t > 1 else pe

    def __call__(self, pixel_values: mx.array, grid) -> mx.array:
        return self.patch_embedding(pixel_values) + self.position_embeddings(*grid)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.embeddings = Embeddings(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        rope_dim = config.hidden_size // config.num_attention_heads // 2
        self._inv_freq = 1.0 / (
            10000.0 ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim)
        )

    def rotary(self, t: int, h: int, w: int):
        s = self.config.hidden_stride
        hpos = np.broadcast_to(np.arange(h)[:, None], (h, w))
        wpos = np.broadcast_to(np.arange(w)[None, :], (h, w))

        def merge_order(p):
            return p.reshape(h // s, s, w // s, s).transpose(0, 2, 1, 3).reshape(-1)

        freqs = np.concatenate(
            [
                np.outer(merge_order(hpos), self._inv_freq),
                np.outer(merge_order(wpos), self._inv_freq),
            ],
            axis=-1,
        ).astype(np.float32)
        if t > 1:
            freqs = np.tile(freqs, (t, 1))
        emb = np.concatenate([freqs, freqs], axis=-1)
        return mx.array(np.cos(emb))[None], mx.array(np.sin(emb))[None]

    def encode(self, pixel_values: mx.array, grid) -> mx.array:
        x = self.embeddings(pixel_values, grid)[None]
        cos, sin = self.rotary(*grid)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x[0]

    def __call__(
        self,
        pixel_values: mx.array,
        grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """pixel_values: (sum_i t*h*w, C*p*p) patches of one or more images in
        merge-block order; grid_thw: (num_images, 3). Returns (num_patches, D)."""
        if grid_thw is None:
            raise ValueError("grid_thw is required for the OXR vision encoder")
        grids = [tuple(int(v) for v in g) for g in np.array(grid_thw).reshape(-1, 3)]
        outputs, start = [], 0
        for g in grids:
            n = g[0] * g[1] * g[2]
            outputs.append(self.encode(pixel_values[start : start + n], g))
            start += n
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=0)
