"""Resized position embeddings must antialias with PyTorch's triangle filter.

Both functions below are ports of Siglip2's ``resize_positional_embeddings``,
which resizes with ``F.interpolate(mode="bilinear", antialias=True)``. The
legacy ``resize_bilinear`` helper antialiases with a Gaussian blur instead, so
downscaling silently shifts every position embedding.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.aya_vision.vision import VisionEmbeddings as AyaVisionEmbeddings
from mlx_vlm.models.phi4mm.vision import VisionEmbeddings as Phi4mmEmbeddings


def _aten_bilinear_aa(pe, size):
    """ATen ``_upsample_bilinear2d_aa`` written out directly, so the test does
    not lean on the repository helper it is supposed to check."""

    def weights(n_in, n_out):
        scale = n_in / n_out
        src = (np.arange(n_out) + 0.5) * scale - 0.5
        if scale <= 1.0:
            src = np.maximum(src, 0.0)
            base = np.floor(src)
            frac = src - base
            idx = np.clip(base[:, None] + np.arange(2)[None, :], 0, n_in - 1)
            return idx.astype(int), np.stack([1.0 - frac, frac], axis=1)
        lo = np.maximum(np.floor(src - scale + 1), 0)
        hi = np.minimum(np.floor(src + scale), n_in - 1)
        pos = lo[:, None] + np.arange(int(np.ceil(2 * scale)))[None, :]
        w = np.maximum(0.0, 1.0 - np.abs(pos - src[:, None]) / scale)
        w = np.where(pos <= hi[:, None], w, 0.0)
        return np.minimum(pos, n_in - 1).astype(int), w / w.sum(axis=1, keepdims=True)

    iy, wy = weights(pe.shape[0], size[0])
    ix, wx = weights(pe.shape[1], size[1])
    return np.einsum("hy,wx,hywxc->hwc", wy, wx, pe[iy][:, :, ix, :]).astype(np.float32)


def _position_embeddings(pos_size, dim):
    h = np.linspace(-1, 1, pos_size)[:, None]
    w = np.linspace(-1, 1, pos_size)[None, :]
    field = np.sin(3 * h) * np.cos(2 * w) + 0.5 * np.sin(5 * w + 1.0)
    gain = 0.05 * np.random.default_rng(0).standard_normal((1, 1, dim)) + 1.0
    return (field[:, :, None] * gain).astype(np.float32)


@pytest.mark.parametrize("size", [(16, 24), (16, 16), (10, 10), (40, 30)])
def test_phi4mm_matches_aten(size):
    pe = _position_embeddings(32, 64)
    out = Phi4mmEmbeddings.resize_positional_embeddings(
        mx.array(pe), mx.array([list(size)]), size[0] * size[1]
    )
    np.testing.assert_allclose(
        np.array(out[0]), _aten_bilinear_aa(pe, size).reshape(-1, 64), atol=1e-5
    )


@pytest.mark.parametrize("size", [(16, 24), (12, 12), (30, 20)])
def test_aya_vision_matches_aten(size):
    pe = _position_embeddings(27, 64)
    n = size[0] * size[1]
    out = AyaVisionEmbeddings.resize_positional_embeddings(
        mx.array(pe), mx.array([list(size)]), n + 8
    )
    np.testing.assert_allclose(
        np.array(out[0, :n]), _aten_bilinear_aa(pe, size).reshape(-1, 64), atol=1e-5
    )
    # Tails are padded with the first embedding, as before.
    pad = np.broadcast_to(np.array(out[0, 0]), np.array(out[0, n:]).shape)
    np.testing.assert_allclose(np.array(out[0, n:]), pad, atol=1e-6)
