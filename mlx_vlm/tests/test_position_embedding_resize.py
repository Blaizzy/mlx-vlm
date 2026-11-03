"""Resized position embeddings must antialias with PyTorch's triangle filter.

Both functions below are ports of Siglip2's ``resize_positional_embeddings``,
which resizes with ``F.interpolate(mode="bilinear", antialias=True)``. The
legacy ``resize_bilinear`` helper antialiases with a Gaussian blur instead, so
downscaling silently shifts every position embedding.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.models.aya_vision.vision import VisionEmbeddings as AyaVisionEmbeddings
from mlx_vlm.models.phi4mm.config import VisionConfig as Phi4mmVisionConfig
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
@pytest.mark.parametrize("training", [False, True])
def test_phi4mm_matches_aten(size, training):
    pe = _position_embeddings(32, 64)
    out = Phi4mmEmbeddings.resize_positional_embeddings(
        mx.array(pe), mx.array([list(size)]), size[0] * size[1], training=training
    )
    np.testing.assert_allclose(
        np.array(out[0]), _aten_bilinear_aa(pe, size).reshape(-1, 64), atol=1e-5
    )


@pytest.mark.parametrize("size", [(16, 24), (12, 12), (30, 20)])
@pytest.mark.parametrize("training", [False, True])
def test_aya_vision_matches_aten(size, training):
    pe = _position_embeddings(27, 64)
    n = size[0] * size[1]
    out = AyaVisionEmbeddings.resize_positional_embeddings(
        mx.array(pe), mx.array([list(size)]), n + 8, training=training
    )
    np.testing.assert_allclose(
        np.array(out[0, :n]), _aten_bilinear_aa(pe, size).reshape(-1, 64), atol=1e-5
    )
    # Tails are padded with the first embedding, as before.
    pad = np.broadcast_to(np.array(out[0, 0]), np.array(out[0, n:]).shape)
    np.testing.assert_allclose(np.array(out[0, n:]), pad, atol=1e-6)


@pytest.mark.parametrize("embeddings", [AyaVisionEmbeddings, Phi4mmEmbeddings])
@pytest.mark.parametrize("size", [(2, 3), (6, 5)])
def test_position_embedding_resize_backward(embeddings, size):
    pe = _position_embeddings(4, 3)
    max_length = size[0] * size[1] + 3

    def loss(positional_embeddings):
        out = embeddings.resize_positional_embeddings(
            positional_embeddings,
            mx.array([size]),
            max_length,
            training=True,
        )
        return mx.sum(out.square())

    value, grad = mx.value_and_grad(loss)(mx.array(pe))
    mx.eval(value, grad)

    # Resizing a basis gives the independent ATen interpolation matrix. Include
    # the repeated first row so the reference also checks gradients from padding.
    basis = np.eye(16, dtype=np.float32).reshape(4, 4, 16)
    weights = _aten_bilinear_aa(basis, size).reshape(-1, 16)
    weights = np.concatenate([weights, np.repeat(weights[:1], 3, axis=0)])
    reference = weights @ pe.reshape(16, 3)
    expected_grad = (2 * weights.T @ reference).reshape(pe.shape)
    np.testing.assert_allclose(value.item(), np.sum(reference**2), rtol=1e-5)
    np.testing.assert_allclose(np.array(grad), expected_grad, rtol=1e-5, atol=1e-5)


def test_phi4mm_training_mode_supports_position_embedding_gradients():
    config = Phi4mmVisionConfig(
        image_size=8, patch_size=2, hidden_size=3, num_attention_heads=1
    )
    embeddings = Phi4mmEmbeddings(config)
    pixels = mx.ones((1, 6, 12))
    shapes = mx.array([[2, 3]])

    embeddings.eval()
    inference_output = embeddings(pixels, spatial_shapes=shapes)
    mx.eval(inference_output)
    embeddings.train()

    def loss(model):
        return mx.mean(model(pixels, spatial_shapes=shapes).square())

    value, grads = nn.value_and_grad(embeddings, loss)(embeddings)
    grad = grads["position_embedding"]["weight"]
    mx.eval(value, grad)
    assert mx.all(mx.isfinite(grad)).item()
    assert mx.any(grad != 0).item()
    np.testing.assert_allclose(
        np.array(embeddings(pixels, spatial_shapes=shapes)),
        np.array(inference_output),
        atol=1e-5,
    )
