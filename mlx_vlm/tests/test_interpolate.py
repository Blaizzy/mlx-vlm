"""`base.interpolate` must scale the spatial dimensions of a channels-first tensor."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.base import interpolate


def _reference(x, size, mode="cubic", align_corners=False):
    """The same operation done explicitly in MLX's channels-last layout."""
    h_dst, w_dst = size
    h_src, w_src = x.shape[-2:]
    upsampler = nn.Upsample(
        scale_factor=(h_dst / h_src, w_dst / w_src),
        mode=mode,
        align_corners=align_corners,
    )
    return upsampler(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


def test_keeps_channels_first_shape():
    x = mx.random.normal((1, 64, 8, 8))
    assert interpolate(x, (16, 16)).shape == (1, 64, 16, 16)
    assert interpolate(x, (4, 12)).shape == (1, 64, 4, 12)


def test_three_dimensional_input():
    x = mx.random.normal((64, 8, 8))
    assert interpolate(x, (16, 16)).shape == (64, 16, 16)


def test_matches_channels_last_reference():
    mx.random.seed(0)
    x = mx.random.normal((1, 128, 8, 8))
    for size in [(16, 16), (12, 20), (8, 8)]:
        got = interpolate(x, size)
        assert mx.abs(got - _reference(x, size)).max() < 1e-5


def test_scaling_stays_spatial():
    # Every channel carries the same pattern. Interpolating spatially keeps them identical;
    # scaling the channel dimension instead would tear them apart.
    pattern = mx.arange(8 * 8, dtype=mx.float32).reshape(1, 1, 8, 8)
    x = mx.broadcast_to(pattern, (1, 32, 8, 8))
    out = interpolate(x, (16, 16))
    assert mx.abs(out - out[:, :1, :, :]).max() < 1e-5


def test_bicubic_is_accepted_as_cubic():
    x = mx.random.normal((1, 32, 8, 8))
    with pytest.raises(ValueError):
        nn.Upsample(scale_factor=(2, 2), mode="bicubic")
    assert (
        mx.abs(
            interpolate(x, (16, 16), mode="bicubic") - interpolate(x, (16, 16))
        ).max()
        < 1e-5
    )
