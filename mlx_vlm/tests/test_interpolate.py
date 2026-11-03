"""Position embeddings must resize spatially and match model references."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.base import interpolate
from mlx_vlm.models.molmo.config import VisionConfig
from mlx_vlm.models.molmo.vision import VisionTransformer


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


@pytest.mark.parametrize(
    "expected",
    [
        pytest.param(
            [[0.3233386, 0.3941329], [0.8250461, 0.8958404]],
            id="downsample",
        ),
        pytest.param(
            [
                [-0.0701176, 0.5642415, 1.0007006, 0.1482393, -0.1323528, -0.1402351],
                [-0.2229101, 0.1868512, 0.4942810, 0.0106209, 0.5294119, 1.1284832],
                [0.4256827, 0.1816451, -0.1049709, -0.1641271, 0.9715413, 2.0014017],
                [2.9982941, 1.4964598, -0.2072083, -0.2663639, 0.0178377, 0.2964788],
                [1.6927243, 0.7162631, 0.1888619, 1.9737202, 1.0588237, -0.2647061],
                [-0.2103535, -0.3552634, 0.5739225, 3.9989953, 2.2569664, -0.2804714],
            ],
            id="upsample",
        ),
    ],
)
def test_molmo_position_embeddings_match_antialiased_bicubic(expected):
    # Frozen PyTorch F.interpolate(mode="bicubic", align_corners=False,
    # antialias=True) outputs avoid adding a PyTorch dependency to the tests.
    pattern = mx.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
        ]
    )
    cls_embedding = mx.array([[42.0, -7.0]])
    model = VisionTransformer(
        VisionConfig(image_emb_dim=2, image_num_pos=17, image_num_layers=0)
    )
    model.positional_embedding = mx.concatenate(
        [cls_embedding, mx.stack([pattern, pattern.T], axis=-1).reshape(-1, 2)]
    )
    expected = mx.array(expected)
    side = expected.shape[0]
    x = mx.full((2, side * side + 1, 2), 0.25)

    result = model.add_pos_emb(x, (side, side))

    expected_pos = mx.stack([expected, expected.T], axis=-1).reshape(-1, 2)
    expected_embedding = mx.concatenate([cls_embedding, expected_pos])
    assert mx.allclose(result, x + expected_embedding[None], atol=1e-6)
