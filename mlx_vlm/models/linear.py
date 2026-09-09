"""Projection helpers shared by model families."""

import mlx.core as mx


def tiled_linear(linear, x):
    """Give large prefill projections a fixed reduction geometry.

    Explicit calls prevent matmul from folding the tile axis into its row
    dimension. The final tile is padded so it uses the same reduction too.
    Small calls retain the model's native geometry, including draft replay.
    """
    tile_size = 256
    if x.ndim != 3 or x.shape[1] <= tile_size:
        return linear(x)
    length = x.shape[1]
    padding = (-length) % tile_size
    if padding:
        x = mx.pad(x, [(0, 0), (0, padding), (0, 0)])
    return mx.concatenate(
        [
            linear(mx.contiguous(x[:, start : start + tile_size]))
            for start in range(0, length, tile_size)
        ],
        axis=1,
    )[:, :length]
