"""Sparse voxel grids and submanifold convolution in MLX.

Coordinate keys are int64 so grids up to the 1536³ decode resolution stay
collision free.
"""

from itertools import product

import mlx.core as mx
import mlx.nn as nn

# Kernel offsets in (x, y, z) row-major order: the order of spconv's KRSC
# weight layout ``[out, kx, ky, kz, in]``, which the checkpoints use.
OFFSETS = tuple(product((-1, 0, 1), repeat=3))

# Bytes of gathered im2col rows built per matrix product.
CONV_CHUNK_BYTES = 256 << 20


def nonzero(mask):
    """Indices of the true entries of a 1-D mask, in increasing order."""
    count = int(mx.sum(mask).item())
    return mx.sort(mx.where(mask, mx.arange(mask.size), mask.size))[:count]


def coord_keys(coords, extent):
    """Lexicographic int64 key of ``(x, y, z)`` rows on a cube of side ``extent``."""
    c = coords.astype(mx.int64)
    return (c[..., 0] * extent + c[..., 1]) * extent + c[..., 2]


def unique_inverse(keys):
    """Sorted unique keys, the row of each unique key's first occurrence and the
    inverse map from each input row to its unique key."""
    order = mx.argsort(keys)
    sorted_keys = keys[order]
    start = mx.concatenate([mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
    first = nonzero(start)
    groups = mx.cumsum(start.astype(mx.int32)) - 1
    inverse = mx.zeros(keys.size, mx.int32).at[order].add(groups)
    return sorted_keys[first], order[first], inverse


def lookup(sorted_keys, query, missing):
    """Position of each query in ``sorted_keys``, or ``missing`` when absent."""
    flat = query.reshape(-1)
    idx = mx.minimum(mx.searchsorted(sorted_keys, flat), sorted_keys.size - 1)
    found = sorted_keys[idx] == flat
    return mx.where(found, idx, missing).reshape(query.shape)


class Grid:
    """Occupied voxels of one object: ``(N, 3)`` integer ``(x, y, z)`` coordinates."""

    def __init__(self, coords, resolution):
        self.coords = coords.astype(mx.int32)
        self.resolution = int(resolution)
        self.cache = {}

    def __len__(self):
        return self.coords.shape[0]

    def neighbors(self):
        """``(N, 27)`` row of the voxel at each kernel offset, ``N`` where empty."""
        if "neighbors" not in self.cache:
            extent = self.resolution + 2
            keys = coord_keys(self.coords + 1, extent)
            order = mx.argsort(keys)
            offsets = mx.array(OFFSETS, dtype=mx.int32)
            query = coord_keys(self.coords[:, None] + 1 + offsets[None], extent)
            position = lookup(keys[order], query, len(self))
            rows = mx.concatenate([order, mx.array([len(self)], order.dtype)])
            self.cache["neighbors"] = rows[position]
        return self.cache["neighbors"]

    def downsample(self):
        """The 2x coarser grid and each voxel's parent row in it.

        Parents are ordered lexicographically, as ``torch.unique`` orders the
        packed codes of the reference downsample.
        """
        if "downsample" not in self.cache:
            coords = self.coords // 2
            resolution = (self.resolution + 1) // 2
            _, first, parent = unique_inverse(coord_keys(coords, resolution))
            self.cache["downsample"] = (Grid(coords[first], resolution), parent)
        return self.cache["downsample"]


def sparse_conv(x, grid, weight, bias=None):
    """Submanifold 3x3x3 convolution of ``x`` over ``grid``.

    ``x`` is ``(N, C)`` or ``(B, N, C)`` with every batch row on the same grid;
    ``weight`` is ``(O, 3, 3, 3, C)``. Each voxel's 27 neighbors are gathered
    into one im2col row so the product accumulates in a single matmul.
    """
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    batch, count, channels = x.shape
    out_channels = weight.shape[0]
    w = weight.reshape(out_channels, -1).T.astype(x.dtype)
    indices = grid.neighbors()
    padded = mx.concatenate([x, mx.zeros((batch, 1, channels), x.dtype)], axis=1)
    rows = max(1, CONV_CHUNK_BYTES // (batch * 27 * channels * x.dtype.size))
    outputs = []
    for start in range(0, count, rows):
        cols = padded[:, indices[start : start + rows]].reshape(batch, -1, w.shape[0])
        if bias is None:
            outputs.append(cols @ w)
        else:
            outputs.append(mx.addmm(bias.astype(x.dtype), cols, w))
    out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
    return out[0] if squeeze else out


class SparseConv3d(nn.Module):
    """Parameters of one spconv ``SubMConv3d``: ``weight`` ``(O, 3, 3, 3, I)``."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = mx.zeros((out_channels, 3, 3, 3, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))

    def __call__(self, x, grid):
        return sparse_conv(x, grid, self.weight, getattr(self, "bias", None))


def downsample_mean(x, grid):
    """Average ``x`` over each 2x2x2 parent block.

    Matches ``torch.scatter_reduce(..., "mean")`` with its default
    ``include_self=True``: the zero the output starts from counts as one
    sample, so each mean divides by the child count plus one.
    """
    child, parent = grid.downsample()
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None]
    batch, _, channels = x.shape
    sums = mx.zeros((batch, len(child), channels), mx.float32)
    sums = sums.at[:, parent].add(x.astype(mx.float32))
    counts = mx.ones((len(child),), mx.float32).at[parent].add(1.0)
    out = (sums / counts[None, :, None]).astype(x.dtype)
    return (out[0] if squeeze else out), child
