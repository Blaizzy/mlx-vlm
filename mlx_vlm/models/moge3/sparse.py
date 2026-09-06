"""Pure-MLX sparse 3D ops matching FlexGEMM's explicit-GEMM semantics.

MoGe-3's refiner is a sparse 3D UNet over voxels addressed by integer
coords ``(batch, i, j, z)``. The reference implementation uses FlexGEMM
(Triton, CUDA-only); here the same ops are expressed with binary search +
gather, which run on any MLX backend.

Conventions (identical to the reference):
- ``coords``: (M, 4) int32, columns ``(batch, i, j, z)``, in ascending
  raster order (see :func:`_coords_to_keys`). The voxelizer emits coords in
  this order and :func:`sparse_pool2x_mean` preserves it, so neighbor
  lookups can binary-search the keys without sorting.
- ``shape``: the spatial extent ``(B, H, W, Z)``. Entries may be Python
  ints or 0-d MLX integer arrays (the voxelizer's ``Z`` is a lazy scalar).
- Submanifold conv weight: ``(Co, K0, K1, K2, Ci)``; kernel slot order is
  ``di``-major, ``dz``-minor. Missing neighbors contribute zero; the bias
  is always added.
"""

import mlx.core as mx
import mlx.nn as nn

# Kernel offsets (di, dj, dz) in product order, matching FlexGEMM.
_KERNEL_OFFSETS = mx.array(
    [(di, dj, dz) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=mx.int32,
)

# Target size of the (rows, 27, Ci) im2col gather in the submanifold conv.
_GATHER_CHUNK_BYTES = 16 << 20


def _coords_to_keys(coords, shape):
    """Raster-scan int64 keys for coords (..., 4) over spatial shape (B, H, W, Z)."""
    _, H, W, Z = shape
    c = coords.astype(mx.int64)
    return ((c[..., 0] * H + c[..., 1]) * W + c[..., 2]) * Z + c[..., 3]


def _axis_in_bounds(c, d, size):
    """(M, 27) bool: ``c + d`` in ``[0, size)`` for ``c`` (M, 1), ``d`` (27,) in {-1, 0, 1}."""
    return ((d >= 0) | (c > 0)) & ((d <= 0) | (c < size - 1))


def submanifold_conv3d_neighbor_map(coords, shape):
    """(M, 27) int32 map: voxel index at each kernel offset, -1 if missing.

    ``coords`` must be in ascending raster order (see module docstring).
    """
    _, H, W, Z = shape
    keys = _coords_to_keys(coords, shape)
    di, dj, dz = (_KERNEL_OFFSETS[:, a] for a in range(3))
    key_offsets = (di.astype(mx.int64) * W + dj) * Z + dz
    nkeys = keys[:, None] + key_offsets[None]
    valid = (
        _axis_in_bounds(coords[:, 1:2], di, H)
        & _axis_in_bounds(coords[:, 2:3], dj, W)
        & _axis_in_bounds(coords[:, 3:4], dz, Z)
    )
    idx = mx.minimum(mx.searchsorted(keys, nkeys.reshape(-1)), keys.shape[0] - 1)
    idx = idx.reshape(nkeys.shape)
    found = (keys[idx] == nkeys) & valid
    return mx.where(found, idx.astype(mx.int32), -1)


class SubmanifoldConv3d(nn.Module):
    """3x3x3 submanifold sparse convolution; weight layout (Co, 3, 3, 3, Ci)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = mx.zeros((out_channels, 3, 3, 3, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, feats, coords, shape, neighbor_map=None):
        if neighbor_map is None:
            neighbor_map = submanifold_conv3d_neighbor_map(coords, shape)
        M, Ci = feats.shape
        Co = self.weight.shape[0]
        # Missing neighbors gather an appended zero row instead of being masked.
        feats_pad = mx.concatenate([feats, mx.zeros((1, Ci), dtype=feats.dtype)])
        idx = mx.where(neighbor_map >= 0, neighbor_map, M)
        w = self.weight.reshape(Co, 27 * Ci).T
        rows = max(1, _GATHER_CHUNK_BYTES // (27 * Ci * feats.dtype.size))
        out = [
            mx.addmm(self.bias, feats_pad[idx[s : s + rows]].reshape(-1, 27 * Ci), w)
            for s in range(0, M, rows)
        ]
        return (out[0] if len(out) == 1 else mx.concatenate(out)), neighbor_map


def _ceil_div(a, b):
    return (a + b - 1) // b


def sparse_pool2x_mean(feats, coords, shape):
    """Mean pool with kernel=stride=2 over (i, j, z).

    Returns (out_feats, out_coords, out_shape, parent_idx) where
    ``parent_idx[m]`` is the output segment of input voxel ``m`` (aligned
    with the input order). Output coords are in ascending raster order, and
    the segment ids double as the gather indices for the symmetric nearest
    upsample.
    """
    B, H, W, Z = shape
    out_shape = (B, _ceil_div(H, 2), _ceil_div(W, 2), _ceil_div(Z, 2))
    tcoords = coords // mx.array([1, 2, 2, 2], dtype=mx.int32)
    tkeys = _coords_to_keys(tcoords, out_shape)
    order = mx.argsort(tkeys)
    sorted_keys = tkeys[order]
    is_new = mx.concatenate(
        [mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]], axis=0
    )
    seg_sorted = mx.cumsum(is_new.astype(mx.int32)) - 1
    num_segments = int(is_new.sum().item())

    parent_idx = mx.zeros(coords.shape[0], dtype=mx.int32)
    parent_idx = parent_idx.at[order].add(seg_sorted)

    sorted_feats = feats[order]
    C = feats.shape[-1]
    sums = mx.zeros((num_segments, C), dtype=feats.dtype)
    sums = sums.at[seg_sorted].add(sorted_feats)
    counts = mx.zeros((num_segments, 1), dtype=feats.dtype)
    counts = counts.at[seg_sorted].add(
        mx.ones(coords.shape[0], dtype=feats.dtype)[:, None]
    )
    out_feats = sums / counts

    M = coords.shape[0]
    first_pos = mx.full((num_segments,), M, dtype=mx.int32)
    first_pos = first_pos.at[seg_sorted].minimum(mx.arange(M, dtype=mx.int32))
    out_coords = tcoords[order[first_pos]]
    return out_feats, out_coords, out_shape, parent_idx


def sparse_upsample2x_nearest(feats, parent_idx, target_coords, target_shape):
    """Nearest 2x upsample: each target voxel copies its parent input voxel.

    ``parent_idx`` comes from the matching :func:`sparse_pool2x_mean` call
    whose input coords are ``target_coords``.
    """
    return feats[parent_idx], target_coords, target_shape
