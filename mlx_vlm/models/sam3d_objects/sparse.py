"""Sparse inference operators using MLX gathers, reductions, and attention."""

from dataclasses import dataclass, field
from itertools import product

import mlx.core as mx
import mlx.nn as nn

from .layers import attend


def nonzero(mask):
    mask = mask.reshape(-1)
    count = int(mx.sum(mask).item())
    return mx.sort(mx.where(mask, mx.arange(mask.size), mask.size))[:count]


def coord_keys(coords, resolution):
    return (
        (coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2]
    ) * resolution + coords[:, 3]


@dataclass
class Grid:
    coords: mx.array
    resolution: int
    cache: dict = field(default_factory=dict)

    def neighbors(self):
        if "neighbors" not in self.cache:
            offsets = mx.array(list(product((-1, 0, 1), repeat=3)), dtype=mx.int32)
            positions = self.coords[:, None, 1:] + offsets[None]
            valid = mx.all((positions >= 0) & (positions < self.resolution), axis=-1)
            r = self.resolution
            nkeys = (
                (self.coords[:, :1] * r + positions[:, :, 0]) * r + positions[:, :, 1]
            ) * r + positions[:, :, 2]
            keys = coord_keys(self.coords, r)
            order = mx.argsort(keys)
            sorted_keys = keys[order]
            index = mx.minimum(
                mx.searchsorted(sorted_keys, nkeys.reshape(-1)), keys.size - 1
            ).reshape(nkeys.shape)
            found = valid & (sorted_keys[index] == nkeys)
            self.cache["neighbors"] = mx.where(found, order[index], keys.size)
        return self.cache["neighbors"]

    def downsample(self):
        if "downsample" not in self.cache:
            coords = self.coords // mx.array([1, 2, 2, 2])
            r = (self.resolution + 1) // 2
            keys = coord_keys(coords, r)
            order = mx.argsort(keys)
            sorted_keys = keys[order]
            start = mx.concatenate(
                [mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]]
            )
            first = nonzero(start)
            groups = mx.cumsum(start.astype(mx.int32)) - 1
            parent = mx.zeros(keys.size, mx.int32).at[order].add(groups)
            child = Grid(coords[order[first]], r)
            self.cache["downsample"] = (child, parent)
        return self.cache["downsample"]

    def subdivide(self):
        if "subdivide" not in self.cache:
            offsets = mx.array(
                [(0, *p) for p in product((0, 1), repeat=3)], dtype=mx.int32
            )
            coords = self.coords[:, None] * mx.array([1, 2, 2, 2]) + offsets[None]
            self.cache["subdivide"] = Grid(coords.reshape(-1, 4), self.resolution * 2)
        return self.cache["subdivide"]

    def windows(self, size, shift):
        key = ("windows", size, shift)
        if key not in self.cache:
            coords = mx.concatenate(
                [self.coords[:, :1], (self.coords[:, 1:] + shift) // size], axis=-1
            )
            keys = coord_keys(coords, (self.resolution + shift + size - 1) // size)
            order = mx.argsort(keys)
            sorted_keys = keys[order]
            starts = nonzero(
                mx.concatenate([mx.array([True]), sorted_keys[1:] != sorted_keys[:-1]])
            ).tolist()
            ends = starts[1:] + [keys.size]
            buckets = {}
            for start, end in zip(starts, ends):
                width = 1 << (end - start - 1).bit_length()
                buckets.setdefault(width, []).append((start, end - start))
            packed = []
            for width, ranges in sorted(buckets.items()):
                first = mx.array([s for s, _ in ranges])[:, None]
                lengths = mx.array([n for _, n in ranges])[:, None]
                offset = mx.arange(width)[None]
                valid = offset < lengths
                indices = mx.where(
                    valid, order[mx.minimum(first + offset, keys.size - 1)], keys.size
                )
                packed.append((indices, valid))
            self.cache[key] = packed
        return self.cache[key]


class SparseConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(
            input_channels, output_channels, kernel, padding=kernel // 2
        )

    def __call__(self, x, grid):
        w, bias = self.conv.weight, self.conv.bias
        if w.shape[1] == 1:
            return mx.addmm(bias, x, w[:, 0, 0, 0].T)
        indices = grid.neighbors()
        padded = mx.concatenate([x, mx.zeros((1, x.shape[-1]), x.dtype)])
        # Limit temporary im2col storage to 16 MiB per matrix product.
        rows = max(1, (16 << 20) // (27 * x.shape[-1] * x.dtype.size))
        weight = w.reshape(w.shape[0], -1).T
        outputs = []
        for start in range(0, x.shape[0], rows):
            values = padded[indices[start : start + rows]].reshape(-1, weight.shape[0])
            outputs.append(mx.addmm(bias, values, weight))
        return mx.concatenate(outputs, axis=0)

    def from_parents(self, x, parent, grid):
        w, bias = self.conv.weight, self.conv.bias
        if w.shape[1] == 1:
            return mx.addmm(bias, x, w[:, 0, 0, 0].T)[parent]
        co, taps = w.shape[0], w.shape[1] * w.shape[2] * w.shape[3]
        weight = w.reshape(co, taps, -1).transpose(2, 1, 0).astype(mx.float32)
        x = x.astype(mx.float32)
        rows = mx.concatenate([parent, mx.array([x.shape[0]], parent.dtype)])
        rows = rows[grid.neighbors()]
        # Bound the float32 partial sums: 256 MiB of projected parent rows per
        # output-channel chunk and 64 MiB of gathered rows per voxel chunk.
        channels = min(co, max(1, (256 << 20) // ((x.shape[0] + 1) * taps * 4)))
        voxels = max(1, (64 << 20) // (taps * channels * 4))
        columns = []
        for c in range(0, co, channels):
            width = min(channels, co - c)
            projected = x @ weight[:, :, c : c + width].reshape(x.shape[-1], -1)
            projected = mx.concatenate(
                [
                    projected.reshape(-1, taps, width),
                    mx.zeros((1, taps, width), mx.float32),
                ]
            )
            columns.append(
                mx.concatenate(
                    [
                        projected[rows[v : v + voxels], mx.arange(taps)].sum(axis=1)
                        for v in range(0, rows.shape[0], voxels)
                    ]
                )
            )
        out = columns[0] if len(columns) == 1 else mx.concatenate(columns, axis=-1)
        return (out + bias).astype(w.dtype)


def pool(x, grid):
    child, parent = grid.downsample()
    sums = (
        mx.zeros((child.coords.shape[0], x.shape[-1]), mx.float32)
        .at[parent]
        .add(x.astype(mx.float32))
    )
    # Upstream torch.scatter_reduce(mean) includes the initial zero value.
    counts = (
        mx.ones((child.coords.shape[0], 1), mx.float32)
        .at[parent]
        .add(mx.ones((x.shape[0], 1)))
    )
    return (sums / counts).astype(x.dtype), child, parent


class SparseGroupNorm(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.weight = mx.ones(channels)
        self.bias = mx.zeros(channels)
        self.groups = min(groups, channels)

    def __call__(self, x):
        # Each streamed request has one object; normalize across all its voxels.
        h = x.astype(mx.float32).reshape(x.shape[0], self.groups, -1)
        mean = h.mean(axis=(0, 2), keepdims=True)
        var = ((h - mean) ** 2).mean(axis=(0, 2), keepdims=True)
        h = ((h - mean) * mx.rsqrt(var + 1e-5)).reshape(x.shape)
        return (h * self.weight + self.bias).astype(x.dtype)


def window_attention(qkv, grid, size, shift):
    n, _, heads, dim = qkv.shape
    padded = mx.concatenate([qkv, mx.zeros((1, 3, heads, dim), qkv.dtype)])
    result = mx.zeros((n + 1, heads * dim), qkv.dtype)
    for indices, valid in grid.windows(size, shift):
        values = padded[indices]
        out = attend(*(values[:, :, i] for i in range(3)), mask=valid[:, None, None, :])
        result = result.at[indices.reshape(-1)].add(
            mx.where(valid[..., None], out, 0).reshape(-1, heads * dim)
        )
    return result[:n]
