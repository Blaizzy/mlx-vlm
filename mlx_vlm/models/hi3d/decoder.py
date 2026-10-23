"""Sparse Point Transformer V3 mesh decoder of Hi3D (the released ``vae``).

Decodes sparse latents into per-voxel corner signs, signed distances and
deformations, doubling the grid at each stage and keeping only the children
the occupancy head predicts. Runs in float32 except the windowed attention,
which runs in float16 like the reference's flash attention.
"""

import logging

import mlx.core as mx
import mlx.nn as nn

from .sparse import Grid, SparseConv3d, nonzero

# Residual blocks pass larger point clouds through unchanged, as the
# reference does; the released weights were measured with this guard active.
RESNET_SKIP_POINTS = 12_000_000

# Child offsets of the octree split, in the checkpoint's ``inv_grid_coord`` order.
CHILD_OFFSETS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


class GroupNorm(nn.Module):
    """Group norm over all voxels of one object, with the reference's moments:
    ``sum / (count + eps)`` and ``E[x²] - mean²``."""

    def __init__(self, channels, group=32, min_group_channels=4, eps=1e-6):
        super().__init__()
        if min_group_channels * group > channels:
            group = channels // min_group_channels
        self.group = group
        self.eps = eps
        self.weights = mx.ones((1, channels))
        self.bias = mx.zeros((1, channels))

    def __call__(self, x):
        count, channels = x.shape
        h = x.astype(mx.float32).reshape(count, self.group, -1)
        total = count * h.shape[-1] + self.eps
        mean = mx.sum(h, axis=(0, 2)) / total
        var = mx.sum(h * h, axis=(0, 2)) / total - mean * mean
        h = (h - mean[:, None]) * mx.rsqrt(var + self.eps)[:, None]
        return h.reshape(count, channels) * self.weights + self.bias


class Swish(nn.Module):
    def __call__(self, x):
        return x * mx.sigmoid(x)


def run(layers, x, grid):
    """Apply a ``PointSequential``: sparse convolutions also take the grid."""
    for layer in layers:
        x = layer(x, grid) if isinstance(layer, SparseConv3d) else layer(x)
    return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = [
            GroupNorm(in_channels),
            Swish(),
            SparseConv3d(in_channels, out_channels),
            GroupNorm(out_channels),
            Swish(),
            SparseConv3d(out_channels, out_channels),
        ]
        if in_channels != out_channels:
            self.res_conv = nn.Linear(in_channels, out_channels)

    def __call__(self, x, grid):
        if x.shape[0] > RESNET_SKIP_POINTS:
            logging.warning(
                "Hi3D decoder residual block skipped at %d points, as in the "
                "reference implementation",
                x.shape[0],
            )
            return x
        shortcut = self.res_conv(x) if "res_conv" in self else x
        return shortcut + run(self.block, x, grid)


def serialize(coords, order):
    """Z-order (Morton) permutation of ``coords``; ``z-trans`` swaps x and y."""
    if order == "z-trans":
        coords = coords[:, [1, 0, 2]]
    elif order != "z":
        raise ValueError(f"unsupported serialization order: {order!r}")
    depth = max(int(mx.max(coords).item()).bit_length(), 1)
    c = coords.astype(mx.int64)
    code = mx.zeros(c.shape[0], mx.int64)
    for bit in range(depth):
        for axis, shift in ((0, 2), (1, 1), (2, 0)):
            code = code | (((c[:, axis] >> bit) & 1) << (3 * bit + shift))
    perm = mx.argsort(code)
    inverse = mx.zeros_like(perm).at[perm].add(mx.arange(perm.size, dtype=perm.dtype))
    return perm, inverse


def window_attention(qkv, perm, inverse, size, scale):
    """Attention within consecutive windows of ``size`` serialized points.

    ``qkv`` is ``(N, 3, H, D)``. When ``N`` is not a multiple of ``size``,
    the remainder attends within the last ``size`` points of the sequence,
    which is what the reference's patch padding (repeating the previous
    patch's tail) computes.
    """
    n, _, heads, dim = qkv.shape
    seq = qkv[perm]
    if n <= size:
        windows = seq[None]
    else:
        full, rest = divmod(n, size)
        windows = seq[: full * size].reshape(full, size, 3, heads, dim)
        if rest:
            windows = mx.concatenate([windows, seq[n - size :][None]])
    q, k, v = (windows[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    out = out.transpose(0, 2, 1, 3).reshape(out.shape[0], -1, heads * dim)
    if n <= size:
        flat = out[0]
    else:
        flat = out[:full].reshape(full * size, -1)
        if rest:
            flat = mx.concatenate([flat, out[full, size - rest :]])
    return flat[inverse]


class SerializedAttention(nn.Module):
    def __init__(self, channels, num_heads, patch_size, order_index):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.order_index = order_index
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def __call__(self, x, orders):
        perm, inverse = orders[self.order_index]
        qkv = self.qkv(x).reshape(x.shape[0], 3, self.num_heads, -1)
        # Flash attention numerics: half-precision inputs and output, float32
        # softmax and accumulation.
        qkv = qkv.astype(mx.float16).astype(mx.float32)
        out = window_attention(qkv, perm, inverse, self.patch_size, self.scale)
        return self.proj(out.astype(mx.float16).astype(x.dtype))


class MLP(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def __call__(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, patch_size, mlp_ratio, order_index):
        super().__init__()
        self.cpe = [
            SparseConv3d(channels, channels),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
        ]
        self.norm1 = [nn.LayerNorm(channels)]
        self.attn = SerializedAttention(channels, num_heads, patch_size, order_index)
        self.norm2 = [nn.LayerNorm(channels)]
        self.mlp = [MLP(channels, int(channels * mlp_ratio))]

    def __call__(self, x, grid, orders):
        x = x + run(self.cpe, x, grid)
        x = x + self.attn(self.norm1[0](x), orders)
        return x + self.mlp[0](self.norm2[0](x))


class ClsHead(nn.Module):
    def __init__(self, channels, outputs):
        super().__init__()
        self.fc = [GroupNorm(channels), Swish(), SparseConv3d(channels, outputs)]

    def __call__(self, x, grid):
        return run(self.fc, x, grid)


class Unpooling(nn.Module):
    """Split each voxel into eight children and keep the predicted occupied ones."""

    def __init__(self, channels):
        super().__init__()
        self.cls_head = ClsHead(channels, 8)
        self.proj = nn.Linear(channels, channels * 8, bias=False)

    def __call__(self, x, grid):
        count, channels = x.shape
        feat = self.proj(x).reshape(count * 8, channels)
        keep = nonzero(self.cls_head(x, grid).reshape(-1) > 0)
        children = mx.array(CHILD_OFFSETS, dtype=mx.int32)
        coords = (grid.coords[:, None] * 2 + children[None]).reshape(-1, 3)
        return feat[keep], Grid(coords[keep], grid.resolution * 2)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = Unpooling(channels)

    def __call__(self, x, grid):
        return self.upsample(x, grid)


class ZToF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = [SparseConv3d(in_channels, out_channels)]

    def __call__(self, x, grid):
        return run(self.conv, x, grid)


class ConvOut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = [
            GroupNorm(in_channels),
            Swish(),
            SparseConv3d(in_channels, out_channels),
        ]

    def __call__(self, x, grid):
        return run(self.conv, x, grid)


class Stage(nn.Module):
    """One decoder stage: residual blocks named ``block{i}``, then ``up``."""

    def __init__(self, blocks, up):
        super().__init__()
        for i, block in enumerate(blocks):
            setattr(self, f"block{i}", block)
        self.num_blocks = len(blocks)
        if up is not None:
            self.up = up

    def __call__(self, x, grid):
        for i in range(self.num_blocks):
            x = getattr(self, f"block{i}")(x, grid)
        if "up" in self:
            x, grid = self.up(x, grid)
        return x, grid


class Mid(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.block_in = ResnetBlock(channels, channels)
        self.num_blocks = config.mid_depth
        for i in range(config.mid_depth):
            block = AttentionBlock(
                channels,
                config.mid_num_head,
                config.patch_size,
                config.mlp_ratio,
                i % len(config.order),
            )
            setattr(self, f"block{i}", block)
        self.block_out = ResnetBlock(channels, channels)

    def __call__(self, x, grid, orders):
        x = self.block_in(x, grid)
        for i in range(self.num_blocks):
            x = getattr(self, f"block{i}")(x, grid, orders)
        return self.block_out(x, grid)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        channels = list(config.channels)
        self.z_to_f = ZToF(config.z_channels, channels[-1])
        self.mid = Mid(channels[-1], config)
        self.dec = Stages(channels, config.depths)
        self.conv_out = ConvOut(channels[0], 8 + config.out_channels)

    def __call__(self, latents, grid: Grid):
        """Per-voxel outputs ``(N', 8 + out_channels)`` and the final grid."""
        x = self.z_to_f(latents.astype(mx.float32), grid)
        orders = [serialize(grid.coords, order) for order in self.config.order]
        x = self.mid(x, grid, orders)
        x, grid = self.dec(x, grid)
        return self.conv_out(x, grid), grid


class Stages(nn.Module):
    """``dec{s}`` stages from the coarsest (``s = n - 1``) to the finest."""

    def __init__(self, channels, depths):
        super().__init__()
        self.names = []
        for s in reversed(range(len(depths))):
            block_in = channels[s]
            block_out = channels[s - 1] if s > 0 else channels[0]
            blocks = []
            for _ in range(depths[s]):
                blocks.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            setattr(self, f"dec{s}", Stage(blocks, Upsample(block_out) if s else None))
            self.names.append(f"dec{s}")

    def __call__(self, x, grid):
        for name in self.names:
            x, grid = getattr(self, name)(x, grid)
        return x, grid


class VAE(nn.Module):
    """Holds the decoder under the checkpoint's ``decoder.`` prefix."""

    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config)

    def __call__(self, latents, grid):
        return self.decoder(latents, grid)
