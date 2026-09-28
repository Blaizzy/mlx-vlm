"""Occupancy, Gaussian, and mesh-feature inference decoders."""

import math

import mlx.core as mx
import mlx.nn as nn

from .layers import (
    Attention,
    FeedForward,
    LayerNorm,
    Sequential,
    layer_norm,
    position_embedding,
)
from .sparse import SparseConv, SparseGroupNorm, window_attention


class DenseResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.norm2 = LayerNorm(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def __call__(self, x):
        return x + self.conv2(nn.silu(self.norm2(self.conv1(nn.silu(self.norm1(x))))))


class DenseUpsample(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Conv3d(inputs, outputs * 8, 3, padding=1)

    def __call__(self, x):
        x = self.conv(x)
        b, d, h, w, c = x.shape
        x = x.reshape(b, d, h, w, c // 8, 2, 2, 2)
        return x.transpose(0, 1, 5, 2, 6, 3, 7, 4).reshape(
            b, d * 2, h * 2, w * 2, c // 8
        )


class StructureDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.structure_channels
        self.input_layer = nn.Conv3d(config.latent_channels, channels[0], 3, padding=1)
        self.middle_block = Sequential(
            *(DenseResBlock(channels[0]) for _ in range(config.structure_res_blocks))
        )
        self.blocks = []
        for i, c in enumerate(channels):
            self.blocks.extend(
                DenseResBlock(c) for _ in range(config.structure_res_blocks)
            )
            if i + 1 < len(channels):
                self.blocks.append(DenseUpsample(c, channels[i + 1]))
        self.out_layer = Sequential(
            LayerNorm(channels[-1]), nn.SiLU(), nn.Conv3d(channels[-1], 1, 3, padding=1)
        )

    def __call__(self, x):
        side = round(x.shape[1] ** (1 / 3))
        x = x.reshape(x.shape[0], side, side, side, x.shape[-1])
        x = self.middle_block(self.input_layer(x.astype(self.input_layer.weight.dtype)))
        for block in self.blocks:
            x = block(x)
        return self.out_layer(x)


class WindowBlock(nn.Module):
    def __init__(self, channels, heads, window, shift):
        super().__init__()
        self.attn = Attention(channels, heads)
        self.mlp = FeedForward(channels)
        self._window, self._shift, self._heads = window, shift, heads

    def __call__(self, x, grid):
        h = self.attn.to_qkv(layer_norm(x, 1e-6)).reshape(
            x.shape[0], 3, self._heads, -1
        )
        h = window_attention(h, grid, self._window, self._shift)
        x = x + self.attn.to_out(h)
        return x + self.mlp(layer_norm(x, 1e-6))


class DecoderBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.decoder_channels
        self.input_layer = nn.Linear(config.latent_channels, c)
        self.blocks = [
            WindowBlock(
                c,
                config.decoder_heads,
                config.window_size,
                config.window_size // 2 * (i % 2),
            )
            for i in range(config.decoder_blocks)
        ]

    def encode(self, x, grid):
        x = self.input_layer(x.astype(self.input_layer.weight.dtype))
        x = x + position_embedding(grid.coords[:, 1:], x.shape[-1]).astype(x.dtype)
        for block in self.blocks:
            x = block(x, grid)
        return x


class GaussianDecoder(DecoderBase):
    def __init__(self, config, count=32):
        super().__init__(config)
        self.out_layer = nn.Linear(config.decoder_channels, 14 * count)
        self.offset_perturbation = mx.zeros((count, 3))
        self._count = count

    def __call__(self, x, grid):
        h = self.out_layer(layer_norm(self.encode(x, grid))).astype(mx.float32)
        n = self._count
        xyz, color, scaling, rotation, opacity = mx.split(
            h, [n * 3, n * 6, n * 9, n * 13], axis=-1
        )
        xyz = (
            grid.coords[:, None, 1:].astype(mx.float32) + 0.5
        ) / grid.resolution + mx.tanh(
            xyz.reshape(-1, n, 3) + self.offset_perturbation
        ) * (
            0.75 / grid.resolution
        )
        scale = nn.softplus(scaling.reshape(-1, 3) + math.log(math.expm1(0.004)))
        scale = mx.sqrt(scale * scale + 0.0009**2)
        rotation = rotation.reshape(-1, 4) * 0.1 + mx.array([1, 0, 0, 0])
        rotation = rotation * mx.rsqrt(
            mx.maximum(mx.sum(rotation * rotation, axis=-1, keepdims=True), 1e-24)
        )
        opacity = mx.sigmoid(opacity.reshape(-1, 1) + math.log(0.1 / 0.9))
        return {
            "positions": xyz.reshape(-1, 3) - 0.5,
            "sh_dc": color.reshape(-1, 3),
            "scales": scale,
            "rotations": rotation,
            "opacities": opacity,
        }


class SubdivideBlock(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        self.act_layers = Sequential(SparseGroupNorm(input_channels), nn.SiLU())
        self.out_layers = Sequential(
            SparseConv(input_channels, channels),
            SparseGroupNorm(channels),
            nn.SiLU(),
            SparseConv(channels, channels),
        )
        self.skip_connection = SparseConv(input_channels, channels, kernel=1)

    def __call__(self, x, grid):
        grid = grid.subdivide()
        parent = mx.repeat(mx.arange(x.shape[0]), 8)
        h = self.out_layers["0"].from_parents(self.act_layers(x), parent, grid)
        h = self.out_layers["2"](self.out_layers["1"](h))
        h = self.out_layers["3"](h, grid)
        return h + self.skip_connection.from_parents(x, parent, grid), grid


class MeshDecoder(DecoderBase):
    def __init__(self, config):
        super().__init__(config)
        c = config.decoder_channels
        self.upsample = [SubdivideBlock(c, c // 4), SubdivideBlock(c // 4, c // 8)]
        self.out_layer = nn.Linear(c // 8, 101)

    def __call__(self, x, grid):
        x = self.encode(x, grid)
        for block in self.upsample:
            x, grid = block(x, grid)
        return self.out_layer(x), grid
