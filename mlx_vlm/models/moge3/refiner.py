"""Sparse 3D UNet refiner for MoGe-3 (pure-MLX port).

Ports ``moge.model.modules.sparse_unet.Sparse3DUNet`` and the FlexGEMM-based
blocks in ``flex_sparse_blocks``. Operates purely on sparse features; the
caller builds the voxelization and interprets the output.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import RefinerConfig
from .conv_stack import run_sequential
from .sparse import SubmanifoldConv3d, sparse_pool2x_mean, sparse_upsample2x_nearest


class SparseResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SubmanifoldConv3d(channels, out_channels)
        self.conv2 = SubmanifoldConv3d(out_channels, out_channels)
        self.skip_connection = (
            nn.Linear(channels, out_channels) if channels != out_channels else None
        )

    def __call__(self, feats, coords, shape, neighbor_map=None):
        h = nn.silu(self.norm1(feats))
        h, neighbor_map = self.conv1(h, coords, shape, neighbor_map=neighbor_map)
        h = nn.silu(h)
        h, neighbor_map = self.conv2(h, coords, shape, neighbor_map=neighbor_map)
        skip = (
            self.skip_connection(feats) if self.skip_connection is not None else feats
        )
        return h + skip, neighbor_map


class PoolDown(nn.Module):
    """2x mean pool followed by a channel projection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def __call__(self, feats, coords, shape):
        feats, coords, shape, parent_idx = sparse_pool2x_mean(feats, coords, shape)
        return self.linear(feats), coords, shape, parent_idx


class NearestUp(nn.Module):
    """Channel projection followed by a 2x nearest upsample onto target coords."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)

    def __call__(self, feats, parent_idx, target_coords, target_shape):
        return sparse_upsample2x_nearest(
            self.linear(feats), parent_idx, target_coords, target_shape
        )


class Sparse3DUNet(nn.Module):
    """Sparse 3D UNet with separated resampling and residual refinement.

    forward(feats, coords, shape, encoder_feature):
    - feats: (M, in_channels) raw input features.
    - coords: (M, 4) int32, columns (batch, i, j, z_bin), raster-sorted.
    - shape: spatial extent (B, H, W, Z).
    - encoder_feature: (B, H/s, W/s, C_enc) dense conditioning map sampled at
      the bottleneck coords, with s = ``encoder_downsample``.
    Returns: (M, out_channels).
    """

    def __init__(self, config: RefinerConfig):
        super().__init__()
        model_channels = config.model_channels
        num_levels = len(model_channels)
        if num_levels < 2:
            raise ValueError("model_channels must have at least 2 levels")
        downsample_factors = config.downsample_factors or [2] * (num_levels - 1)
        if len(downsample_factors) != num_levels - 1:
            raise ValueError(
                "downsample_factors must have len(model_channels) - 1 entries"
            )
        if any(f != 2 for f in downsample_factors):
            raise ValueError("MoGe refiner only supports downsample factor 2")
        self.encoder_downsample = config.encoder_downsample
        if self.encoder_downsample != math.prod(downsample_factors):
            raise ValueError("encoder_downsample must equal prod(downsample_factors)")

        def resolve(value, n, name):
            value = [value] * n if isinstance(value, int) else list(value)
            if len(value) != n:
                raise ValueError(f"{name} must have length {n}, got {value}")
            return value

        encoder_block_counts = resolve(
            config.encoder_blocks_per_level, num_levels, "encoder_blocks_per_level"
        )
        decoder_block_counts = resolve(
            config.decoder_blocks_per_level,
            num_levels - 1,
            "decoder_blocks_per_level",
        )

        self.input_proj = nn.Linear(config.in_channels, model_channels[0])
        self.encoder_fuse = nn.Linear(config.encoder_channels, model_channels[-1])
        bottleneck_channels = model_channels[-1]
        self.fuse_proj = [
            nn.Linear(bottleneck_channels * 2, bottleneck_channels),
            nn.SiLU(),
            nn.Linear(bottleneck_channels, bottleneck_channels),
        ]

        self.down_stages = [
            [SparseResBlock3d(ch) for _ in range(encoder_block_counts[i])]
            for i, ch in enumerate(model_channels)
        ]
        self.downsample_blocks = [
            PoolDown(model_channels[i], model_channels[i + 1])
            for i in range(num_levels - 1)
        ]
        self.bottleneck_stage = [
            SparseResBlock3d(model_channels[-1])
            for _ in range(config.bottleneck_blocks)
        ]
        # Decoder runs from the deepest level up: transition i goes from
        # level (num_levels - 1 - i) to level (num_levels - 2 - i).
        self.upsample_blocks = [
            NearestUp(model_channels[level], model_channels[level - 1])
            for level in range(num_levels - 1, 0, -1)
        ]
        self.up_stages = [
            [SparseResBlock3d(model_channels[level - 1]) for _ in range(n)]
            for level, n in zip(range(num_levels - 1, 0, -1), decoder_block_counts)
        ]

        self.out_proj = nn.Linear(model_channels[0], config.out_channels)

    def __call__(self, feats, coords, shape, encoder_feature):
        point_cloud_h, point_cloud_w = shape[1], shape[2]
        encoder_h, encoder_w = encoder_feature.shape[1], encoder_feature.shape[2]
        assert (
            point_cloud_h == encoder_h * self.encoder_downsample
            and point_cloud_w == encoder_w * self.encoder_downsample
        ), "point cloud and encoder feature resolutions are inconsistent"

        feats = self.input_proj(feats)

        num_levels = len(self.down_stages)
        num_transitions = len(self.downsample_blocks)
        # Per-level conv neighbor maps, built on the down pass and reused by
        # the bottleneck and the symmetric up pass.
        level_conv_maps: List[Optional[mx.array]] = [None] * num_levels
        # Per-transition parent indices, reused by the symmetric upsample.
        level_parent_idx: List[Optional[mx.array]] = [None] * num_transitions
        skip_features: List = [None] * num_transitions

        for i, stage in enumerate(self.down_stages):
            conv_map = None
            for block in stage:
                feats, conv_map = block(feats, coords, shape, neighbor_map=conv_map)
            level_conv_maps[i] = conv_map
            if i < num_transitions:
                skip_features[i] = (feats, coords, shape)
                feats, coords, shape, parent_idx = self.downsample_blocks[i](
                    feats, coords, shape
                )
                level_parent_idx[i] = parent_idx

        enc_feat = encoder_feature[coords[:, 0], coords[:, 1], coords[:, 2]]
        enc_feat = self.encoder_fuse(enc_feat)
        feats = run_sequential(
            self.fuse_proj, mx.concatenate([feats, enc_feat], axis=-1)
        )

        conv_map = level_conv_maps[num_levels - 1]
        for block in self.bottleneck_stage:
            feats, conv_map = block(feats, coords, shape, neighbor_map=conv_map)

        for i, (upsample_block, stage) in enumerate(
            zip(self.upsample_blocks, self.up_stages)
        ):
            target_level = num_levels - 2 - i
            skip_feats, coords, shape = skip_features[target_level]
            feats, coords, shape = upsample_block(
                feats, level_parent_idx[target_level], coords, shape
            )
            feats = feats + skip_feats
            conv_map = level_conv_maps[target_level]
            for block in stage:
                feats, conv_map = block(feats, coords, shape, neighbor_map=conv_map)

        return self.out_proj(feats)
