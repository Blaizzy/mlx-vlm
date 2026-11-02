"""Octree and Gaussian decoders, adapted from speridlabs/idol-sdk (Apache-2.0)."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gaussian import Gaussian
from .layers import (
    ModulatedTransformerCrossOnlyBlock,
    PcdAbsolutePositionEmbedderv2,
    SinusoidalEmbedder,
    TransformerCrossBlock,
    sanitize_weights,
)


def systematic_sample(probs, counts, offsets=None):
    """Split integer counts across bins with one uniform offset per row.

    CDF rounding gives the same systematic resampling distribution as searching
    every sample, without allocating a grid proportional to the point count.
    ``offsets`` in [0, 1) can be supplied for reproducible reference tests.
    """
    probs = mx.maximum(probs.astype(mx.float32), 0)
    total = mx.sum(probs, axis=-1, keepdims=True)
    probs = mx.where(total > 0, probs / mx.maximum(total, 1e-20), 1 / probs.shape[-1])
    cdf = mx.minimum(mx.cumsum(probs, axis=-1), 1)
    if offsets is None:
        offsets = mx.random.uniform(shape=counts.shape)
    cumulative = mx.clip(
        mx.floor(cdf * counts[..., None] - offsets[..., None]) + 1,
        0,
        counts[..., None],
    ).astype(mx.int32)
    cumulative = mx.concatenate(
        [mx.zeros_like(cumulative[..., :1]), cumulative[..., :-1], counts[..., None]],
        axis=-1,
    )
    return cumulative[..., 1:] - cumulative[..., :-1]


class OctreeProbabilityFixedlenDecoder(nn.Module):
    def __init__(
        self, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio=4.0
    ):
        super().__init__()
        self.dtype = mx.float16
        self.in_proj = nn.Linear(3, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(model_channels)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.l_embedder = SinusoidalEmbedder(
            model_channels, max_period=1024, two_pi=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 6 * model_channels)
        )
        self.blocks = [
            ModulatedTransformerCrossOnlyBlock(
                model_channels, cond_channels, num_heads, mlp_ratio
            )
            for _ in range(num_blocks)
        ]
        self.out_proj = nn.Linear(model_channels, 8)

    def sanitize(self, weights):
        return sanitize_weights(self, weights, transformer_dtype=self.dtype)

    def __call__(self, x, level, cond):
        h = self.input_layer(self.in_proj(x) + self.pos_embedder(x))
        mod = self.adaLN_modulation(self.l_embedder(level)).astype(self.dtype)
        out, context = h.astype(self.dtype), cond.astype(self.dtype)
        for block in self.blocks:
            out = block(out, mod, context)
        return self.out_proj(
            mx.fast.layer_norm(out.astype(cond.dtype), None, None, eps=1e-5)
        )

    @staticmethod
    def sample(model, cond, num_points, level):
        if num_points < 1 or level < 1:
            raise ValueError("num_points and level must be positive")
        # Variable-length packing is a CPU operation once per octree level.
        # All neural-network inference and random sampling remain on MLX.
        batch = cond.shape[0]
        coords = np.zeros((batch, 1, 3), dtype=np.int32)
        counts = np.full((batch, 1), num_points, dtype=np.int32)
        children = np.array(
            [[i, j, k] for k in range(2) for j in range(2) for i in range(2)],
            dtype=np.int32,
        )
        for depth in range(1, level + 1):
            centers = (mx.array(coords).astype(mx.float32) + 0.5) / (1 << (depth - 1))
            probs = mx.softmax(
                model(centers, mx.full((batch,), 1 << depth), cond), axis=-1
            )
            sampled = np.array(systematic_sample(probs, mx.array(counts))).reshape(
                batch, -1
            )
            child_coords = (coords[:, :, None] * 2 + children).reshape(batch, -1, 3)
            valid = sampled > 0
            max_valid = int(valid.sum(axis=1).max())
            coords = np.zeros((batch, max_valid, 3), dtype=np.int32)
            counts = np.zeros((batch, max_valid), dtype=np.int32)
            for i in range(batch):
                n = int(valid[i].sum())
                coords[i, :n] = child_coords[i, valid[i]]
                counts[i, :n] = sampled[i, valid[i]]
        points = np.stack([np.repeat(c, n, axis=0) for c, n in zip(coords, counts)])
        return (
            mx.array(points).astype(mx.float32) + mx.random.uniform(shape=points.shape)
        ) / (1 << level)


class GaussianFixedlenDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        cond_channels,
        num_blocks,
        num_heads,
        representation_config,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.dtype = mx.float16
        self.rep_config = representation_config
        self.layout = {
            "_xyz": (3,),
            "_features_dc": (1, 3),
            "_scaling": (3,),
            "_rotation": (4,),
            "_opacity": (1,),
            "_offset_scale": (1,),
        }
        self.ranges = {}
        start = 0
        per_anchor = representation_config["num_gaussians"]
        for name, shape in self.layout.items():
            size = per_anchor * math.prod(shape)
            self.ranges[name] = (start, start + size)
            start += size
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderv2(model_channels)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.blocks = [
            TransformerCrossBlock(model_channels, cond_channels, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ]
        self.out_proj = nn.Linear(model_channels, start)
        self.points_offset_perturbation = mx.zeros((per_anchor, 3))
        self.base_offset_scale = mx.array(0.0)

    def sanitize(self, weights):
        return sanitize_weights(self, weights, transformer_dtype=self.dtype)

    def __call__(self, points, cond):
        h = self.input_layer(self.in_proj(points) + self.pos_embedder(points))
        out, context = h.astype(self.dtype), cond.astype(self.dtype)
        for block in self.blocks:
            out = block(out, context)
        return self.out_proj(
            mx.fast.layer_norm(out.astype(cond.dtype), None, None, eps=1e-5)
        )

    def _slice(self, features, name):
        start, end = self.ranges[name]
        return features[:, :, start:end].reshape(
            features.shape[0], -1, self.rep_config["num_gaussians"], *self.layout[name]
        )

    def to_representation(self, points, features):
        config = self.rep_config
        offset_scale = nn.softplus(
            self._slice(features, "_offset_scale") + self.base_offset_scale
        )
        offset = (
            self._slice(features, "_xyz") * config["lr"]["_xyz"]
            + self.points_offset_perturbation
        )
        xyz = (
            mx.tanh(offset) * (0.5 * config["perturbe_size"]) * offset_scale
            + points[:, :, None]
        )
        result = []
        for i in range(features.shape[0]):
            raw = {
                name: (self._slice(features, name)[i] * config["lr"][name]).reshape(
                    -1, *self.layout[name]
                )
                for name in ("_features_dc", "_scaling", "_rotation", "_opacity")
            }
            result.append(
                Gaussian(
                    xyz[i].reshape(-1, 3),
                    raw["_features_dc"],
                    raw["_scaling"],
                    raw["_rotation"],
                    raw["_opacity"],
                    config["scaling_bias"],
                    config["opacity_bias"],
                    config["3d_filter_kernel_size"],
                )
            )
        return result
