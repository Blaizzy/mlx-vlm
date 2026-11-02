"""Gaussian activation and standard 3DGS PLY export (Idol SDK, Apache-2.0)."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class Gaussian:
    def __init__(
        self,
        raw_xyz,
        features_dc,
        raw_scaling,
        raw_rotation,
        raw_opacity,
        scaling_bias,
        opacity_bias,
        minimum_kernel_size,
    ):
        self.xyz = raw_xyz - 0.5
        self.features_dc = features_dc
        scaling_offset = scaling_bias + np.log(-np.expm1(-scaling_bias))
        scaling = nn.softplus(raw_scaling + float(scaling_offset))
        self.scaling = mx.sqrt(scaling * scaling + minimum_kernel_size**2)
        rotation = raw_rotation + mx.array([1.0, 0.0, 0.0, 0.0])
        self.rotation = rotation / mx.maximum(
            mx.linalg.norm(rotation, axis=-1, keepdims=True), 1e-12
        )
        self.opacity = mx.sigmoid(
            raw_opacity + float(np.log(opacity_bias / (1 - opacity_bias)))
        )

    def save_ply(self, path):
        """Write a binary little-endian PLY, rotating the z-up object to y-up."""
        xyz, features, scaling, rotation, opacity = [
            np.array(x.astype(mx.float32))
            for x in (
                self.xyz,
                self.features_dc,
                self.scaling,
                self.rotation,
                self.opacity,
            )
        ]
        xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
        w, x, y, z = rotation.T
        rotation = np.stack([w - x, w + x, y - z, y + z], axis=1) * np.sqrt(0.5)
        opacity = np.clip(opacity, 1e-6, 1 - 1e-6)
        attributes = np.concatenate(
            [
                xyz,
                np.zeros_like(xyz),
                features.reshape(-1, 3),
                np.log(opacity / (1 - opacity)),
                np.log(np.maximum(scaling, 1e-6)),
                rotation,
            ],
            axis=1,
        ).astype("<f4")
        if not np.isfinite(attributes).all():
            raise ValueError("Cannot export non-finite Gaussian attributes")
        names = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        header = "ply\nformat binary_little_endian 1.0\nelement vertex {}\n{}\nend_header\n".format(
            len(xyz), "\n".join(f"property float {name}" for name in names)
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.write(header.encode("ascii"))
            handle.write(attributes.tobytes())
