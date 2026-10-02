"""DPT dense head, pose head and scale head (channel-last)."""

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dpt import Scratch, reassemble_layers
from ..interpolate import resize_bilinear_nhwc
from .config import PredHeadConfig


class DPTHead(nn.Module):
    """DPT feature pyramid over four (N, h, w, C_i) token grids, regressed to
    ``out_channels`` maps at the image resolution."""

    def __init__(self, input_dims: List[int], config: PredHeadConfig):
        super().__init__()
        dims, features = config.layer_dims, config.feature_dim
        self.projects = [nn.Conv2d(c, d, 1) for c, d in zip(input_dims, dims)]
        self.resize_layers = reassemble_layers(dims)
        self.scratch = Scratch(dims, features)
        del self.scratch.refinenet4.resConfUnit1
        hidden = features // 2
        self.conv1 = nn.Conv2d(features, hidden, 3, padding=1)
        self.conv2 = [
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, config.dense_channels, 1),
        ]

    def __call__(self, features: List[mx.array], size: Tuple[int, int]) -> mx.array:
        layers = []
        for i, x in enumerate(features):
            x = self.resize_layers[i](self.projects[i](x))
            layers.append(getattr(self.scratch, f"layer{i + 1}_rn")(x))
        h3, w3 = layers[2].shape[1:3]
        path = self.scratch.refinenet4(layers[3])[:, :h3, :w3]
        path = self.scratch.refinenet3(path, layers[2])
        path = self.scratch.refinenet2(path, layers[1])
        path = self.scratch.refinenet1(path, layers[0])
        x = resize_bilinear_nhwc(self.conv1(path), size, align_corners=True)
        for layer in self.conv2:
            x = layer(x)
        return x


class ResConvBlock(nn.Module):
    """Residual stack of three 1x1 convolutions (as linears)."""

    def __init__(self, dim: int):
        super().__init__()
        self.res_conv1 = nn.Linear(dim, dim)
        self.res_conv2 = nn.Linear(dim, dim)
        self.res_conv3 = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.relu(self.res_conv1(x))
        y = nn.relu(self.res_conv2(y))
        return x + nn.relu(self.res_conv3(y))


class PoseHead(nn.Module):
    """Per-view camera translation (3) and rotation (4) from the token grid."""

    def __init__(self, input_dim: int, hidden: int, config: PredHeadConfig):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.res_conv = [ResConvBlock(hidden) for _ in range(config.num_resconv_block)]
        self.more_mlps = [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ]
        self.fc_t = nn.Linear(hidden, 3)
        self.fc_rot = nn.Linear(hidden, config.rot_representation_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (N, h, w, C) -> (N, 3 + rot_dim)."""
        x = self.proj(x)
        for block in self.res_conv:
            x = block(x)
        x = x.mean(axis=(1, 2))
        for layer in self.more_mlps:
            x = layer(x)
        return mx.concatenate([self.fc_t(x), self.fc_rot(x)], axis=-1)


class ScaleHead(nn.Module):
    def __init__(self, input_dim: int, config: PredHeadConfig):
        super().__init__()
        hidden = config.scale_hidden_dim
        self.proj = nn.Linear(input_dim, hidden)
        self.mlp = [
            nn.Linear(hidden, hidden) for _ in range(config.scale_num_mlp_layers)
        ]
        self.output_proj = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        for layer in self.mlp:
            x = nn.relu(layer(x))
        return self.output_proj(x)


def _bounded(x: mx.array, vmin: float, vmax: float) -> mx.array:
    if vmin == -math.inf and vmax == math.inf:
        return x
    return mx.clip(x, vmin, vmax)


def scale_value(x: mx.array, mode: str, vmin=-math.inf, vmax=math.inf) -> mx.array:
    """The ``linear`` / ``square`` / ``exp`` value adaptors (depth, scale)."""
    if mode == "square":
        x = x * x
    elif mode == "exp":
        x = mx.exp(x)
    elif mode != "linear":
        raise ValueError(f"MapAnything: unknown adaptor mode {mode!r}")
    return _bounded(x, vmin, vmax)


def ray_directions(x: mx.array, args: dict) -> mx.array:
    if args.get("ray_directions_mode", "linear") != "linear":
        raise ValueError("MapAnything: ray directions only support the linear mode")
    x = _bounded(
        x,
        args.get("ray_directions_vmin", -math.inf),
        args.get("ray_directions_vmax", math.inf),
    )
    if args.get("ray_directions_clamp_min_of_z_dir", False):
        z = mx.maximum(x[..., 2:3], args.get("ray_directions_z_dir_min", 1.0))
        x = mx.concatenate([x[..., :2], z], axis=-1)
    if args.get("ray_directions_normalize_to_unit_sphere", True):
        return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    if args.get("ray_directions_normalize_to_unit_image_plane", False):
        return x / x[..., 2:3]
    return x


def confidence(x: mx.array, args: dict) -> mx.array:
    kind = args.get("confidence_type", "exp")
    vmin, vmax = args.get("confidence_vmin", 1.0), args.get("confidence_vmax", math.inf)
    if kind == "exp":
        return vmin + mx.minimum(mx.exp(x), vmax - vmin)
    if kind == "sigmoid":
        return mx.sigmoid(x) * (vmax - vmin) + vmin
    raise ValueError(f"MapAnything: unknown confidence type {kind!r}")


def camera_translation(x: mx.array, args: dict) -> mx.array:
    mode = args.get("cam_trans_mode", "linear")
    if mode != "linear":
        d = mx.linalg.norm(x, axis=-1, keepdims=True)
        x = x / mx.maximum(d, 1e-8)
        if mode == "square":
            x = x * d * d
        elif mode == "exp":
            x = x * mx.expm1(d)
        else:
            raise ValueError(f"MapAnything: unknown translation mode {mode!r}")
    return _bounded(
        x, args.get("cam_trans_vmin", -math.inf), args.get("cam_trans_vmax", math.inf)
    )


def quaternions(x: mx.array, args: dict) -> mx.array:
    if args.get("quaternions_mode", "linear") != "linear":
        raise ValueError("MapAnything: quaternions only support the linear mode")
    x = _bounded(
        x,
        args.get("quaternions_vmin", -math.inf),
        args.get("quaternions_vmax", math.inf),
    )
    if args.get("quaternions_normalize", True):
        x = x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    return x
