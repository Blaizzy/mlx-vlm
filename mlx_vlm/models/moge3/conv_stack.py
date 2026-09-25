"""Convolutional stacks (neck and decoder heads) for MoGe-3.

Ports ``moge.model.modules.conv_stack`` and ``moge.model.modules.mlp`` to
channel-last MLX. Submodules are plain lists so parameter names match the
reference checkpoints exactly (``res_blocks.1.0.layers.2.weight``, ...).
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import ConvStackConfig, ScaleHeadConfig


def run_sequential(modules, x):
    for module in modules:
        x = module(x)
    return x


def _per_level(value, num_levels: int) -> list:
    """Broadcast a scalar config entry to one value per pyramid level."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * num_levels


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2),
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "group_norm":
        return nn.GroupNorm(max(1, channels // 32), channels, pytorch_compatible=True)
    if kind == "layer_norm":
        return nn.GroupNorm(1, channels, pytorch_compatible=True)
    if kind == "instance_norm":
        return nn.InstanceNorm(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {kind}")


def _replicate_pad_indices(height: int, width: int, padding: int) -> mx.array:
    """Flat int32 gather indices that replicate-pad an (H*W) plane by ``padding``.

    ``mx.pad(mode="edge")`` lowers to several full copies of the activation;
    a single clamped gather is bit-identical and 3-4x cheaper.
    """
    rows = mx.clip(mx.arange(-padding, height + padding), 0, height - 1)
    cols = mx.clip(mx.arange(-padding, width + padding), 0, width - 1)
    return (rows[:, None] * width + cols[None, :]).reshape(-1)


class Conv2dReplicate(nn.Module):
    """Conv2d with replicate (edge) padding, matching torch padding_mode.

    Stores the weight itself (MLX layout: out, kh, kw, in) so parameter names
    match the reference checkpoint.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        k = kernel_size
        scale = 1.0 / (in_channels * k * k) ** 0.5
        self.weight = mx.random.uniform(
            -scale, scale, (out_channels, k, k, in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x):
        p = self.padding
        if p:
            N, H, W, C = x.shape
            flat = _replicate_pad_indices(H, W, p)
            x = mx.take(x.reshape(N, H * W, C), flat, axis=1)
            x = x.reshape(N, H + 2 * p, W + 2 * p, C)
        return mx.conv2d(x, self.weight) + self.bias


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        kernel_size: int = 3,
        activation: str = "relu",
        in_norm: str = "layer_norm",
        hidden_norm: str = "group_norm",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        act = _ACTIVATIONS[activation]
        self.layers = [
            _make_norm(in_norm, in_channels),
            act(),
            Conv2dReplicate(in_channels, hidden_channels, kernel_size),
            _make_norm(hidden_norm, hidden_channels),
            act(),
            Conv2dReplicate(hidden_channels, out_channels, kernel_size),
        ]
        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def __call__(self, x):
        return run_sequential(self.layers, x) + self.skip_connection(x)


def make_resampler(in_channels: int, out_channels: int, type_: str):
    """Return the 2x resampler as a plain module list (names ``0``, ``1``)."""
    if type_ == "conv_transpose":
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            Conv2dReplicate(out_channels, out_channels, 3),
        ]
    if type_ == "bilinear":
        upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
    elif type_ == "nearest":
        upsample = nn.Upsample(scale_factor=2, mode="nearest")
    else:
        raise NotImplementedError(f"Unsupported resampler type: {type_}")
    return [upsample, Conv2dReplicate(in_channels, out_channels, 3)]


class ConvStack(nn.Module):
    def __init__(self, config: ConvStackConfig):
        super().__init__()
        dim_res = config.dim_res_blocks
        num_levels = len(dim_res)
        dim_in = _per_level(config.dim_in, num_levels)
        dim_out = _per_level(config.dim_out, num_levels)
        resamplers = _per_level(config.resamplers, num_levels - 1)
        num_res_blocks = _per_level(config.num_res_blocks, num_levels)

        self.input_blocks = [
            nn.Conv2d(d, r, kernel_size=1) if d is not None else nn.Identity()
            for d, r in zip(dim_in, dim_res)
        ]
        self.resamplers = [
            make_resampler(dim_prev, dim_succ, resampler)
            for dim_prev, dim_succ, resampler in zip(
                dim_res[:-1], dim_res[1:], resamplers
            )
        ]
        self.res_blocks = [
            [
                ResidualConvBlock(
                    d,
                    d,
                    config.dim_times_res_block_hidden * d,
                    activation=config.activation,
                    in_norm=config.res_block_in_norm,
                    hidden_norm=config.res_block_hidden_norm,
                )
                for _ in range(n)
            ]
            for d, n in zip(dim_res, num_res_blocks)
        ]
        self.output_blocks = [
            nn.Conv2d(r, d, kernel_size=1) if d is not None else nn.Identity()
            for d, r in zip(dim_out, dim_res)
        ]

    def __call__(self, in_features: List[mx.array]) -> List[mx.array]:
        out_features = []
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            x = feature if i == 0 else x + feature
            x = run_sequential(self.res_blocks[i], x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = run_sequential(self.resamplers[i], x)
        return out_features


def make_scale_head(config: ScaleHeadConfig) -> list:
    """MLP as a plain module list: Linear-ReLU pairs, then the final Linear."""
    dims = config.dims
    modules = [
        m
        for d_in, d_out in zip(dims[:-2], dims[1:-1])
        for m in (nn.Linear(d_in, d_out), nn.ReLU())
    ]
    modules.append(nn.Linear(dims[-2], dims[-1]))
    return modules
