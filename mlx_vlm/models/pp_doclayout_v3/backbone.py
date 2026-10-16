"""HGNetV2 backbone for PP-DocLayoutV3.

Mirrors ``transformers.models.hgnet_v2.modeling_hgnet_v2`` (stem with
padded 2x2 branches, HG blocks with 1x1 aggregation, depthwise
downsampling) in NHWC MLX. BatchNorms run frozen at inference, exactly
like HF ``FrozenBatchNorm2d`` after ``replace_batch_norm``.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import HGNetV2Config


def _resolve_activation(name: Optional[str]):
    if name is None:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation {name!r}")


class HGConv(nn.Module):
    """Conv2d (no bias) + BatchNorm + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels, eps=eps)
        self.activation = _resolve_activation(activation)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class HGLightConv(nn.Module):
    """1x1 (no activation) + depthwise kxk + activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[str] = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.conv1 = HGConv(
            in_channels, out_channels, kernel_size=1, activation=None, eps=eps
        )
        self.conv2 = HGConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            activation=activation,
            eps=eps,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2(self.conv1(x))


class HGStem(nn.Module):
    """Stem: 3x3/s2, padded 2x2 branches (plus pooled shortcut), 3x3/s2, 1x1."""

    def __init__(self, config: HGNetV2Config, eps: float = 1e-5) -> None:
        super().__init__()
        stem = [3, 32, 48]
        strides = [2, 1, 1, 2, 1]
        act = config.hidden_act
        self.stem1 = HGConv(stem[0], stem[1], 3, strides[0], activation=act, eps=eps)
        self.stem2a = HGConv(
            stem[1], stem[1] // 2, 2, strides[1], activation=act, eps=eps
        )
        self.stem2b = HGConv(
            stem[1] // 2, stem[1], 2, strides[2], activation=act, eps=eps
        )
        self.stem3 = HGConv(
            stem[1] * 2, stem[1], 3, strides[3], activation=act, eps=eps
        )
        self.stem4 = HGConv(stem[1], stem[2], 1, strides[4], activation=act, eps=eps)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        e = self.stem1(x)
        e = mx.pad(e, [(0, 0), (0, 1), (0, 1), (0, 0)])
        a = self.stem2b(mx.pad(self.stem2a(e), [(0, 0), (0, 1), (0, 1), (0, 0)]))
        p = self.pool(e)
        return self.stem4(self.stem3(mx.concatenate([p, a], axis=-1)))


class HGBasicLayer(nn.Module):
    """HG block: chained convs, concat, 1x1 squeeze + excite, residual."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        layer_num: int = 6,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        activation: Optional[str] = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = []
        for i in range(layer_num):
            inch = in_channels if i == 0 else mid_channels
            if light_block:
                blk = HGLightConv(inch, mid_channels, kernel_size, activation, eps)
            else:
                blk = HGConv(inch, mid_channels, kernel_size, 1, 1, activation, eps)
            self.layers.append(blk)
        total = in_channels + layer_num * mid_channels
        self.aggregation = [
            HGConv(total, out_channels // 2, 1, 1, 1, activation, eps),
            HGConv(out_channels // 2, out_channels, 1, 1, 1, activation, eps),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        feats = [x]
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = mx.concatenate(feats, axis=-1)
        for agg in self.aggregation:
            x = agg(x)
        if self.residual:
            x = x + identity
        return x


class HGStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_blocks: int,
        num_layers: int,
        downsample: bool,
        downsample_stride: int,
        light_block: bool,
        kernel_size: int,
        activation: Optional[str] = "relu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if downsample:
            # Depthwise 3x3/s2, conv + BN only (no activation).
            self.downsample = HGConv(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=downsample_stride,
                groups=in_channels,
                activation=None,
                eps=eps,
            )
        else:
            self.downsample = None
        self.blocks = [
            HGBasicLayer(
                in_channels if i == 0 else out_channels,
                mid_channels,
                out_channels,
                num_layers,
                kernel_size,
                residual=(i != 0),
                light_block=light_block,
                activation=activation,
                eps=eps,
            )
            for i in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        if self.downsample is not None:
            x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGEncoder(nn.Module):
    """Stage container so flattened keys read ``encoder.stages.N...``."""

    def __init__(self, config: HGNetV2Config, eps: float = 1e-5) -> None:
        super().__init__()
        self.stages = [
            HGStage(
                config.stage_in_channels[i],
                config.stage_mid_channels[i],
                config.stage_out_channels[i],
                config.stage_num_blocks[i],
                config.stage_numb_of_layers[i],
                config.stage_downsample[i],
                config.stage_downsample_strides[i],
                config.stage_light_block[i],
                config.stage_kernel_size[i],
                config.hidden_act,
                eps,
            )
            for i in range(4)
        ]

    def __call__(self, x: mx.array) -> list:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class HGNetV2Backbone(nn.Module):
    """Stem (stride 4) + stages 1-4 (strides 4/8/16/32)."""

    def __init__(self, config: HGNetV2Config, eps: float = 1e-5) -> None:
        super().__init__()
        self.config = config
        self.embedder = HGStem(config, eps)
        self.encoder = HGEncoder(config, eps)

    def __call__(self, x: mx.array) -> list:
        return self.encoder(self.embedder(x))
