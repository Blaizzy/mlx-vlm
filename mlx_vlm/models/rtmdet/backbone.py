"""CSPNeXt backbone used by RTMDet.

All layers follow mmdetection's naming — `stem`, `stage1..4`, `conv`/`bn`/`blocks`/
`attention` — so the saved checkpoint can be loaded key-for-key after the usual
Conv2d / depthwise transposes applied in Model.sanitize.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn


class ConvBN(nn.Module):
    """Standard Conv + BN + SiLU, named `conv`/`bn` to match MMDet ConvModule."""

    def __init__(self, c_in: int, c_out: int, kernel: int, stride: int = 1,
                 padding: int = None, groups: int = 1, use_silu: bool = True):
        super().__init__()
        if padding is None:
            padding = (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out,
            kernel_size=kernel, stride=stride, padding=padding,
            groups=groups, bias=False,
        )
        self.bn = nn.BatchNorm(c_out, eps=1e-5, momentum=0.03)
        self.use_silu = use_silu

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_silu:
            x = nn.silu(x)
        return x


class DWConvBN(nn.Module):
    """Depthwise-separable Conv + BN + SiLU pair, named depthwise_conv / pointwise_conv."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 5):
        super().__init__()
        # Depthwise: groups = c_in, c_out == c_in
        self.depthwise_conv = ConvBN(c_in, c_in, kernel=kernel, stride=1, groups=c_in)
        self.pointwise_conv = ConvBN(c_in, c_out, kernel=1, stride=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    """Residual block: x + conv2(conv1(x)), where conv2 is depthwise-separable."""

    def __init__(self, c_in: int, c_out: int, add_identity: bool = True):
        super().__init__()
        self.conv1 = ConvBN(c_in, c_out, kernel=3, stride=1)
        self.conv2 = DWConvBN(c_out, c_out, kernel=5)
        self.add_identity = add_identity and (c_in == c_out)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv2(self.conv1(x))
        if self.add_identity:
            y = y + x
        return y


class ChannelAttention(nn.Module):
    """Global-avg-pool → 1x1 conv → Hardsigmoid gate (matches mmdet's CSPLayer attention)."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C)  →  pool to (B, 1, 1, C)
        pooled = mx.mean(x, axis=(1, 2), keepdims=True)
        # MLX 0.31 has no nn.hardsigmoid: clip((x+3)/6, 0, 1)
        y = self.fc(pooled)
        gate = mx.clip((y + 3.0) / 6.0, 0.0, 1.0)
        return x * gate


class CSPLayer(nn.Module):
    """Cross-Stage-Partial layer: split → N residual blocks on one branch → concat → fuse.

    main_conv  : 1x1, 2*c_out → channels on main branch
    short_conv : 1x1, 2*c_out → channels on shortcut branch
    blocks     : list of CSPNeXtBlock applied to the main branch
    final_conv : 1x1, (main+short) → c_out
    attention  : channel attention applied on fused output
    """

    def __init__(self, c_in: int, c_out: int, num_blocks: int,
                 add_identity: bool = True, use_attention: bool = True):
        super().__init__()
        half = c_out // 2
        self.main_conv = ConvBN(c_in, half, kernel=1, stride=1)
        self.short_conv = ConvBN(c_in, half, kernel=1, stride=1)
        self.blocks = [CSPNeXtBlock(half, half, add_identity=add_identity)
                       for _ in range(num_blocks)]
        self.final_conv = ConvBN(c_out, c_out, kernel=1, stride=1)
        if use_attention:
            self.attention = ChannelAttention(c_out)
        else:
            self.attention = None

    def __call__(self, x: mx.array) -> mx.array:
        main = self.main_conv(x)
        for blk in self.blocks:
            main = blk(main)
        short = self.short_conv(x)
        y = mx.concatenate([main, short], axis=-1)  # channels-last
        y = self.final_conv(y)
        if self.attention is not None:
            y = self.attention(y)
        return y


class SPPBottleneck(nn.Module):
    """SPP with 5x5 / 9x9 / 13x13 max-pools in parallel, concatenated with the
    1x1-reduced input, then 1x1-fused back to `c_out`.

    Mmdet stores this as `stageN.2` (when present) with children
    conv1 (reduce, 1x1) / poolings / conv2 (fuse, 1x1)."""

    def __init__(self, c_in: int, c_out: int, kernels: List[int] = (5, 9, 13)):
        super().__init__()
        mid = c_in // 2
        self.conv1 = ConvBN(c_in, mid, kernel=1, stride=1)
        self.poolings = [
            lambda y, k=k: max_pool_same(y, k)
            for k in kernels
        ]
        self.conv2 = ConvBN(mid * (len(kernels) + 1), c_out, kernel=1, stride=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        ys = [x] + [p(x) for p in self.poolings]
        return self.conv2(mx.concatenate(ys, axis=-1))


def max_pool_same(x: mx.array, kernel: int) -> mx.array:
    """Max-pool with stride=1 and same-padding — channels-last."""
    pad = kernel // 2
    # MLX doesn't ship a channels-last MaxPool2d with padding out of the box
    # in older versions; emulate via pad + nn.MaxPool2d if available.  Here we
    # use a simple implementation via im2col / sliding windows through mx.
    # Pad spatially with -inf so edges behave correctly.
    x_pad = mx.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   mode="constant", constant_values=mx.array(-1e9, dtype=x.dtype))
    # sliding window max via multiple roll + max
    out = x_pad
    B, H, W, C = x_pad.shape
    # Build kernel x kernel windows by gathering shifted copies and taking max
    acc = None
    for i in range(kernel):
        for j in range(kernel):
            sl = x_pad[:, i:i + (H - kernel + 1), j:j + (W - kernel + 1), :]
            acc = sl if acc is None else mx.maximum(acc, sl)
    return acc


class CSPNeXt(nn.Module):
    """CSPNeXt backbone: stem (3 ConvBN) + 4 stages.

    Each stage i has:
      stage{i}.0 : downsample ConvBN (stride 2)
      stage{i}.1 : CSPLayer
      (last stage only) stage{i}.2 : SPPBottleneck
      (last stage) stage{i}.(2 or 3) : another CSPLayer?

    Reading the dumped keys, the last stage (stage4) uses an SPPBottleneck but
    is stored as `stage4.2`, with `stage4.1` being the CSPLayer.  We replicate
    that structure.
    """

    def __init__(self, channels: List[int], num_blocks: List[int]):
        super().__init__()
        c0, c1, c2, c3, c4 = channels
        # stem: three ConvBN
        self.stem = [
            ConvBN(3,  c0 // 2, kernel=3, stride=2),
            ConvBN(c0 // 2, c0 // 2, kernel=3, stride=1),
            ConvBN(c0 // 2, c0,     kernel=3, stride=1),
        ]
        # Build 4 stages.  Each stage is a list:
        #   [downsample ConvBN s=2, CSPLayer, (optional) SPPBottleneck].
        # The last stage adds SPPBottleneck as index 2.
        self.stage1 = [
            ConvBN(c0, c1, kernel=3, stride=2),
            CSPLayer(c1, c1, num_blocks=num_blocks[0]),
        ]
        self.stage2 = [
            ConvBN(c1, c2, kernel=3, stride=2),
            CSPLayer(c2, c2, num_blocks=num_blocks[1]),
        ]
        self.stage3 = [
            ConvBN(c2, c3, kernel=3, stride=2),
            CSPLayer(c3, c3, num_blocks=num_blocks[2]),
        ]
        # stage4 inserts an SPPBottleneck between the downsample ConvBN and the
        # final CSPLayer (keys go .0 = downsample, .1 = SPP, .2 = CSPLayer).
        self.stage4 = [
            ConvBN(c3, c4, kernel=3, stride=2),
            SPPBottleneck(c4, c4),
            CSPLayer(c4, c4, num_blocks=num_blocks[3], use_attention=True),
        ]

    def __call__(self, x: mx.array):
        # stem
        for m in self.stem:
            x = m(x)
        # stage1 (feature map but usually not returned)
        for m in self.stage1: x = m(x)
        c1 = x
        for m in self.stage2: x = m(x)
        c2 = x  # P3 — stride 8
        for m in self.stage3: x = m(x)
        c3 = x  # P4 — stride 16
        for m in self.stage4: x = m(x)
        c4 = x  # P5 — stride 32
        return c2, c3, c4
