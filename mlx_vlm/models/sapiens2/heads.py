"""Sapiens2 decode heads, all tensors channel-last."""

import math
from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn


class PixelShuffle(nn.Module):
    """Channel-last equivalent of ``nn.PixelShuffle``."""

    def __init__(self, upscale: int = 2):
        super().__init__()
        self.upscale = upscale

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        r = self.upscale
        C //= r * r
        x = x.reshape(B, H, W, C, r, r)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        return x.reshape(B, H * r, W * r, C)


class _Flatten(nn.Module):
    """Channel-first flatten of a channel-last map, matching torch Flatten
    on ``(B, C, H, W)``."""

    def __call__(self, x: mx.array) -> mx.array:
        return x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)


def _gemm(cols: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """(B, H, W, X) patch matrix -> (B, H, W, out) in one matmul."""
    B, H, W, X = cols.shape
    weight = weight.reshape(weight.shape[0], X)
    return mx.addmm(bias, cols.reshape(-1, X), weight.T).reshape(B, H, W, -1)


def _conv_weight(out_channels: int, k: int, in_channels: int) -> mx.array:
    """Random init in ``nn.Conv2d``'s (out, kh, kw, in) layout; checkpoints
    overwrite it."""
    scale = 1 / math.sqrt(in_channels * k * k)
    return mx.random.uniform(-scale, scale, (out_channels, k, k, in_channels))


class Conv1x1(nn.Module):
    """1x1 conv with ``nn.Conv2d``'s parameter layout, run as one matmul
    (faster than MLX's conv path here)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = _conv_weight(out_channels, 1, in_channels)
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return _gemm(x, self.weight, self.bias)


class Conv3x3(nn.Module):
    """3x3 stride-1 'same' conv with ``nn.Conv2d``'s parameter layout.

    Small maps run as im2col + one gemm, large maps as ``mx.conv2d`` —
    each path is faster in its regime for the head shapes.
    """

    max_im2col_per_image = 96 * 2**20

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = _conv_weight(out_channels, 3, in_channels)
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        if H * W * 9 * C > self.max_im2col_per_image:
            return mx.conv2d(x, self.weight, padding=1) + self.bias
        xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        cols = mx.concatenate(
            [xp[:, i : i + H, j : j + W, :] for i in range(3) for j in range(3)],
            axis=-1,
        )  # (B, H, W, 9C) in (kh, kw, in) order, matching the weight layout
        return _gemm(cols, self.weight, self.bias)


class DeconvUpsample(nn.Module):
    """ConvTranspose2d(kernel 4, stride 2, padding 1, no bias) with
    ``nn.ConvTranspose2d``'s (out, kh, kw, in) weight, run as one gemm.

    Each output phase (even/odd row, even/odd column) is a 2x2 conv over
    the 1-padded input with a fixed subset of the 4x4 kernel, so the whole
    layer is one gemm over 2x2 patches. Same arithmetic as the transposed
    conv, much faster than ``mx.conv_transpose2d`` for these shapes.
    """

    # Kernel rows used by output phase 0 / 1 along one axis, for taps
    # (i-1, i) and (i, i+1) of the input.
    _PHASE_TAPS = ((3, 1), (2, 0))

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = _conv_weight(out_channels, 4, in_channels)
        self._matmul_weight = None  # (source weight, gemm layout), built once

    def _gemm_weight(self) -> mx.array:
        weight = self.weight  # (out, 4, 4, in)
        cached = self._matmul_weight
        if cached is not None and cached[0] is weight:
            return cached[1]
        out_ch, _, _, in_ch = weight.shape
        taps = mx.array(self._PHASE_TAPS)
        phases = [weight[:, taps[a]][:, :, taps[b]] for a in (0, 1) for b in (0, 1)]
        # (phase, out, th, tw, in) -> rows (th, tw, in), cols (phase, out)
        gemm = mx.stack(phases).transpose(2, 3, 4, 0, 1).reshape(4 * in_ch, 4 * out_ch)
        self._matmul_weight = (weight, gemm)
        return gemm

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        out_ch = self.weight.shape[0]
        xp = mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)])
        cols = mx.concatenate(
            [
                xp[:, th : th + H + 1, tw : tw + W + 1, :]
                for th in (0, 1)
                for tw in (0, 1)
            ],
            axis=-1,
        )  # (B, H+1, W+1, 4C) patch layout (th, tw, in)
        y = cols.reshape(-1, 4 * C) @ self._gemm_weight()
        y = y.reshape(B, H + 1, W + 1, 2, 2, out_ch)
        # Phase (a, b) fills output rows 2i+a and columns 2j+b.
        y = mx.stack(
            [y[:, a : H + a, b : W + b, a, b] for a in (0, 1) for b in (0, 1)],
            axis=3,
        )
        y = y.reshape(B, H, W, 2, 2, out_ch).transpose(0, 1, 3, 2, 4, 5)
        return y.reshape(B, 2 * H, 2 * W, out_ch)


_DECONV_PADDING = {3: (1, 1), 2: (0, 0)}  # kernel -> (padding, output_padding)


def _deconv_block(in_ch: int, out_ch: int, k: int) -> list:
    """ConvTranspose2d(stride 2) + InstanceNorm + SiLU; padding rules of the original head."""
    if k == 4:
        deconv = DeconvUpsample(in_ch, out_ch)
    elif k in _DECONV_PADDING:
        padding, output_padding = _DECONV_PADDING[k]
        deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            k,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
    else:
        raise ValueError(f"Unsupported deconv kernel size {k}")
    return [deconv, nn.InstanceNorm(out_ch), nn.SiLU()]


def _conv(in_ch: int, out_ch: int, k: int, stride: int = 1) -> nn.Module:
    """Conv2d with 'same' padding; 1x1 stride-1 convs use the matmul path."""
    if k == 1 and stride == 1:
        return Conv1x1(in_ch, out_ch)
    if k == 3 and stride == 1:
        return Conv3x3(in_ch, out_ch)
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=(k - 1) // 2)


def _conv_block(in_ch: int, out_ch: int, k: int, stride: int = 1) -> list:
    """Conv2d + InstanceNorm + SiLU."""
    return [_conv(in_ch, out_ch, k, stride), nn.InstanceNorm(out_ch), nn.SiLU()]


def _run(layers: Sequence[nn.Module], x: mx.array) -> mx.array:
    for layer in layers:
        x = layer(x)
    return x


class DeconvHead(nn.Module):
    """Seg/pose head: ConvTranspose2d upsampling + optional convs + predictor."""

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        predictor_name: str,
        deconv_out_channels: Optional[Sequence[int]] = (256, 256, 256),
        deconv_kernel_sizes: Optional[Sequence[int]] = (4, 4, 4),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.predictor_name = predictor_name

        cur = in_channels
        deconv_layers: List[nn.Module] = []
        for out_ch, k in zip(deconv_out_channels or [], deconv_kernel_sizes or []):
            deconv_layers += _deconv_block(cur, out_ch, k)
            cur = out_ch
        self.deconv_layers = deconv_layers

        conv_layers: List[nn.Module] = []
        for out_ch, k in zip(conv_out_channels or [], conv_kernel_sizes or []):
            conv_layers += _conv_block(cur, out_ch, k)
            cur = out_ch
        self.conv_layers = conv_layers

        setattr(self, predictor_name, Conv1x1(cur, num_labels))

    def __call__(self, x: mx.array) -> mx.array:
        x = _run(self.deconv_layers, x)
        x = _run(self.conv_layers, x)
        return getattr(self, self.predictor_name)(x)


class PixelShuffleHead(nn.Module):
    """Normal/matting/pointmap head: Conv2d + PixelShuffle(2) upsampling
    blocks, optional convs, 1x1 predictor and an optional scale-regression
    branch (pointmap)."""

    def __init__(
        self,
        in_channels: int,
        num_labels: int,
        predictor_name: str,
        upsample_channels: Sequence[int] = (768, 384, 192, 96),
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        scale_conv_out_channels: Optional[Sequence[int]] = None,
        scale_conv_kernel_sizes: Optional[Sequence[int]] = None,
        scale_final_layer: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.predictor_name = predictor_name

        self.input_conv = _conv_block(in_channels, in_channels, 3)

        cur = in_channels
        self.upsample_blocks: List[list] = []
        for out_ch in upsample_channels:
            self.upsample_blocks.append(
                [
                    Conv3x3(cur, out_ch * 4),
                    PixelShuffle(2),
                    nn.InstanceNorm(out_ch),
                    nn.SiLU(),
                ]
            )
            cur = out_ch

        conv_layers: List[nn.Module] = []
        for out_ch, k in zip(conv_out_channels or [], conv_kernel_sizes or []):
            conv_layers += _conv_block(cur, out_ch, k)
            cur = out_ch
        self.conv_layers = conv_layers

        setattr(self, predictor_name, Conv1x1(cur, num_labels))

        # Pointmap scale regression branch (f_canonical / f_actual).
        if scale_conv_out_channels is not None:
            scale_layers: List[nn.Module] = []
            cur = in_channels
            for out_ch, k in zip(scale_conv_out_channels, scale_conv_kernel_sizes):
                scale_layers += _conv_block(cur, out_ch, k, stride=2)
                cur = out_ch
            self.scale_conv_layers = scale_layers

            # Flatten + Linear stack, SiLU after every Linear but the last.
            final: List[nn.Module] = [_Flatten()]
            dims = list(scale_final_layer)
            for i in range(1, len(dims)):
                final.append(nn.Linear(dims[i - 1], dims[i]))
                if i < len(dims) - 1:
                    final.append(nn.SiLU())
            self.scale_final_layer = final
        else:
            self.scale_conv_layers = None
            self.scale_final_layer = None

    def __call__(self, x: mx.array) -> Tuple[mx.array, Optional[mx.array]]:
        out = _run(self.input_conv, x)
        for block in self.upsample_blocks:
            out = _run(block, out)
        out = _run(self.conv_layers, out)
        out = getattr(self, self.predictor_name)(out)

        if self.scale_conv_layers is None:
            return out
        s = _run(self.scale_conv_layers, x)
        scale = _run(self.scale_final_layer, s)
        return out, scale
