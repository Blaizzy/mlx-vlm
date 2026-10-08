"""DPT reassemble and fusion blocks (channel-last), shared by the dense heads."""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .interpolate import resize_bilinear_nhwc


class PatchUpsample(nn.Module):
    """``ConvTranspose2d`` with kernel == stride (no overlap), run as one
    matmul plus a pixel shuffle. The weight keeps the MLX
    ``nn.ConvTranspose2d`` layout (out, k, k, in)."""

    def __init__(self, channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.weight = mx.zeros((channels, factor, factor, channels))
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        B, h, w, C = x.shape
        k = self.factor
        out = self.weight.shape[0]
        weight = self.weight.reshape(-1, C)
        y = (x @ weight.T).reshape(B, h, w, out, k, k)
        y = y.transpose(0, 1, 4, 2, 5, 3).reshape(B, h * k, w * k, out)
        return y + self.bias


def reassemble_layers(dims: List[int]) -> List[nn.Module]:
    """The four reassemble resamplings (4x, 2x, 1x, 1/2x) at ``dims`` channels."""
    return [
        PatchUpsample(dims[0], 4),
        PatchUpsample(dims[1], 2),
        nn.Identity(),
        nn.Conv2d(dims[3], dims[3], kernel_size=3, stride=2, padding=1),
    ]


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.conv1(nn.relu(x))
        out = self.conv2(nn.relu(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """Adds the skip features, refines, upsamples (2x or to ``size``) and
    projects, with bilinear ``align_corners=True`` resampling."""

    def __init__(self, features: int):
        super().__init__()
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0
        )
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def __call__(
        self,
        x: mx.array,
        res: Optional[mx.array] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> mx.array:
        output = x
        if res is not None:
            output = output + self.resConfUnit1(res)
        output = self.resConfUnit2(output)
        if size is None:
            size = (2 * output.shape[1], 2 * output.shape[2])
        output = resize_bilinear_nhwc(output, size, align_corners=True)
        return self.out_conv(output)


class Scratch(nn.Module):
    """The ``layer{i}_rn`` projections to ``features`` and the four
    ``refinenet{i}`` fusion blocks, under the reference ``scratch.*`` keys."""

    def __init__(self, layer_dims: List[int], features: int):
        super().__init__()
        for i, c in enumerate(layer_dims):
            setattr(
                self,
                f"layer{i + 1}_rn",
                nn.Conv2d(c, features, kernel_size=3, stride=1, padding=1, bias=False),
            )
        for i in range(4):
            setattr(self, f"refinenet{i + 1}", FeatureFusionBlock(features))
