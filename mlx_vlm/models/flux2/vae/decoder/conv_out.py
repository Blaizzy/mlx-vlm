import mlx.core as mx
from mlx import nn


class Flux2ConvOut(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        output = super().__call__(input_array)
        return mx.transpose(output, (0, 3, 1, 2))
