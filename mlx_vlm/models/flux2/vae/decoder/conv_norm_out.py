import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig


class Flux2ConvNormOut(nn.GroupNorm):
    def __init__(self, channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__(
            num_groups=num_groups, dims=channels, eps=eps, pytorch_compatible=True
        )

    def __call__(self, input_array: mx.array) -> mx.array:
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        output = (
            super()
            .__call__(input_array.astype(mx.float32))
            .astype(ModelConfig.precision)
        )
        return mx.transpose(output, (0, 3, 1, 2))
