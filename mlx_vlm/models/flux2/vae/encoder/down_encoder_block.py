import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.vae.common.downsample_2d import Flux2Downsample2D
from mlx_vlm.models.flux2.vae.common.resnet_block_2d import Flux2ResnetBlock2D


class Flux2DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        eps: float = 1e-6,
        groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                eps=eps,
                groups=groups,
            )
            for i in range(num_layers)
        ]
        self.downsamplers = (
            [Flux2Downsample2D(out_channels, padding=downsample_padding)]
            if add_downsample
            else []
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        return hidden_states
