import mlx.core as mx
from mlx import nn

from mlx_vlm.models.bonsai.vae.common.resnet_block_2d import Flux2ResnetBlock2D
from mlx_vlm.models.bonsai.vae.common.upsample_2d import Flux2Upsample2D


class Flux2UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        eps: float = 1e-6,
        groups: int = 32,
        add_upsample: bool = True,
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
        self.upsamplers = (
            [Flux2Upsample2D(out_channels, out_channels)] if add_upsample else []
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states
