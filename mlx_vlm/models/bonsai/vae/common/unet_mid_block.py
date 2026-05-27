import mlx.core as mx
from mlx import nn

from mlx_vlm.models.bonsai.vae.common.attention import Flux2AttentionBlock
from mlx_vlm.models.bonsai.vae.common.resnet_block_2d import Flux2ResnetBlock2D


class Flux2UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        groups: int = 32,
        add_attention: bool = True,
    ):
        super().__init__()
        self.resnets = [
            Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
            Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
        ]
        self.attentions = (
            [Flux2AttentionBlock(channels, groups=groups, eps=eps)]
            if add_attention
            else []
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states)
        if self.attentions:
            hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states
