import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig
from mlx_vlm.models.flux2.vae.common.unet_mid_block import Flux2UNetMidBlock2D
from mlx_vlm.models.flux2.vae.encoder.conv_in import Flux2ConvIn
from mlx_vlm.models.flux2.vae.encoder.conv_norm_out import Flux2ConvNormOut
from mlx_vlm.models.flux2.vae.encoder.conv_out import Flux2ConvOut
from mlx_vlm.models.flux2.vae.encoder.down_encoder_block import Flux2DownEncoderBlock2D


class Flux2Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.conv_in = Flux2ConvIn(
            in_channels=in_channels, out_channels=block_out_channels[0]
        )
        self.down_blocks = []
        for i, output_channel in enumerate(block_out_channels):
            input_channel = (
                block_out_channels[i - 1] if i > 0 else block_out_channels[0]
            )
            is_final_block = i == len(block_out_channels) - 1
            self.down_blocks.append(
                Flux2DownEncoderBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block,
                    eps=eps,
                    groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    downsample_padding=0,
                )
            )

        self.mid_block = Flux2UNetMidBlock2D(
            channels=block_out_channels[-1],
            eps=eps,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )
        self.conv_norm_out = Flux2ConvNormOut(
            num_channels=block_out_channels[-1],
            num_groups=norm_num_groups,
            eps=eps,
        )
        self.conv_out = Flux2ConvOut(
            in_channels=block_out_channels[-1], out_channels=2 * out_channels
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = nn.silu(hidden_states).astype(ModelConfig.precision)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
