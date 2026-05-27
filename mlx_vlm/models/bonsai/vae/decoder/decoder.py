import mlx.core as mx
from mlx import nn

from mlx_vlm.models.bonsai.constants import ModelConfig
from mlx_vlm.models.bonsai.vae.common.unet_mid_block import Flux2UNetMidBlock2D
from mlx_vlm.models.bonsai.vae.decoder.conv_in import Flux2ConvIn
from mlx_vlm.models.bonsai.vae.decoder.conv_norm_out import Flux2ConvNormOut
from mlx_vlm.models.bonsai.vae.decoder.conv_out import Flux2ConvOut
from mlx_vlm.models.bonsai.vae.decoder.up_decoder_block import Flux2UpDecoderBlock2D


class Flux2Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.conv_in = Flux2ConvIn(
            in_channels=in_channels, out_channels=block_out_channels[-1]
        )
        self.mid_block = Flux2UNetMidBlock2D(
            channels=block_out_channels[-1],
            eps=eps,
            groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )
        self.up_blocks = []
        reversed_channels = list(reversed(block_out_channels))
        for i, output_channel in enumerate(reversed_channels):
            prev_output_channel = output_channel if i == 0 else reversed_channels[i - 1]
            is_final_block = i == len(reversed_channels) - 1
            self.up_blocks.append(
                Flux2UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block + 1,
                    eps=eps,
                    groups=norm_num_groups,
                    add_upsample=not is_final_block,
                )
            )

        self.conv_norm_out = Flux2ConvNormOut(
            channels=block_out_channels[0], num_groups=norm_num_groups, eps=eps
        )
        self.conv_out = Flux2ConvOut(
            in_channels=block_out_channels[0], out_channels=out_channels
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = nn.silu(hidden_states).astype(ModelConfig.precision)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states
