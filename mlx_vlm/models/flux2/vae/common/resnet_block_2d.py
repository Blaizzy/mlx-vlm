import mlx.core as mx
from mlx import nn

from mlx_vlm.models.flux2.constants import ModelConfig


class Flux2ResnetBlock2D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, eps: float = 1e-6, groups: int = 32
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(
            num_groups=groups, dims=in_channels, eps=eps, pytorch_compatible=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups, dims=out_channels, eps=eps, pytorch_compatible=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = mx.transpose(hidden_states, (0, 2, 3, 1))

        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.norm1(hidden_states.astype(mx.float32)).astype(
            ModelConfig.precision
        )
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states.astype(mx.float32)).astype(
            ModelConfig.precision
        )
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return mx.transpose(hidden_states, (0, 3, 1, 2))
