import mlx.core as mx
from mlx import nn


class Flux2Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.repeat(hidden_states, 2, axis=2)
        hidden_states = mx.repeat(hidden_states, 2, axis=3)
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
