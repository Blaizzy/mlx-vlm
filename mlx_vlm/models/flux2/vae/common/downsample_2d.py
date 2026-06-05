import mlx.core as mx
from mlx import nn


class Flux2Downsample2D(nn.Module):
    def __init__(self, channels: int, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=padding
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.pad(hidden_states, ((0, 0), (0, 0), (0, 1), (0, 1)))
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        hidden_states = self.conv(hidden_states)
        return mx.transpose(hidden_states, (0, 3, 1, 2))
