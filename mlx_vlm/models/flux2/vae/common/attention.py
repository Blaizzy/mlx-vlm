import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mlx_vlm.models.flux2.constants import ModelConfig


class Flux2AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(
            num_groups=groups, dims=channels, eps=eps, pytorch_compatible=True
        )
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1))
        batch, height, width, channels = hidden_states.shape

        normed = self.group_norm(hidden_states.astype(mx.float32)).astype(
            ModelConfig.precision
        )
        q = self.to_q(normed).reshape(batch, height * width, 1, channels)
        k = self.to_k(normed).reshape(batch, height * width, 1, channels)
        v = self.to_v(normed).reshape(batch, height * width, 1, channels)

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scale = 1 / mx.sqrt(q.shape[-1])
        attended = scaled_dot_product_attention(q, k, v, scale=scale)
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch, height, width, channels
        )
        attended = self.to_out(attended)

        hidden_states = hidden_states + attended
        return mx.transpose(hidden_states, (0, 3, 1, 2))
