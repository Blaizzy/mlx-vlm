import mlx.core as mx
from mlx import nn


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int = 2000, axes_dim: tuple[int, ...] = (32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        cos_out = []
        sin_out = []
        pos = ids.astype(mx.float32)
        for i, dim in enumerate(self.axes_dim):
            cos, sin = self._get_1d_rope(dim, pos[..., i])
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = mx.concatenate(cos_out, axis=-1)
        freqs_sin = mx.concatenate(sin_out, axis=-1)
        return freqs_cos, freqs_sin

    def _get_1d_rope(self, dim: int, pos: mx.array) -> tuple[mx.array, mx.array]:
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (self.theta**scale)
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(omega, axis=0)
        out = pos_expanded * omega_expanded
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        return cos_out, sin_out
