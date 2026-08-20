import mlx.core as mx
from mlx import nn


class Flux2SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return self.gate_fn(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 3.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_in(x)
        x = self.act(x)
        x = self.linear_out(x)
        return x
