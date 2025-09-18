import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., dim]
        rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, in_features: int = 1536, hidden: int = 4224):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden, bias=False)  # up
        self.fc3 = nn.Linear(in_features, hidden, bias=False)  # gate
        self.fc2 = nn.Linear(hidden, in_features, bias=False)  # down

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(mx.silu(self.fc1(x)) * self.fc3(x))
