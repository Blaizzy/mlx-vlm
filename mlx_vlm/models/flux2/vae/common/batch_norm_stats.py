import mlx.core as mx
from mlx import nn


class Flux2BatchNormStats(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-4, momentum: float = 0.1):
        super().__init__()
        self.running_mean = mx.zeros((num_features,), dtype=mx.float32)
        self.running_var = mx.ones((num_features,), dtype=mx.float32)
        self.eps = eps
        self.momentum = momentum
