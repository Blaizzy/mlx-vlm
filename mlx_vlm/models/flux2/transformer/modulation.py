import mlx.core as mx
from mlx import nn


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=False)

    def __call__(self, temb: mx.array):
        mod = nn.silu(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mx.expand_dims(mod, axis=1)
        mod_params = mx.split(mod, 3 * self.mod_param_sets, axis=-1)
        return tuple(
            mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets)
        )
