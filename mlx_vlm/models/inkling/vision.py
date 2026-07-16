import math
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import VisionConfig


def _prime_factors(number: int) -> List[int]:
    factors = []
    while number % 2 == 0:
        factors.append(2)
        number //= 2
    p = 3
    while p * p <= number:
        while number % p == 0:
            factors.append(p)
            number //= p
        p += 2
    if number > 1:
        factors.append(number)
    return factors


def _plan_out_scales(
    temporal_patch_size: int, patch_size: int, n_layers: int, n_channels: int
) -> np.ndarray:
    """Plan the (t, h, w, c) fold ratios for each layer of the hMLP encoder.

    Port of transformers' `plan_out_scales`. Only the *ratios* between
    consecutive planned rows (and each row's channel width) matter: the real
    activation tensor starts at `(temporal_patch_size, patch_size, patch_size,
    num_channels)` and is folded down to `(1, 1, 1, hidden)` by the time the
    last layer runs; the absolute (t, h, w) numbers in `scales` are a planning
    device, not a shape assertion on the real tensor.
    """
    h = np.cumprod(np.array(_prime_factors(patch_size)[::-1], dtype=np.int64))
    t = np.cumprod(np.array(_prime_factors(temporal_patch_size)[::-1], dtype=np.int64))

    h_ch = np.ceil(h**2 * n_channels / 64).astype(np.int64) * 64
    t_ch = np.ceil(h[-1] ** 2 * n_channels * t).astype(np.int64) * 64

    base = np.array([[1, 1, 1, n_channels]], dtype=np.int64)
    spatial = np.stack([np.ones_like(h), h, h, h_ch], axis=1)
    temporal = np.stack(
        [t, np.full_like(t, h[-1]), np.full_like(t, h[-1]), t_ch], axis=1
    )
    scales = np.concatenate([base, spatial, temporal], axis=0)

    size_reduction = np.prod(scales[:, :-1], axis=1).astype(np.float64)

    total_elements = patch_size * patch_size * temporal_patch_size * n_channels
    log_ideal_scales = np.linspace(0, math.log(total_elements), n_layers + 1)
    cost_matrix = np.abs(log_ideal_scales[:, None] - np.log(size_reduction)[None, :])

    if n_layers >= scales.shape[0]:
        idxs = np.argmin(cost_matrix, axis=1)
    else:
        from scipy.optimize import linear_sum_assignment

        _, idxs = linear_sum_assignment(cost_matrix)

    idxs = np.array(idxs)
    idxs[0] = 0
    idxs[-1] = scales.shape[0] - 1
    return scales[idxs]


class InklingVisionEncoderLayer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, t_fold: int, hw_fold: int, add_norm: bool
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.add_norm = add_norm
        if add_norm:
            self.layer_norm = nn.RMSNorm(output_dim)
        self.hw_fold = hw_fold
        self.t_fold = t_fold

    def _fold_timespace_to_depth(self, x: mx.array) -> mx.array:
        """(B, T, H, W, C) -> (B, T//t, H//hw, W//hw, C * t * hw**2)."""
        B, T, H, W, C = x.shape
        t_new = T // self.t_fold
        h_new = H // self.hw_fold
        w_new = W // self.hw_fold

        x = x.reshape(B, t_new, self.t_fold, h_new, self.hw_fold, w_new, self.hw_fold, C)
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
        x = x.reshape(B, t_new, h_new, w_new, self.t_fold * self.hw_fold * self.hw_fold * C)
        return x

    def __call__(self, x: mx.array) -> mx.array:
        if self.hw_fold > 1 or self.t_fold > 1:
            x = self._fold_timespace_to_depth(x)

        x = self.projection(x)
        if self.add_norm:
            x = self.layer_norm(x)
            x = nn.gelu(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        n_layers = config.num_hidden_layers
        scales = _plan_out_scales(
            config.temporal_patch_size,
            config.patch_size,
            n_layers,
            config.num_channels,
        )

        self.encoder_layers = []
        for i in range(n_layers):
            start_scale = scales[i]
            end_scale = scales[i + 1]
            t_fold = int(end_scale[0] // start_scale[0])
            hw_fold = int(end_scale[1] // start_scale[1])
            shuffle_mult = t_fold * hw_fold * hw_fold
            output_dim = (
                config.text_hidden_size if i == n_layers - 1 else int(end_scale[3])
            )
            self.encoder_layers.append(
                InklingVisionEncoderLayer(
                    input_dim=int(start_scale[3]) * shuffle_mult,
                    output_dim=output_dim,
                    hw_fold=hw_fold,
                    t_fold=t_fold,
                    add_norm=i != n_layers - 1,
                )
            )
        self.final_norm = nn.RMSNorm(config.text_hidden_size)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        num_patches = pixel_values.shape[0]
        h = pixel_values
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.final_norm(h)
        return h.reshape(num_patches, -1)
