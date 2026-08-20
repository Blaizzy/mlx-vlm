"""DINOv3 RoPE position encoding.

Uses split-half rotation (NOT pairwise) and separate coordinate normalization.
Periods are a learned buffer loaded from weights.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class DINOv3RoPE(nn.Module):
    """RoPE embedding for DINOv3 ViT.

    Weight key: rope_embed.periods (D_head // 4,)
    """

    def __init__(self, head_dim: int, base: float = 100.0):
        super().__init__()
        n_periods = head_dim // 4
        # initialized same as PyTorch; overwritten by weight loading
        periods = base ** (2.0 * mx.arange(n_periods) / (head_dim // 2))
        self.periods = periods  # (n_periods,)

    def __call__(self, H: int, W: int) -> tuple[mx.array, mx.array]:
        """Compute sin/cos tables for an H x W patch grid.

        Returns (sin, cos) each of shape (H*W, head_dim), float32.
        """
        # coords in [-1, +1]
        coords_h = (mx.arange(0.5, H) / H) * 2 - 1  # (H,)
        coords_w = (mx.arange(0.5, W) / W) * 2 - 1  # (W,)

        # meshgrid -> (H, W, 2)
        gh, gw = mx.meshgrid(coords_h, coords_w, indexing="ij")
        coords = mx.stack([gh, gw], axis=-1)  # (H, W, 2)
        coords = coords.reshape(-1, 2)  # (HW, 2)

        # angles: (HW, 2, n_periods) -> (HW, 2*n_periods) -> (HW, head_dim)
        periods = self.periods.astype(mx.float32)
        angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
        angles = angles.reshape(angles.shape[0], -1)  # (HW, D/2)
        angles = mx.concatenate([angles, angles], axis=-1)  # (HW, D)

        return mx.sin(angles), mx.cos(angles)


def rope_rotate_half(x: mx.array) -> mx.array:
    """Split-half rotation: [-x2, x1]."""
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(
    q: mx.array,
    k: mx.array,
    sin: mx.array,
    cos: mx.array,
    prefix: int,
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to q and k, skipping the first `prefix` tokens.

    q, k: (B, heads, N, D)
    sin, cos: (HW, D)  — patch tokens only
    prefix: number of non-patch tokens (CLS + storage)
    """
    # broadcast sin/cos to (1, 1, HW, D)
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    q_prefix = q[:, :, :prefix, :]
    k_prefix = k[:, :, :prefix, :]

    q_patch = q[:, :, prefix:, :] * cos + rope_rotate_half(q[:, :, prefix:, :]) * sin
    k_patch = k[:, :, prefix:, :] * cos + rope_rotate_half(k[:, :, prefix:, :]) * sin

    q = mx.concatenate([q_prefix, q_patch], axis=2)
    k = mx.concatenate([k_prefix, k_patch], axis=2)
    return q, k
