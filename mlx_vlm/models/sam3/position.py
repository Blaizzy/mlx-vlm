"""Position encodings: Sinusoidal 2D and Rotary Position Embeddings for SAM3."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D position embedding (used in DETR encoder/decoder and memory encoder)."""

    def __init__(
        self,
        num_pos_feats: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) feature map in MLX channel-last format
        Returns:
            pos: (B, H, W, num_pos_feats*2) position encoding
        """
        B, H, W, _ = x.shape

        # Use cumsum-style 1-indexed positions matching HF
        # not_mask = all True -> cumsum gives [1, 2, ..., H] and [1, 2, ..., W]
        y_embed = mx.broadcast_to(
            (mx.arange(H) + 1).reshape(1, H, 1), (B, H, W)
        ).astype(mx.float32)
        x_embed = mx.broadcast_to(
            (mx.arange(W) + 1).reshape(1, 1, W), (B, H, W)
        ).astype(mx.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = mx.arange(self.num_pos_feats).astype(mx.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t  # (B, H, W, D)
        pos_y = y_embed[..., None] / dim_t  # (B, H, W, D)

        # Interleave sin/cos matching HF: stack([sin(even), cos(odd)]).flatten
        pos_x = mx.stack([mx.sin(pos_x[..., 0::2]), mx.cos(pos_x[..., 1::2])], axis=-1)
        pos_x = pos_x.reshape(*pos_x.shape[:-2], -1)
        pos_y = mx.stack([mx.sin(pos_y[..., 0::2]), mx.cos(pos_y[..., 1::2])], axis=-1)
        pos_y = pos_y.reshape(*pos_y.shape[:-2], -1)

        pos = mx.concatenate([pos_y, pos_x], axis=-1)  # (B, H, W, 2*D)
        return pos


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Compute 2D axial rotary position embeddings matching HF Sam3ViTRotaryEmbedding.

    Returns:
        cos: (end_x*end_y, dim) cosine embeddings
        sin: (end_x*end_y, dim) sine embeddings
    """
    # Frequencies: step by 4 (not 2) because we split dim into 4 parts: x_pair, y_pair
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 4).astype(mx.float32) / dim))

    # Grid positions (row-major: y changes with row, x changes with column)
    flat_idx = mx.arange(end_x * end_y)
    x_positions = (flat_idx % end_x).astype(mx.float32)
    y_positions = (flat_idx // end_x).astype(mx.float32)

    # Outer products: (N, dim//4) each
    freqs_x = x_positions[:, None] * freqs[None, :]
    freqs_y = y_positions[:, None] * freqs[None, :]

    # Concatenate x and y: (N, dim//2)
    inv_freq = mx.concatenate([freqs_x, freqs_y], axis=-1)

    # repeat_interleave(2): [f0, f0, f1, f1, ...] -> (N, dim)
    inv_freq = mx.stack([inv_freq, inv_freq], axis=-1).reshape(inv_freq.shape[0], -1)

    return mx.cos(inv_freq), mx.sin(inv_freq)


def rotate_pairwise(x: mx.array) -> mx.array:
    """Pairwise rotation: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)"""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    rotated = mx.stack([-x2, x1], axis=-1)
    return rotated.reshape(*rotated.shape[:-2], -1)


def apply_rotary_enc(
    xq: mx.array,
    xk: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply 2D rotary position encoding matching HF implementation.

    Formula: q_out = q * cos + rotate_pairwise(q) * sin

    Args:
        xq: (B, H, N, D) queries (already transposed for SDPA)
        xk: (B, H, N, D) keys
        cos: (N, D) cosine embeddings
        sin: (N, D) sine embeddings
    Returns:
        xq_out, xk_out: rotated queries and keys
    """
    xq_out = xq * cos + rotate_pairwise(xq) * sin
    xk_out = xk * cos + rotate_pairwise(xk) * sin
    return xq_out, xk_out


def apply_rotary_enc_1d(
    xq: mx.array,
    xk: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    repeat_freqs_k: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Apply 1D RoPE for tracker memory attention (RoPEAttention).

    Args:
        xq: (B, N_q, H, D) queries
        xk: (B, N_k, H, D) keys
        freqs_cos: (N, D//2) cosine frequencies
        freqs_sin: (N, D//2) sine frequencies
        repeat_freqs_k: if True, tile freqs to match key length
    Returns:
        xq_out, xk_out
    """
    # Reshape cos/sin: (1, N, 1, D//2)
    cos_q = freqs_cos[None, : xq.shape[1], None, :]
    sin_q = freqs_sin[None, : xq.shape[1], None, :]

    if repeat_freqs_k:
        N_k = xk.shape[1]
        N_f = freqs_cos.shape[0]
        repeats = (N_k + N_f - 1) // N_f
        cos_k = mx.tile(freqs_cos, (repeats, 1))[None, :N_k, None, :]
        sin_k = mx.tile(freqs_sin, (repeats, 1))[None, :N_k, None, :]
    else:
        cos_k = freqs_cos[None, : xk.shape[1], None, :]
        sin_k = freqs_sin[None, : xk.shape[1], None, :]

    xq_r, xq_i = xq[..., 0::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., 0::2], xk[..., 1::2]

    xq_out_r = xq_r * cos_q - xq_i * sin_q
    xq_out_i = xq_r * sin_q + xq_i * cos_q
    xk_out_r = xk_r * cos_k - xk_i * sin_k
    xk_out_i = xk_r * sin_k + xk_i * cos_k

    # Interleave back using stack + reshape
    xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
    xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

    return xq_out, xk_out


def init_2d_freqs(
    dim: int,
    feat_h: int,
    feat_w: int,
    theta: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Initialize 2D RoPE frequencies for memory attention.

    Returns:
        freqs_cos: (feat_h*feat_w, dim//2)
        freqs_sin: (feat_h*feat_w, dim//2)
    """
    half = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, half, 2).astype(mx.float32) / half))

    t_y = mx.arange(feat_h).astype(mx.float32)
    t_x = mx.arange(feat_w).astype(mx.float32)

    grid_y, grid_x = mx.meshgrid(t_y, t_x, indexing="ij")
    grid_y = grid_y.reshape(-1)  # (H*W,)
    grid_x = grid_x.reshape(-1)  # (H*W,)

    freqs_y = mx.outer(grid_y, freqs)  # (H*W, half//2)
    freqs_x = mx.outer(grid_x, freqs)  # (H*W, half//2)

    # Interleave y and x frequencies
    freqs_all = mx.concatenate([freqs_y, freqs_x], axis=-1)  # (H*W, half)

    return mx.cos(freqs_all), mx.sin(freqs_all)
