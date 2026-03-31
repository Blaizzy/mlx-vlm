"""RF-DETR Segmentation Head."""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


class DepthwiseConvBlock(nn.Module):
    """ConvNeXt-style depthwise convolution block."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) channel-last."""
        residual = x
        x = self.dwconv(x)  # (B, H, W, C) depthwise conv
        x = self.norm(x)  # LayerNorm on channel dim
        x = nn.gelu(self.pwconv1(x))  # pointwise + activation
        return residual + x


class MLPBlock(nn.Module):
    """MLP block for query feature processing."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = [
            nn.Linear(dim, dim * 4),
            None,  # placeholder for GELU (applied manually)
            nn.Linear(dim * 4, dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, N, C)."""
        residual = x
        x = self.norm_in(x)
        x = self.layers[0](x)
        x = nn.gelu(x)
        x = self.layers[2](x)
        return residual + x


class SegmentationHead(nn.Module):
    """Segmentation head that produces per-query mask predictions."""

    def __init__(
        self,
        in_dim: int = 256,
        num_blocks: int = 4,
        bottleneck_ratio: int = 1,
        downsample_ratio: int = 4,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.interaction_dim = in_dim // bottleneck_ratio

        # Spatial feature processing blocks
        self.blocks = [DepthwiseConvBlock(in_dim) for _ in range(num_blocks)]

        # Projection layers
        self.spatial_features_proj = nn.Conv2d(
            in_dim, self.interaction_dim, kernel_size=1
        )
        self.query_features_block = MLPBlock(in_dim)
        self.query_features_proj = nn.Linear(in_dim, self.interaction_dim)

        # Learnable bias for mask logits
        self.bias = mx.zeros((1,))

    def __call__(
        self,
        spatial_features: mx.array,
        query_features: mx.array,
        image_size: Tuple[int, int],
    ) -> mx.array:
        """
        Args:
            spatial_features: (B, H, W, C) backbone output features (channel-last)
            query_features: (B, N, C) decoder output hidden states
            image_size: (H, W) original image dimensions
        Returns:
            mask_logits: (B, N, H', W') where H'=H//downsample_ratio
        """
        # Downsample spatial features
        target_h = image_size[0] // self.downsample_ratio
        target_w = image_size[1] // self.downsample_ratio
        sf = _interpolate_spatial(spatial_features, target_h, target_w)

        # Process through DepthwiseConvBlocks
        for block in self.blocks:
            sf = block(sf)

        # Project spatial features: (B, H', W', C) -> (B, H', W', interaction_dim)
        sf_proj = self.spatial_features_proj(sf)

        # Process and project query features: (B, N, C) -> (B, N, interaction_dim)
        qf = self.query_features_block(query_features)
        qf_proj = self.query_features_proj(qf)

        # Compute mask logits via einsum: (B, H', W', C) x (B, N, C) -> (B, N, H', W')
        # Transpose sf_proj to (B, C, H', W') for the einsum
        B, H, W, C = sf_proj.shape
        # einsum "bhwc,bnc->bnhw"
        mask_logits = mx.einsum("bhwc,bnc->bnhw", sf_proj, qf_proj)
        mask_logits = mask_logits + self.bias

        return mask_logits


def _interpolate_spatial(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear interpolation for spatial feature downsampling.

    Args:
        x: (B, H, W, C) channel-last input
        target_h, target_w: target spatial dimensions
    Returns:
        (B, target_h, target_w, C)
    """
    B, H, W, C = x.shape
    if H == target_h and W == target_w:
        return x

    # Simple bilinear interpolation via grid sampling
    y_coords = mx.linspace(0, H - 1, target_h)
    x_coords = mx.linspace(0, W - 1, target_w)

    yy = mx.broadcast_to(y_coords[:, None], (target_h, target_w))
    xx = mx.broadcast_to(x_coords[None, :], (target_h, target_w))

    y0 = mx.clip(mx.floor(yy).astype(mx.int32), 0, H - 1)
    y1 = mx.clip(y0 + 1, 0, H - 1)
    x0 = mx.clip(mx.floor(xx).astype(mx.int32), 0, W - 1)
    x1 = mx.clip(x0 + 1, 0, W - 1)

    fy = (yy - y0.astype(yy.dtype))[..., None]
    fx = (xx - x0.astype(xx.dtype))[..., None]

    val_00 = x[:, y0, x0, :]
    val_01 = x[:, y0, x1, :]
    val_10 = x[:, y1, x0, :]
    val_11 = x[:, y1, x1, :]

    return (
        val_00 * (1 - fy) * (1 - fx)
        + val_01 * (1 - fy) * fx
        + val_10 * fy * (1 - fx)
        + val_11 * fy * fx
    )
