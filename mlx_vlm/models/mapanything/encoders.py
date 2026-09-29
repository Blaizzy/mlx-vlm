"""Image and geometric-input encoders (channel-last)."""

from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..dinov2.dinov2 import DINOv2
from .config import EncoderConfig
from .layers import Block, layer_norm


class ImageEncoder(DINOv2):
    """torch.hub DINOv2, optionally cut to its first ``keep_first_n_layers``
    blocks and without the final norm (``norm_returned_features=False``)."""

    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        del self.mask_token
        self.blocks = [
            Block(
                config.embed_dim,
                config.num_heads,
                config.mlp_ratio,
                config.ffn,
                eps=config.layer_norm_eps,
            )
            for _ in range(config.depth)
        ]
        if not config.norm_returned_features:
            del self.norm

    def __call__(self, images: mx.array) -> Tuple[mx.array, mx.array]:
        """images: (B, H, W, 3) normalized -> float32 patch tokens (B, N, D)
        and the cls + register tokens (B, 1 + R, D)."""
        x = self.prepare_tokens(images.astype(self.patch_embed.proj.weight.dtype))
        for block in self.blocks:
            x = block(x)
        if "norm" in self:
            x = layer_norm(self.norm, x)
        prefix = 1 + self.num_register_tokens
        return x[:, prefix:], x[:, :prefix]


def pixel_unshuffle(x: mx.array, factor: int) -> mx.array:
    """(B, H, W, C) -> (B, H / r, W / r, C * r * r) in ``nn.PixelUnshuffle``'s
    channel order (c, i, j)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // factor, factor, W // factor, factor, C)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(B, H // factor, W // factor, C * factor * factor)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = self.shortcut(x) if "shortcut" in self else x
        out = self.conv2(nn.gelu(self.conv1(x)))
        return nn.gelu(out + identity)


class DenseRepresentationEncoder(nn.Module):
    """Patchifies a dense map (rays, depth) with a pixel unshuffle, residual
    convolutions and a LayerNorm."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, dims: List[int]):
        super().__init__()
        self.patch_size = patch_size
        self.conv_in = nn.Conv2d(in_chans * patch_size**2, dims[0], 3, padding=1)
        self.encoder = [ResidualBlock(a, b) for a, b in zip(dims[:-1], dims[1:])]
        self.encoder.append(nn.Conv2d(dims[-1], embed_dim, 1))
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) -> (B, H / p, W / p, D)."""
        x = self.conv_in(pixel_unshuffle(x, self.patch_size))
        for layer in self.encoder:
            x = layer(x)
        return layer_norm(self.norm_layer, x)


class GlobalRepresentationEncoder(nn.Module):
    """MLP with GELUs and a LayerNorm for per-view vectors (pose, scale)."""

    def __init__(self, in_chans: int, embed_dim: int, dims: List[int]):
        super().__init__()
        sizes = [in_chans, *dims, embed_dim]
        self.layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, C) -> (B, D)."""
        for layer in self.layers[:-1]:
            x = nn.gelu(layer(x))
        return layer_norm(self.norm_layer, self.layers[-1](x))
