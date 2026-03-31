"""DINOv2 backbone and MultiScaleProjector for RF-DETR."""

import math
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..kernels import bicubic_interpolate
from .config import DINOv2Config, ProjectorConfig

# ─── DINOv2 Backbone ───


class DINOv2Embeddings(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        self.num_windows = 4  # RF-DETR default for base/small
        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        # Position embeddings: size from positional_encoding_size or image_size
        pe_grid = config.positional_encoding_size or (
            config.image_size // config.patch_size
        )
        self.position_embeddings = mx.zeros(
            (1, 1 + pe_grid * pe_grid, config.hidden_size)
        )
        self.patch_embeddings = PatchEmbeddings(config)

    def interpolate_pos_encoding(self, x: mx.array, h: int, w: int) -> mx.array:
        """Interpolate position embeddings for variable input resolution."""
        num_patches = x.shape[1] - 1  # exclude cls token
        pos_embed = self.position_embeddings
        num_positions = pos_embed.shape[1] - 1  # stored positions (exclude cls)

        if num_patches == num_positions:
            return pos_embed

        cls_pos = pos_embed[:, :1, :]  # (1, 1, D)
        patch_pos = pos_embed[:, 1:, :]  # (1, N, D)

        dim = patch_pos.shape[-1]
        orig_h = orig_w = int(math.sqrt(num_positions))
        new_h = h // self.config.patch_size
        new_w = w // self.config.patch_size

        # Reshape to 2D grid: (1, orig_h, orig_w, D) -> (1, D, orig_h, orig_w) for bicubic
        patch_pos = patch_pos.reshape(1, orig_h, orig_w, dim)
        patch_pos = patch_pos.transpose(0, 3, 1, 2)  # (1, D, orig_h, orig_w)

        # Bicubic interpolation (matching PyTorch DINOv2)
        patch_pos = bicubic_interpolate(patch_pos, size=(new_h, new_w), antialias=True)

        # Back to (1, new_h*new_w, D)
        patch_pos = patch_pos.transpose(0, 2, 3, 1)  # (1, new_h, new_w, D)
        patch_pos = patch_pos.reshape(1, -1, dim)
        return mx.concatenate([cls_pos, patch_pos], axis=1)

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        nw = self.num_windows
        D = self.config.hidden_size
        patch_h = H // self.config.patch_size
        patch_w = W // self.config.patch_size

        # Patch embed: (B, H, W, C) -> (B, num_patches, D)
        x = self.patch_embeddings(x)
        # Prepend cls token
        cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, D))
        x = mx.concatenate([cls_tokens, x], axis=1)
        # Add position embeddings
        x = x + self.interpolate_pos_encoding(x, H, W)

        # Window the embeddings for windowed attention
        if nw > 1:
            cls_with_pos = x[:, :1, :]  # (B, 1, D)
            patches = x[:, 1:, :]  # (B, H*W, D)

            # Partition patches into windows
            patches = _window_partition(
                patches, patch_h, patch_w, nw
            )  # (B*nw², wh*ww, D)

            # Replicate CLS token for each window
            cls_windowed = mx.tile(cls_with_pos, (nw * nw, 1, 1))  # (B*nw², 1, D)

            # Prepend CLS to each window
            x = mx.concatenate([cls_windowed, patches], axis=1)  # (B*nw², 1+wh*ww, D)

        return x


class PatchEmbeddings(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) channel-last
        x = self.projection(x)  # (B, H', W', hidden_size)
        B, H, W, D = x.shape
        x = x.reshape(B, H * W, D)  # (B, num_patches, hidden_size)
        return x


class DINOv2Attention(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, D = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.o_proj(x)


class DINOv2MLP(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class DINOv2Layer(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.attention = DINOv2Attention(config)
        self.mlp = DINOv2MLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # LayerScale parameters
        self.layer_scale1 = mx.ones((config.hidden_size,))
        self.layer_scale2 = mx.ones((config.hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm attention with LayerScale
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + self.layer_scale1 * x

        # Pre-norm MLP with LayerScale
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.layer_scale2 * x
        return x


class DINOv2Encoder(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.layers = [DINOv2Layer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, x: mx.array, output_indices: List[int]
    ) -> Tuple[mx.array, List[mx.array]]:
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in output_indices:
                features.append(x)
        return x, features


def _window_partition(
    x: mx.array, patch_h: int, patch_w: int, num_windows: int
) -> mx.array:
    """Partition spatial tokens into windows.

    Args:
        x: (B, H*W, D) patch tokens (no cls)
        patch_h, patch_w: spatial dimensions
        num_windows: number of windows per side
    Returns:
        (B * num_windows^2, window_h * window_w, D)
    """
    B, N, D = x.shape
    wh = patch_h // num_windows
    ww = patch_w // num_windows
    # (B, pH, pW, D) -> (B, nw, wh, nw, ww, D) -> (B, nw, nw, wh, ww, D) -> (B*nw*nw, wh*ww, D)
    x = x.reshape(B, patch_h, patch_w, D)
    x = x.reshape(B, num_windows, wh, num_windows, ww, D)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nw, nw, wh, ww, D)
    x = x.reshape(B * num_windows * num_windows, wh * ww, D)
    return x


def _window_unpartition(
    x: mx.array, B: int, patch_h: int, patch_w: int, num_windows: int
) -> mx.array:
    """Unpartition windows back to full spatial sequence.

    Args:
        x: (B * num_windows^2, window_h * window_w, D)
        B: original batch size
        patch_h, patch_w: spatial dimensions
        num_windows: number of windows per side
    Returns:
        (B, H*W, D) patch tokens
    """
    wh = patch_h // num_windows
    ww = patch_w // num_windows
    D = x.shape[-1]
    nw = num_windows
    # (B*nw*nw, wh*ww, D) -> (B, nw, nw, wh, ww, D) -> (B, nw, wh, nw, ww, D) -> (B, pH, pW, D)
    x = x.reshape(B, nw, nw, wh, ww, D)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nw, wh, nw, ww, D)
    x = x.reshape(B, patch_h * patch_w, D)
    return x


class DINOv2Backbone(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        self.num_windows = 4  # Overridden by Model.__init__
        # Windowed layers from config, or derive from out_feature_indexes
        if config.window_block_indexes is not None:
            self.window_block_indexes = set(config.window_block_indexes)
        else:
            self.window_block_indexes = set(
                i
                for i in range(config.num_hidden_layers)
                if i not in config.out_feature_indexes
            )
        self.embeddings = DINOv2Embeddings(config)
        self.encoder = DINOv2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Args:
            x: (B, H, W, C) channel-last image
        Returns:
            List of feature maps at specified layers, each (B, h, w, D)
        """
        B, H, W, C = x.shape
        patch_h = H // self.config.patch_size
        patch_w = W // self.config.patch_size
        nw = self.num_windows
        nw2 = nw * nw

        # Embed patches with windowing: (B*nw², 1+window_tokens, D)
        hidden = self.embeddings(x)

        # Run encoder: hidden states stay in windowed format throughout
        features = []
        for i, layer in enumerate(self.encoder.layers):
            is_global = i not in self.window_block_indexes

            if is_global:
                # Merge all windows for global attention
                # (B*nw², tokens_per_window, D) -> (B, nw²*tokens_per_window, D)
                Bw, T, D = hidden.shape
                hidden = hidden.reshape(B, nw2 * T, D)

            # Run attention
            hidden = layer(hidden)

            if is_global:
                # Split back to windows
                hidden = hidden.reshape(B * nw2, T, D)

            if i in self.config.out_feature_indexes:
                # Apply model layernorm before extraction (as in PyTorch backbone)
                normed = self.layernorm(hidden)
                # Remove CLS from each window, unpartition
                patches = normed[:, 1:, :]  # (B*nw², wh*ww, D) remove CLS per window
                patches = _window_unpartition(
                    patches, B, patch_h, patch_w, nw
                )  # (B, H*W, D)
                feat = patches.reshape(B, patch_h, patch_w, -1)
                features.append(feat)

        return features


# ─── MultiScaleProjector (C2f) ───


class ConvBN(nn.Module):
    """Conv2d + LayerNorm (weights stored as 'bn' in checkpoint despite being LayerNorm)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.LayerNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        x = self.bn(x)
        return nn.silu(x)


class Bottleneck(nn.Module):
    """Bottleneck block with two 3x3 convolutions. No residual (shortcut=False in RF-DETR)."""

    def __init__(self, channels: int):
        super().__init__()
        self.cv1 = ConvBN(channels, channels, kernel_size=3, padding=1)
        self.cv2 = ConvBN(channels, channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Cross Stage Partial bottleneck with 2 convolutions (C2f from YOLOv8)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 3,
        bottleneck_channels: int = 128,
    ):
        super().__init__()
        # cv1 reduces channels, output is split in half
        self.cv1 = ConvBN(in_channels, out_channels, kernel_size=1)
        # Bottleneck modules operate on half channels
        self.m = [Bottleneck(bottleneck_channels) for _ in range(num_bottlenecks)]
        # cv2 merges all outputs: out_channels + bottleneck_channels * num_bottlenecks
        concat_channels = out_channels + bottleneck_channels * num_bottlenecks
        self.cv2 = ConvBN(concat_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        # cv1 then split
        x = self.cv1(x)  # (B, H, W, out_channels)
        split_dim = x.shape[-1] // 2
        x1 = x[..., :split_dim]  # first half stays as-is
        x2 = x[..., split_dim:]  # second half goes through bottlenecks

        outputs = [x]  # full cv1 output
        y = x2
        for bottleneck in self.m:
            y = bottleneck(y)
            outputs.append(y)

        # Concatenate along channel dim and compress
        x = mx.concatenate(outputs, axis=-1)
        return self.cv2(x)


class MultiScaleProjector(nn.Module):
    """Projects concatenated multi-scale backbone features through C2f block."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        c2f = C2f(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            num_bottlenecks=config.num_bottlenecks,
            bottleneck_channels=config.bottleneck_channels,
        )
        final_norm = nn.LayerNorm(config.hidden_dim)
        self.stages = [[c2f, final_norm]]

    def __call__(self, features: List[mx.array]) -> mx.array:
        """
        Args:
            features: list of (B, h, w, D) feature maps from backbone
        Returns:
            (B, h, w, hidden_dim) projected features
        """
        # Concatenate all features along channel dimension
        x = mx.concatenate(features, axis=-1)  # (B, h, w, sum(D))
        # Process through C2f + final LayerNorm
        c2f, final_norm = self.stages[0]
        x = c2f(x)
        x = final_norm(x)
        return x


# ─── VisionModel wrapper for framework ───


class VisionModel(nn.Module):
    """Wrapper for mlx-vlm framework compatibility."""

    def __init__(self, config=None):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return None

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return weights


# ─── Utility ───


def _interpolate_2d(x: mx.array, new_h: int, new_w: int) -> mx.array:
    """Bilinear interpolation for 2D feature maps.

    Args:
        x: (B, H, W, C) input
        new_h, new_w: target spatial dimensions
    Returns:
        (B, new_h, new_w, C) interpolated output
    """
    B, H, W, C = x.shape
    if H == new_h and W == new_w:
        return x

    # Create sampling grid
    y_coords = mx.linspace(0, H - 1, new_h)
    x_coords = mx.linspace(0, W - 1, new_w)

    # Meshgrid
    yy = mx.broadcast_to(y_coords[:, None], (new_h, new_w))
    xx = mx.broadcast_to(x_coords[None, :], (new_h, new_w))

    # Floor and ceil indices
    y0 = mx.clip(mx.floor(yy).astype(mx.int32), 0, H - 1)
    y1 = mx.clip(y0 + 1, 0, H - 1)
    x0 = mx.clip(mx.floor(xx).astype(mx.int32), 0, W - 1)
    x1 = mx.clip(x0 + 1, 0, W - 1)

    # Fractional parts
    fy = (yy - y0.astype(yy.dtype))[..., None]  # (new_h, new_w, 1)
    fx = (xx - x0.astype(xx.dtype))[..., None]  # (new_h, new_w, 1)

    # Gather corners for all batches
    # x shape: (B, H, W, C) -> index with (new_h, new_w)
    val_00 = x[:, y0, x0, :]  # (B, new_h, new_w, C)
    val_01 = x[:, y0, x1, :]
    val_10 = x[:, y1, x0, :]
    val_11 = x[:, y1, x1, :]

    # Bilinear interpolation
    result = (
        val_00 * (1 - fy) * (1 - fx)
        + val_01 * (1 - fy) * fx
        + val_10 * fy * (1 - fx)
        + val_11 * fy * fx
    )
    return result
