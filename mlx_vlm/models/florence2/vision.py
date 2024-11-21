import inspect
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class VisionConfig:
    """Configuration class for Florence2 Vision model (DaViT)."""

    model_type: str = "davit"
    in_chans: int = 3
    num_classes: int = 1000
    depths: List[int] = field(default_factory=lambda: [1, 1, 9, 1])
    dim_embed: List[int] = field(default_factory=lambda: [128, 256, 512, 1024])
    num_heads: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    num_groups: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    window_size: int = 12
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    patch_size: List[int] = field(default_factory=lambda: [7, 3, 3, 3])
    patch_stride: List[int] = field(default_factory=lambda: [4, 2, 2, 2])
    patch_padding: List[int] = field(default_factory=lambda: [3, 1, 1, 1])
    patch_prenorm: List[bool] = field(
        default_factory=lambda: [False, False, False, False]
    )
    qkv_bias: bool = True
    conv_at_attn: bool = True
    conv_at_ffn: bool = True
    hidden_size: int = 768
    image_size: Tuple[int, int] = (768, 768)

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def check_array_shape(arr):
    shape = arr.shape

    # Check if the shape has 4 dimensions
    if len(shape) != 4:
        return False

    out_channels, kH, KW, _ = shape

    # Check if out_channels is the largest, and kH and KW are the same
    if (out_channels >= kH) and (out_channels >= KW) and (kH == KW):
        return True
    else:
        return False


class MlpFC(nn.Module):
    """MLP FC module"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.gelu = nn.GELU()

    def __call__(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class Mlp(nn.Module):
    """MLP module"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.net = MlpFC(in_features, hidden_features, out_features)

    def __call__(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    """Depthwise Convolution"""

    def __init__(
        self,
        dim_in: int,
        kernel_size: int,
        padding: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()

        self.dw = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
            groups=dim_in,
        )

    def __call__(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.reshape(B, H, W, C))

        x = x.transpose(0, 3, 1, 2)

        size = (x.shape[-2], x.shape[-1])
        x = x.flatten(2).transpose(0, 2, 1)
        return x, size


class ConvEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size: int = 7,
        in_chans: int = 3,
        embed_dim: int = 64,
        stride: int = 4,
        padding: int = 2,
        norm_layer: Optional[nn.Module] = None,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )

        if norm_layer and pre_norm:
            self.norm = norm_layer(in_chans)
        elif norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.pre_norm = pre_norm

    def __call__(self, x, size):
        H, W = size
        if len(x.shape) == 3:

            if self.norm and self.pre_norm:
                x = self.norm(x)

            x = x.reshape(-1, H, W, x.shape[-1])
        else:
            x = x.transpose(0, 2, 3, 1)

        x = self.proj(x)

        B, H, W, C = x.shape

        x = x.reshape(B, H * W, C)

        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):
    """Channel Attention module"""

    def __init__(self, dim: int, groups: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.groups = groups
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, groups, N, C//groups)

        q = q * (float(N) ** -0.5)

        # For multi-head attention, we need to keep the groups dimension
        attention = mx.matmul(q.transpose(0, 1, 3, 2), k)  # (B, groups, N, N)
        attention = mx.softmax(attention, axis=-1)

        x = mx.matmul(attention, v.transpose(0, 1, 3, 2)).transpose(
            0, 1, 3, 2
        )  # (B, groups, N, C//groups)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)

        return x, size


def window_partition(x: mx.array, window_size: int):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = mx.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = mx.reshape(
        mx.transpose(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: mx.array, batch_size: int, window_size: int, H: int, W: int
):
    """Merge windows back to feature map"""
    B = batch_size
    x = mx.reshape(
        windows, (B, H // window_size, W // window_size, window_size, window_size, -1)
    )
    x = mx.reshape(mx.transpose(x, (0, 1, 3, 2, 4, 5)), (B, H, W, -1))
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention module"""

    def __init__(
        self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x, size):
        H, W = size
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size {L} == {H * W}"

        x = mx.reshape(x, (B, H, W, C))

        # Calculate padding
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        # MLX padding
        x = mx.pad(x, [(0, 0), (pad_t, pad_b), (pad_l, pad_r), (0, 0)])

        _, Hp, Wp, _ = x.shape

        # Window partition
        x = window_partition(x, self.window_size)
        x = mx.reshape(x, (-1, self.window_size * self.window_size, C))

        # Multi-head self attention
        B_, N, C = x.shape
        qkv = mx.reshape(self.qkv(x), (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        q = q * self.scale
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
        attn = mx.softmax(attn, axis=-1)

        x = mx.reshape(mx.transpose(mx.matmul(attn, v), (0, 2, 1, 3)), (B_, N, C))
        x = self.proj(x)

        # Merge windows
        x = mx.reshape(x, (-1, self.window_size, self.window_size, C))
        x = window_reverse(x, B, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = mx.reshape(x, (B, H * W, C))
        return x, size


class PreNorm(nn.Module):
    """Pre-normalization module"""

    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def __call__(self, x, size):
        shortcut = x
        if self.norm is not None:
            x = self.norm(x)
        x, size = self.fn(x, size)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = shortcut + x
        return x, size


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + mx.random.uniform(shape)
        random_tensor = mx.floor(random_tensor)
        output = x * random_tensor / keep_prob
        return output


class SpatialBlock(nn.Module):
    """Spatial attention block"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.conv1 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1), None) if conv_at_attn else None
        )

        self.window_attn = PreNorm(
            nn.LayerNorm(dim),
            WindowAttention(dim, num_heads, window_size, qkv_bias=qkv_bias),
            drop_path,
        )

        self.conv2 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1), None) if conv_at_ffn else None
        )

        self.ffn = PreNorm(
            nn.LayerNorm(dim),
            Mlp(
                in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim
            ),
            drop_path,
        )

    def __call__(self, x, size):
        if self.conv1 is not None:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2 is not None:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class ChannelBlock(nn.Module):
    """Channel attention block"""

    def __init__(
        self,
        dim: int,
        groups: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        self.conv1 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1), None) if conv_at_attn else None
        )

        self.channel_attn = PreNorm(
            nn.LayerNorm(dim),
            ChannelAttention(dim, groups=groups, qkv_bias=qkv_bias),
            drop_path,
        )

        self.conv2 = (
            PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1), None) if conv_at_ffn else None
        )

        self.ffn = PreNorm(
            nn.LayerNorm(dim),
            Mlp(
                in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim
            ),
            drop_path,
        )

    def __call__(self, x, size):
        if self.conv1 is not None:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2 is not None:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: Tuple[float, float] = (0.0, 0.0),
        conv_at_attn: bool = True,
        conv_at_ffn: bool = True,
    ):
        super().__init__()
        self.spatial_block = SpatialBlock(
            dim,
            num_heads,
            window_size,
            drop_path_rate=drop_path_rate[0],
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            conv_at_attn=conv_at_attn,
            conv_at_ffn=conv_at_ffn,
        )
        self.channel_block = ChannelBlock(
            dim,
            num_groups,
            drop_path_rate=drop_path_rate[1],
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            conv_at_attn=conv_at_attn,
            conv_at_ffn=conv_at_ffn,
        )

    def __call__(self, x, size):
        x, size = self.spatial_block(x, size)
        x, size = self.channel_block(x, size)
        return x, size


class VisionModel(nn.Module):
    """DaViT: Dual Attention Vision Transformer"""

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.dim_embed = config.dim_embed
        self.num_heads = config.num_heads
        self.num_groups = config.num_groups
        self.num_stages = len(self.dim_embed)
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        if self.model_type not in ["davit", ""]:
            raise ValueError(
                f"Model type {self.model_type} not supported. Currently only 'davit' is supported"
            )

        # Convert PyTorch's linspace to MLX equivalent
        total_blocks = sum(config.depths) * 2
        dpr = [
            i * config.drop_path_rate / (total_blocks - 1) for i in range(total_blocks)
        ]

        depth_offset = 0
        self.convs = []
        self.blocks = []

        for i in range(self.num_stages):

            conv_embed = ConvEmbed(
                patch_size=config.patch_size[i],
                stride=config.patch_stride[i],
                padding=config.patch_padding[i],
                in_chans=config.in_chans if i == 0 else self.dim_embed[i - 1],
                embed_dim=self.dim_embed[i],
                norm_layer=nn.LayerNorm,
                pre_norm=config.patch_prenorm[i],
            )
            self.convs.append(conv_embed)

            block = []
            for j in range(config.depths[i]):
                block.append(
                    Block(
                        self.dim_embed[i],
                        config.num_heads[i],
                        config.num_groups[i],
                        config.window_size,
                        config.mlp_ratio,
                        config.qkv_bias,
                        (dpr[depth_offset + j * 2], dpr[depth_offset + j * 2 + 1]),
                        config.conv_at_attn,
                        config.conv_at_ffn,
                    )
                )

            self.blocks.append(block)

            depth_offset += config.depths[i] * 2

    def __call__(self, x):
        input_size = x.shape[2:]

        # Process through stages
        for conv, blks in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            for blk in blks:
                x, input_size = blk(x, input_size)

        return x

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            elif "convs" in k:
                if "proj.weight" in k:
                    # PyTorch conv2d weight tensors have shape:
                    #   [out_channels, in_channels, kH, KW]
                    # MLX conv2d expects the weight be of shape:
                    #   [out_channels, kH, KW, in_channels]
                    if check_array_shape(v):
                        sanitized_weights[k] = v
                    else:
                        sanitized_weights[k] = v.transpose(0, 2, 3, 1)
                else:
                    sanitized_weights[k] = v
            elif "blocks" in k:
                if "dw.weight" in k:
                    sanitized_weights[k] = (
                        v.transpose(0, 2, 3, 1) if v.shape[1] < v.shape[-1] else v
                    )
                else:
                    sanitized_weights[k] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights
