from typing import Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from PIL.Image import Resampling

from ..base import interpolate
from ..kernels import bicubic_interpolate


def get_abs_pos_sam(abs_pos, tgt_size):

    dtype = abs_pos.dtype

    src_size = abs_pos.shape[1]

    if src_size != tgt_size:
        old_pos_embed = abs_pos.transpose(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.astype(mx.float32)
        new_pos_embed = bicubic_interpolate(
            old_pos_embed, size=(tgt_size, tgt_size)
        ).astype(dtype)
        new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def __call__(self, x: mx.array):
        return self.lin2(self.act(self.lin1(x)))


def resize_image(image_np, new_size=(96, 96), order=1):
    """
    Resize an image with multiple channels using PIL.

    Args:
    image_np (numpy.ndarray): The input image array of shape (batch, channels, height, width).
    new_size (tuple): The target size as (height, width).
    order (int): The order of interpolation (used to determine resampling method).

    Returns:
    numpy.ndarray: The resized image array in the same format as input.
    """
    # Remove batch dimension
    image_np = np.array(image_np[0])

    # Get dimensions
    channels, height, width = image_np.shape

    # Choose interpolation method based on order parameter
    resample_method = Resampling.BILINEAR  # Default to bilinear
    if order == 0:
        resample_method = Resampling.NEAREST
    elif order == 2 or order == 3:
        resample_method = Resampling.BICUBIC

    # Handle different channel configurations
    if channels == 1:
        # For single-channel images (grayscale)
        # Reshape to 2D array (height, width)
        image_2d = image_np.reshape(height, width)

        # Create PIL image - ensure proper mode and data type conversion
        pil_image = Image.fromarray(image_2d.astype(np.float32))

        # Resize using PIL (note: PIL takes width, height order)
        resized_pil = pil_image.resize(
            (new_size[1], new_size[0]), resample=resample_method
        )

        # Convert back to numpy array, reshape to add channel dimension
        resized_np = np.array(resized_pil).reshape((1, new_size[0], new_size[1]))
    else:
        # For multi-channel images, process each channel individually
        resized_channels = []

        for c in range(channels):
            channel_data = image_np[c]
            pil_channel = Image.fromarray(channel_data.astype(np.float32))
            resized_channel = pil_channel.resize(
                (new_size[1], new_size[0]), resample=resample_method
            )
            resized_channels.append(np.array(resized_channel))

        # Stack channels back together
        resized_np = np.stack(resized_channels, axis=0)

    # Add batch dimension back and convert to mx.array
    return mx.array(resized_np)[None, :]


class SAMEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            downsample_channels (list): Channels for downsampling layers.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = mx.zeros(
                (img_size // patch_size, img_size // patch_size, embed_dim)
            )[None, :]

        self.blocks = []
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = [
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            nn.LayerNorm(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.LayerNorm(out_chans),
        ]

        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, 1024, kernel_size=3, stride=2, padding=1, bias=False
        )

    def __call__(self, x: mx.array):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x += get_abs_pos_sam(self.pos_embed, x.shape[1])

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        for _, n in enumerate(self.neck):
            x = n(x)

        x = x.transpose(0, 2, 1, 3)

        x2 = self.net_2(x)
        x3 = self.net_3(x2)

        return x3


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def __call__(self, x: mx.array):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings

            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x: mx.array):
        B, H, W, _ = x.shape
        x = mx.array(x)
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        qkv_reshaped = qkv.reshape(3, B * self.num_heads, H * W, -1)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        q = q.reshape(B, self.num_heads, H * W, -1)
        k = k.reshape(B, self.num_heads, H * W, -1)
        v = v.reshape(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            rel_h = rel_h.reshape(
                B, self.num_heads, rel_h.shape[1], rel_h.shape[2], rel_h.shape[3]
            )
            rel_w = rel_w.reshape(
                B, self.num_heads, rel_w.shape[1], rel_w.shape[2], rel_w.shape[3]
            )
            attn_bias = (rel_h + rel_w).reshape(
                B, self.num_heads, rel_h.shape[2], rel_h.shape[3] * rel_w.shape[4]
            )
            x = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=attn_bias
            )
            # x = _attention_rel_h_rel_w(q, k, v, rel_h, rel_w)
        else:
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = (
            x.reshape(B, self.num_heads, H, W, -1)
            .transpose(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)

        return x


def window_partition(x: mx.array, window_size: int) -> Tuple[mx.array, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (mx.array): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])

    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: np.ndarray,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (ndarray): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: mx.array) -> mx.array:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (mx.array): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        dtype = rel_pos.dtype
        rel_pos = rel_pos.astype(mx.float32)
        rel_pos_resized = mx.transpose(
            rel_pos.reshape(1, rel_pos.shape[0], -1), (0, 2, 1)
        )

        # Linear interpolation
        scale = rel_pos_resized.shape[2] / max_rel_dist
        indices = mx.arange(max_rel_dist) * scale
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, rel_pos_resized.shape[2] - 1)
        weight = indices - idx_floor

        rel_pos_resized = (
            mx.take(rel_pos_resized, idx_floor, axis=2) * (1 - weight)
            + mx.take(rel_pos_resized, idx_ceil, axis=2) * weight
        ).astype(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).transpose(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = mx.arange(q_size, dtype=mx.float32)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size, dtype=mx.float32)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.astype(mx.int32)]


def add_decomposed_rel_pos(
    q: np.ndarray,
    rel_pos_h: np.ndarray,
    rel_pos_w: np.ndarray,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> np.ndarray:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (ndarray): attention map.
        q (ndarray): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (ndarray): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (ndarray): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (ndarray): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = mx.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = mx.einsum("bhwc,wkc->bhwk", r_q, Rw)
    rel_h = rel_h[..., None]
    rel_w = rel_w[..., None, :]
    rel_h = rel_h.reshape(B, q_h * q_w, k_h, 1)
    rel_w = rel_w.reshape(B, q_h * q_w, 1, k_w)

    return rel_h, rel_w


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def __call__(self, x: mx.array):
        x = self.proj(x)
        return x
