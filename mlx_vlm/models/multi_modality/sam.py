import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from PIL.Image import Resampling


@dataclass
class SAMViTCfg:
    image_size: Union[Tuple[int, int], int] = 1024
    width: int = 768
    layers: int = 12
    heads: int = 12
    patch_size: int = 16
    window_size: int = 14
    prompt_embed_dim: int = 256
    global_attn_indexes: Union[List[int], Tuple[int]] = (2, 5, 8, 11)
    downsample_channels: Union[List[int], Tuple[int]] = (512, 1024)


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
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
        downsample_channels: Tuple[int, ...] = (512, 1024),
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
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
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
                rel_pos_zero_init=rel_pos_zero_init,
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

        in_channels = out_chans
        self.downsamples = []
        for i in range(len(downsample_channels)):
            out_channels = downsample_channels[i]
            self.downsamples.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            in_channels = out_channels

        self.sam_hd = True
        if self.sam_hd:
            self.hd_alpha_downsamples = mx.zeros((1))
            self.neck_hd = copy.deepcopy(self.neck)

    def __call__(self, x: mx.array):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x += self.pos_embed

        global_features = []
        for _, blk in enumerate(self.blocks):
            x = blk(x)
            if self.sam_hd and blk.window_size == 0:
                global_features.append(x)

        for _, n in enumerate(self.neck):
            x = n(x)

        x = x.transpose(0, 3, 1, 2)
        x = resize_image(x)

        x = x.transpose(0, 2, 3, 1)

        for _, downsample in enumerate(self.downsamples):
            x = downsample(x)

        if self.sam_hd:
            first_global_feature = global_features[0]
            for _, n_hd in enumerate(self.neck_hd):
                first_global_feature = n_hd(first_global_feature)

            first_global_feature = first_global_feature.transpose(0, 3, 1, 2)

            first_global_feature = resize_image(first_global_feature)

            first_global_feature = first_global_feature.transpose(0, 2, 3, 1)
            for _, downsample in enumerate(self.downsamples):
                first_global_feature = downsample(first_global_feature)

            x = x + first_global_feature * self.hd_alpha_downsamples

        return x


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
        rel_pos_zero_init: bool = True,
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
            rel_pos_zero_init=rel_pos_zero_init,
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
        rel_pos_zero_init: bool = True,
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
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1)

        def do_attention(q, k, v):
            attn = (q * self.scale) @ k.transpose(0, -1, -2)
            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(
                    attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                )

            attn = mx.softmax(attn, axis=-1)
            x = (
                (attn @ v)
                .reshape(B, self.num_heads, H, W, -1)
                .transpose(0, 2, 3, 1, 4)
                .reshape(B, H, W, -1)
            )

            return x

        x = do_attention(q, k, v)
        x = self.proj(x)

        return x


def window_partition(
    x: np.ndarray, window_size: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (ndarray): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
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


def get_rel_pos(q_size: int, k_size: int, rel_pos: np.ndarray) -> np.ndarray:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (ndarray): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    rel_pos = np.array(rel_pos)
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = np.expand_dims(rel_pos, axis=0)
        rel_pos_resized = np.transpose(rel_pos_resized, (0, 2, 1))
        rel_pos_resized = np.interp(
            np.linspace(0, max_rel_dist - 1, num=max_rel_dist),
            np.arange(rel_pos.shape[0]),
            rel_pos_resized[0],
        )
        rel_pos_resized = np.transpose(rel_pos_resized, (1, 0))
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = np.arange(q_size)[:, np.newaxis] * max(k_size / q_size, 1.0)
    k_coords = np.arange(k_size)[np.newaxis, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    relative_coords = relative_coords.astype(np.int64)
    return rel_pos_resized[relative_coords]


def add_decomposed_rel_pos(
    attn: np.ndarray,
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

    rel_h = np.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = np.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.reshape(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, np.newaxis]
        + rel_w[:, :, :, np.newaxis, :]
    ).reshape(B, q_h * q_w, k_h * k_w)

    return attn


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
