from typing import Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn


def get_abs_pos_sam(abs_pos, tgt_size):
    """Interpolate absolute positional embeddings to target size."""
    dtype = abs_pos.dtype
    src_size = abs_pos.shape[1]

    if src_size != tgt_size:
        # Transpose to (B, C, H, W) for interpolation
        old_pos_embed = abs_pos.transpose(0, 3, 1, 2)
        old_pos_embed = old_pos_embed.astype(mx.float32)

        # Bicubic interpolation
        from ..kernels import bicubic_interpolate

        new_pos_embed = bicubic_interpolate(
            old_pos_embed, size=(tgt_size, tgt_size), antialias=True
        ).astype(dtype)

        # Transpose back to (B, H, W, C)
        new_pos_embed = new_pos_embed.transpose(0, 2, 3, 1)
        return new_pos_embed
    else:
        return abs_pos


class MLPBlock(nn.Module):
    """MLP block with GELU activation."""

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

    def __call__(self, x: mx.array) -> mx.array:
        return self.lin2(self.act(self.lin1(x)))


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
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
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
            # Initialize relative positional embeddings
            self.rel_pos_h = mx.zeros((2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = mx.zeros((2 * input_size[1] - 1, head_dim))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, _ = x.shape

        # QKV projection and reshape
        qkv = (
            self.qkv(x)
            .reshape(B, H * W, 3, self.num_heads, -1)
            .transpose(2, 0, 3, 1, 4)
        )

        # Separate q, k, v
        qkv_reshaped = qkv.reshape(3, B * self.num_heads, H * W, -1)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]

        # Compute relative positional embeddings if needed
        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        # Reshape for attention
        q = q.reshape(B, self.num_heads, H * W, -1)
        k = k.reshape(B, self.num_heads, H * W, -1)
        v = v.reshape(B, self.num_heads, H * W, -1)

        # Apply scaled dot product attention
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
        else:
            x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Reshape output
        x = (
            x.reshape(B, self.num_heads, H, W, -1)
            .transpose(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )

        x = self.proj(x)
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation."""

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
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def __call__(self, x: mx.array) -> mx.array:
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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

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
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        return x


class SAMEncoder(nn.Module):
    """Vision Transformer encoder based on SAM architecture."""

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
        final_out_chans: int = 1024,
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
            out_chans (int): Output channels for neck.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (tuple): Indexes for blocks using global attention.
            final_out_chans (int): Final output channels after net_3 (1024 for OCR, 896 for OCR-2).
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.use_abs_pos = use_abs_pos
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size
            self.pos_embed = mx.zeros(
                (1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

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

        # Neck layers for output processing
        self.neck = [
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            nn.LayerNorm(out_chans, eps=1e-6),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(out_chans, eps=1e-6),
        ]

        # Additional downsampling layers
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(
            512, final_out_chans, kernel_size=3, stride=2, padding=1, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embeddings
        if self.use_abs_pos:
            x = x + get_abs_pos_sam(self.pos_embed, x.shape[1])

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply neck layers
        for n in self.neck:
            x = n(x)

        # Additional downsampling
        x = self.net_2(x)
        x = self.net_3(x)

        return x


# Utility functions


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
    windows: mx.array,  # FIXED: Changed from np.ndarray to mx.array
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> mx.array:  # FIXED: Changed return type from implicit to mx.array
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (mx.array): input tokens with [B * num_windows, window_size, window_size, C].
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

    # Interpolate rel pos if needed
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = rel_pos.astype(mx.float32)
        rel_pos_resized = rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(0, 2, 1)

        # Linear interpolation
        scale = rel_pos_resized.shape[2] / max_rel_dist
        indices = mx.arange(max_rel_dist, dtype=mx.float32) * scale
        idx_floor = mx.floor(indices).astype(mx.int32)
        idx_ceil = mx.minimum(idx_floor + 1, rel_pos_resized.shape[2] - 1)
        weight = indices - idx_floor.astype(mx.float32)

        rel_pos_resized = (
            mx.take(rel_pos_resized, idx_floor, axis=2) * (1 - weight)
            + mx.take(rel_pos_resized, idx_ceil, axis=2) * weight
        ).astype(dtype)

        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).transpose(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different
    q_coords = mx.arange(q_size, dtype=mx.float32)[:, None] * max(k_size / q_size, 1.0)
    k_coords = mx.arange(k_size, dtype=mx.float32)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.astype(mx.int32)]


def add_decomposed_rel_pos(
    q: mx.array,  # FIXED: Changed from np.ndarray to mx.array
    rel_pos_h: mx.array,  # FIXED: Changed from np.ndarray to mx.array
    rel_pos_w: mx.array,  # FIXED: Changed from np.ndarray to mx.array
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> Tuple[mx.array, mx.array]:  # FIXED: Added explicit return type
    """
    Calculate decomposed Relative Positional Embeddings.

    Args:
        q (mx.array): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (mx.array): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (mx.array): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        Tuple of (rel_h, rel_w): relative position biases for height and width.
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
