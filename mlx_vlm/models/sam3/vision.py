"""SAM3 Vision Encoder: ViT backbone with windowed/global attention + FPN neck.

Weight key prefix: detector_model.vision_encoder.backbone.* and detector_model.vision_encoder.neck.*
"""

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .config import VisionEncoderConfig, ViTConfig
from .position import apply_rotary_enc, compute_axial_cis

# ---------------------------------------------------------------------------
# ViT Components
# ---------------------------------------------------------------------------


class PatchProjection(nn.Module):
    """Inner projection layer to match weight key: patch_embeddings.projection."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.projection(x)


class PatchEmbeddings(nn.Module):
    """Patch embedding with Conv2d projection.

    Weight key: embeddings.patch_embeddings.projection.weight
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embeddings = PatchProjection(config)
        # Absolute position embeddings: (1, num_pretrain_patches, hidden_size)
        num_patches = (config.pretrain_image_size // config.patch_size) ** 2
        self.position_embeddings = mx.zeros((1, num_patches, config.hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) image in MLX channel-last format
        Returns:
            (B, num_patches, hidden_size)
        """
        x = self.patch_embeddings(x)  # (B, H', W', hidden_size)
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)  # (B, N, C)
        return x


class VitAttention(nn.Module):
    """Multi-head attention with optional windowed attention and 2D RoPE."""

    def __init__(self, config: ViTConfig, use_rope: bool = True):
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

        self.use_rope = use_rope

    def __call__(
        self,
        x: mx.array,
        rope_cos: Optional[mx.array] = None,
        rope_sin: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: (B, N, C) input (or (B, H, W, C) spatial)
            rope_cos: (N, D) cosine RoPE
            rope_sin: (N, D) sine RoPE
        Returns:
            same shape as input
        """
        input_shape = x.shape
        if x.ndim == 4:
            B, H, W, C = x.shape
            N = H * W
            x = x.reshape(B, N, C)
        else:
            B, N, C = x.shape

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

        # Apply RoPE after transpose to (B, H, N, D) - matching HF
        if self.use_rope and rope_cos is not None:
            q, k = apply_rotary_enc(q, k, rope_cos, rope_sin)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.o_proj(out)

        if len(input_shape) == 4:
            out = out.reshape(input_shape)
        return out


class VitMLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class VitBlock(nn.Module):
    """ViT transformer block with optional windowed attention.

    Operates on spatial (B, H, W, C) tensors matching HF Sam3ViTLayer.
    """

    def __init__(self, config: ViTConfig, is_global: bool = False):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = VitAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = VitMLP(config)
        self.window_size = 0 if is_global else config.window_size
        self.is_global = is_global

    def __call__(
        self,
        x: mx.array,
        rope_cos: Optional[mx.array] = None,
        rope_sin: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: (B, H, W, C) spatial features
            rope_cos: (N, D) cosine RoPE embeddings
            rope_sin: (N, D) sine RoPE embeddings
        """
        residual = x
        x = self.layer_norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
            x = self.attention(x, rope_cos, rope_sin)
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            x = self.attention(x, rope_cos, rope_sin)

        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


def _window_partition(
    x: mx.array, window_size: int
) -> Tuple[mx.array, Tuple[int, int]]:
    """Partition spatial features into non-overlapping windows.

    Args:
        x: (B, H, W, C) spatial features
    Returns:
        windows: (B*nH*nW, ws, ws, C) windowed features
        pad_hw: (Hp, Wp) padded dimensions
    """
    B, H, W, C = x.shape
    ws = window_size

    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h > 0 or pad_w > 0:
        x = mx.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)])
    Hp, Wp = H + pad_h, W + pad_w

    nH, nW = Hp // ws, Wp // ws
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nH, nW, ws, ws, C)
    x = x.reshape(B * nH * nW, ws, ws, C)
    return x, (Hp, Wp)


def _window_unpartition(
    x: mx.array,
    window_size: int,
    pad_hw: Tuple[int, int],
    original_hw: Tuple[int, int],
) -> mx.array:
    """Reverse window partition.

    Args:
        x: (B*nH*nW, ws, ws, C)
    Returns:
        (B, H, W, C)
    """
    ws = window_size
    Hp, Wp = pad_hw
    H, W = original_hw
    nH, nW = Hp // ws, Wp // ws
    B = x.shape[0] // (nH * nW)
    C = x.shape[-1]

    x = x.reshape(B, nH, nW, ws, ws, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nH, ws, nW, ws, C)
    x = x.reshape(B, Hp, Wp, C)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]

    return x


class ViTBackbone(nn.Module):
    """Vision Transformer backbone with windowed + global attention and 2D RoPE.

    Weight keys: detector_model.vision_encoder.backbone.*
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = PatchEmbeddings(config)

        feat_size = config.image_size // config.patch_size
        self.feat_size = feat_size

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        global_set = set(config.global_attn_indexes)
        self.layers = [
            VitBlock(config, is_global=(i in global_set))
            for i in range(config.num_hidden_layers)
        ]

        # Precompute RoPE (cos, sin) for window and global attention
        self._rope_window_cos, self._rope_window_sin = compute_axial_cis(
            config.hidden_size // config.num_attention_heads,
            config.window_size,
            config.window_size,
            theta=config.rope_theta,
        )
        self._rope_global_cos, self._rope_global_sin = compute_axial_cis(
            config.hidden_size // config.num_attention_heads,
            feat_size,
            feat_size,
            theta=config.rope_theta,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) image — supports any resolution divisible by patch_size
        Returns:
            (B, feat_H, feat_W, hidden_size)
        """
        B = x.shape[0]
        input_h, input_w = x.shape[1], x.shape[2]
        H = input_h // self.config.patch_size
        W = input_w // self.config.patch_size

        x = self.embeddings(x)  # (B, N, C)

        # Tile position embeddings to match actual feature size
        pos = self._tile_pos_embed(self.embeddings.position_embeddings, H, W)
        x = x + pos

        # Reshape to spatial format: (B, H, W, C) matching HF
        x = x.reshape(B, H, W, -1)
        x = self.layer_norm(x)

        # Compute RoPE for actual size if different from default
        if H != self.feat_size or W != self.feat_size:
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            global_cos, global_sin = compute_axial_cis(
                head_dim,
                H,
                W,
                theta=self.config.rope_theta,
            )
        else:
            global_cos = self._rope_global_cos
            global_sin = self._rope_global_sin

        for layer in self.layers:
            if layer.is_global:
                x = layer(x, global_cos, global_sin)
            else:
                x = layer(x, self._rope_window_cos, self._rope_window_sin)

        return x

    def _tile_pos_embed(
        self,
        pos: mx.array,
        target_h: Optional[int] = None,
        target_w: Optional[int] = None,
    ) -> mx.array:
        """Tile position embeddings to match target spatial dimensions.

        HF SAM3 uses tiling (repeating), not interpolation.
        pos: (1, pretrain_size^2, hidden_size)
        """
        N = pos.shape[1]
        pretrain_size = int(math.sqrt(N))
        target_h = target_h or self.feat_size
        target_w = target_w or self.feat_size
        hidden_size = pos.shape[-1]

        if pretrain_size == target_h and pretrain_size == target_w:
            return pos

        pos = pos.reshape(1, pretrain_size, pretrain_size, hidden_size)

        # Tile to cover target size
        repeat_h = target_h // pretrain_size + 1
        repeat_w = target_w // pretrain_size + 1

        # Tile along H and W dimensions
        pos = mx.tile(pos, (1, repeat_h, repeat_w, 1))
        # Crop to target size
        pos = pos[:, :target_h, :target_w, :]
        pos = pos.reshape(1, target_h * target_w, hidden_size)
        return pos


# ---------------------------------------------------------------------------
# FPN Neck
# ---------------------------------------------------------------------------


class FPNLayer(nn.Module):
    """Single FPN scale: upscale -> project -> refine.

    For 4x upscaling: scale_layers = {0: ConvT, 2: ConvT} (GELU at index 1 has no weight)
    For 2x upscaling: scale_layers = {0: ConvT}
    Weight indices must match PyTorch's nn.Sequential numbering.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        fpn_kernel_size: int = 2,
        fpn_stride: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_upscale = 0

        # Build scale layers matching PyTorch Sequential indices
        # For 4x: [ConvT(idx=0), GELU(idx=1), ConvT(idx=2)]
        # For 2x: [ConvT(idx=0)]
        # Use a list with None placeholders for non-parameterized ops
        current_channels = in_channels
        if scale_factor >= 4.0:
            mid = current_channels // 2
            mid2 = mid // 2
            # scale_layers indices: 0=ConvT, 1=None(GELU), 2=ConvT
            self.scale_layers = [
                nn.ConvTranspose2d(
                    current_channels,
                    mid,
                    kernel_size=fpn_kernel_size,
                    stride=fpn_stride,
                ),
                None,  # placeholder for GELU (index 1)
                nn.ConvTranspose2d(
                    mid, mid2, kernel_size=fpn_kernel_size, stride=fpn_stride
                ),
            ]
            current_channels = mid2
            self.num_upscale = 2
        elif scale_factor >= 2.0:
            mid = current_channels // 2
            self.scale_layers = [
                nn.ConvTranspose2d(
                    current_channels,
                    mid,
                    kernel_size=fpn_kernel_size,
                    stride=fpn_stride,
                ),
            ]
            current_channels = mid
            self.num_upscale = 1
        else:
            self.scale_layers = []

        self.has_scale_layers = self.num_upscale > 0
        self.is_downsample = scale_factor <= 0.5

        # 1x1 projection + 3x3 refinement (weights have biases)
        self.proj1 = nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=True)
        self.proj2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (B, H, W, C) feature map
        Returns:
            (B, H', W', out_channels)
        """
        if self.has_scale_layers:
            for layer in self.scale_layers:
                if layer is None:
                    x = nn.gelu(x)
                else:
                    x = layer(x)
        elif self.is_downsample:
            # MaxPool2d equivalent: stride 2
            B, H, W, C = x.shape
            x = x.reshape(B, H // 2, 2, W // 2, 2, C)
            x = mx.max(x, axis=(2, 4))

        x = self.proj1(x)
        x = self.proj2(x)
        return x


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck.

    Weight keys: detector_model.vision_encoder.neck.fpn_layers.*
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        backbone_config = config.backbone_config
        in_channels = backbone_config.hidden_size

        self.fpn_layers = [
            FPNLayer(
                in_channels,
                config.fpn_hidden_size,
                sf,
                config.fpn_kernel_size,
                config.fpn_stride,
            )
            for sf in config.scale_factors
        ]

    def __call__(self, x: mx.array) -> Tuple[List[mx.array], List[mx.array]]:
        """
        Args:
            x: (B, H, W, C) backbone output
        Returns:
            features: list of (B, H_i, W_i, D) multi-scale features
        """
        features = []
        for layer in self.fpn_layers:
            features.append(layer(x))
        return features


# ---------------------------------------------------------------------------
# Full Vision Encoder
# ---------------------------------------------------------------------------


class VisionEncoder(nn.Module):
    """Complete vision encoder: ViT backbone + FPN neck.

    Weight keys: detector_model.vision_encoder.*
    """

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.backbone = ViTBackbone(config.backbone_config)
        self.neck = FPNNeck(config)

    def __call__(self, x: mx.array) -> List[mx.array]:
        """
        Args:
            x: (B, H, W, C) image
        Returns:
            Multi-scale feature list from FPN
        """
        features = self.backbone(x)  # (B, H', W', C)
        fpn_features = self.neck(features)
        return fpn_features


class VisionModel(nn.Module):
    """Wrapper for mlx-vlm compatibility."""

    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.vision_encoder = VisionEncoder(config)

    def __call__(self, x: mx.array) -> List[mx.array]:
        return self.vision_encoder(x)

    @staticmethod
    def sanitize(weights):
        """No-op: all sanitization handled in Model.sanitize."""
        return weights
