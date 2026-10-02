"""VGGT-Omega camera, dense (DPT) and text-alignment heads, channel-last."""

import math
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..interpolate import resize_bilinear_nhwc
from ..sapiens2.heads import Conv1x1, PixelShuffle, _conv_weight, _run
from ..video_depth_anything.dpt import ResidualConvUnit
from .config import ModelConfig
from .layers import Block, norm32


class CameraHead(nn.Module):
    """Transformer over the camera and register tokens of all frames; the
    camera token of each frame gives its 9D pose encoding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = 2 * config.embed_dim
        eps = config.layer_norm_eps
        self.token_norm = nn.LayerNorm(dim, eps=eps)
        self.trunk = [
            Block(dim, config.head_num_heads, config.mlp_ratio, eps=eps)
            for _ in range(config.head_depth)
        ]
        self.trunk_norm = nn.LayerNorm(dim, eps=eps)
        self.camera_branch = [
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 9),
        ]

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens: camera and register tokens (B, S, P, 2D) -> pose encoding
        (B, S, 9) float32: translation (3), quaternion XYZW (4),
        vertical/horizontal FoV (2)."""
        B, S, P, _ = tokens.shape
        x = norm32(self.token_norm, tokens).reshape(B, S * P, -1)
        for block in self.trunk:
            x = block(x)
        camera = norm32(self.trunk_norm, x.reshape(B, S, P, -1)[:, :, 0])
        dtype = self.camera_branch[0].weight.dtype
        raw = _run(self.camera_branch, camera.astype(dtype)).astype(mx.float32)
        return mx.concatenate([raw[..., :7], nn.relu(raw[..., 7:]) + 0.01], axis=-1)


class TextAlignmentHead(nn.Module):
    """Language-aligned sequence embedding read out from camera/register tokens."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = 2 * config.embed_dim
        eps = config.layer_norm_eps
        self.token_norm = nn.LayerNorm(dim, eps=eps)
        self.language_token = mx.zeros((1, 1, dim))
        self.readout_blocks = [
            Block(dim, config.head_num_heads, config.mlp_ratio, eps=eps)
            for _ in range(config.head_depth)
        ]
        self.language_token_norm = nn.LayerNorm(dim, eps=eps)
        self.embedding_projector = [
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.LayerNorm(dim // 2, eps=eps),
            nn.Linear(dim // 2, dim),
        ]

    def __call__(self, tokens: mx.array) -> Dict[str, mx.array]:
        """tokens: camera and register tokens (B, S, P, 2D)."""
        B, S, P, _ = tokens.shape
        x = norm32(self.token_norm, tokens).reshape(B, S * P, -1)
        language = mx.broadcast_to(self.language_token, (B, 1, x.shape[-1]))
        x = mx.concatenate([language.astype(mx.float32), x], axis=1)
        for block in self.readout_blocks:
            x = block(x)
        token = norm32(self.language_token_norm, x[:, 0])

        first, act, norm, last = self.embedding_projector
        dtype = first.weight.dtype
        h = norm32(norm, act(first(token.astype(dtype))))
        embedding = last(h.astype(dtype)).astype(mx.float32)
        embedding = embedding / mx.maximum(
            mx.linalg.norm(embedding, axis=-1, keepdims=True), 1e-12
        )
        return {"text_alignment_embedding": embedding, "text_alignment_token": token}


class PatchDeconv(nn.Module):
    """``ConvTranspose2d`` with kernel size == stride: every input pixel
    writes its own k x k output patch, so the layer is one matmul.

    Holds the ``nn.ConvTranspose2d`` (out, k, k, in) weight layout.
    """

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.weight = _conv_weight(channels, k, channels)
        self.bias = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        out, k = self.weight.shape[0], self.weight.shape[1]
        # (out, i, j, in) -> (in, (i, j, out))
        weight = self.weight.transpose(3, 1, 2, 0).reshape(C, k * k * out)
        y = mx.addmm(mx.tile(self.bias, k * k), x.reshape(-1, C), weight)
        y = y.reshape(B, H, W, k, k, out).transpose(0, 1, 3, 2, 4, 5)
        return y.reshape(B, H * k, W * k, out)


class FusionBlock(nn.Module):
    """DPT feature fusion: residual conv units, align-corners bilinear
    resize to the next level, then a 1x1 conv."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.out_conv = Conv1x1(features, features)
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def __call__(
        self,
        x: mx.array,
        residual: Optional[mx.array] = None,
        *,
        size: Tuple[int, int],
    ) -> mx.array:
        if residual is not None:
            x = x + self.resConfUnit1(residual)
        x = self.resConfUnit2(x)
        x = resize_bilinear_nhwc(x, size, align_corners=True).astype(x.dtype)
        return self.out_conv(x)


class Scratch(nn.Module):
    """Container matching the checkpoint's ``dense_head.scratch.*`` keys."""

    def __init__(self, in_channels: List[int], features: int):
        super().__init__()
        for i, c in enumerate(in_channels):
            conv = nn.Conv2d(c, features, kernel_size=3, padding=1, bias=False)
            setattr(self, f"layer{i + 1}_rn", conv)
        for i in range(1, 4):
            setattr(self, f"refinenet{i}", FusionBlock(features))
        self.refinenet4 = FusionBlock(features, has_residual=False)

    def __call__(self, features: List[mx.array]) -> mx.array:
        layer_1, layer_2, layer_3, layer_4 = (
            getattr(self, f"layer{i + 1}_rn")(x) for i, x in enumerate(features)
        )
        out = self.refinenet4(layer_4, size=layer_3.shape[1:3])
        out = self.refinenet3(out, layer_3, size=layer_2.shape[1:3])
        out = self.refinenet2(out, layer_2, size=layer_1.shape[1:3])
        return self.refinenet1(out, layer_1, size=layer_1.shape[1:3])


def _position_embedding(
    height: int, width: int, channels: int, aspect_ratio: float
) -> mx.array:
    """(height, width, channels) sin/cos embedding of a UV grid spanning the
    image diagonal, times the head's 0.1 ratio."""
    diagonal = math.sqrt(aspect_ratio**2 + 1.0)
    span_x = aspect_ratio / diagonal * (width - 1) / width
    span_y = 1.0 / diagonal * (height - 1) / height
    x = mx.linspace(-span_x, span_x, width)
    y = mx.linspace(-span_y, span_y, height)

    quarter = channels // 4
    omega = 1.0 / 100.0 ** (mx.arange(quarter, dtype=mx.float32) / quarter)

    def embed(pos):
        angles = pos[:, None] * omega
        return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)

    emb_x = mx.broadcast_to(embed(x)[None], (height, width, 2 * quarter))
    emb_y = mx.broadcast_to(embed(y)[:, None], (height, width, 2 * quarter))
    return mx.concatenate([emb_x, emb_y], axis=-1) * 0.1


class DenseHead(nn.Module):
    """DPT head: depth (exp activation) and confidence (1 + exp) at the
    input resolution."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim_in = 2 * config.embed_dim
        channels = config.dense_out_channels
        features = config.dense_features
        self.patch_size = config.patch_size
        self.frames_chunk_size = config.dense_frames_chunk_size
        shuffle = config.patch_size // 4

        self.norm = nn.LayerNorm(dim_in, eps=config.layer_norm_eps)
        self.projects = [Conv1x1(dim_in, c) for c in channels]
        self.resize_layers = [
            PatchDeconv(channels[0], 4),
            PatchDeconv(channels[1], 2),
            nn.Identity(),
            nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=2, padding=1),
        ]
        self.scratch = Scratch(channels, features)
        self.proj = Conv1x1(features, shuffle**2)
        self.proj_conf = Conv1x1(features, shuffle**2)
        self.pixel_shuffle = PixelShuffle(shuffle)
        self._position_tables = {}

    def __call__(
        self, layers: List[mx.array], image_size: Tuple[int, int]
    ) -> Tuple[mx.array, mx.array]:
        """layers: cached patch tokens (B, S, h * w, 2D) -> depth
        (B, S, H, W, 1) and confidence (B, S, H, W), float32."""
        S = layers[0].shape[1]
        step = self.frames_chunk_size or S
        chunks = [
            self._forward([x[:, start : start + step] for x in layers], image_size)
            for start in range(0, S, step)
        ]
        depth, conf = (mx.concatenate(parts, axis=1) for parts in zip(*chunks))
        return depth, conf

    def _add_position_embedding(self, x: mx.array, aspect_ratio: float) -> mx.array:
        """Tables are cached per size, like the RoPE tables."""
        key = (*x.shape[1:], aspect_ratio)
        if key not in self._position_tables:
            self._position_tables[key] = _position_embedding(*key)
        return x + self._position_tables[key].astype(x.dtype)

    def _forward(self, layers: List[mx.array], image_size: Tuple[int, int]):
        H, W = image_size
        h, w = H // self.patch_size, W // self.patch_size
        aspect_ratio = W / H
        dtype = self.projects[0].weight.dtype

        features = []
        for project, resize, x in zip(self.projects, self.resize_layers, layers):
            B, S = x.shape[:2]
            x = norm32(self.norm, x).astype(dtype).reshape(B * S, h, w, -1)
            x = self._add_position_embedding(project(x), aspect_ratio)
            features.append(resize(x))

        fused = self._add_position_embedding(self.scratch(features), aspect_ratio)
        depth = self.pixel_shuffle(self.proj(fused)).astype(mx.float32)
        conf = self.pixel_shuffle(self.proj_conf(fused)).astype(mx.float32)
        depth = mx.exp(depth).reshape(B, S, H, W, 1)
        conf = (1.0 + mx.exp(conf)).reshape(B, S, H, W)
        return depth, conf
