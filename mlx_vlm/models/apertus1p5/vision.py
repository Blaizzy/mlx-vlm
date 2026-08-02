"""Apertus 1.5 vision tokenizer.

An encode-only port of the EMU3.5 vision tokenizer (BAAI, Apache-2.0): a
VQ-GAN-style encoder with a 131k IBQ codebook and 16x spatial downsampling.
Only the inference path is ported — Apertus 1.5 generates text, so the decoder
and the differentiable index-backpropagation training path are omitted.

Everything here runs in NHWC, MLX's native convolution layout, so no transposes
are needed around the convolutions.

The tokenizer must run in ``float32``: code assignment is an argmax over 131k
codebook logits and half precision flips roughly 8% of the codes. ``Model``
keeps these weights in ``float32`` when sanitizing and vetoes quantization for
this submodule.
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, config: VisionConfig, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(
            config.num_groups, in_channels, eps=config.norm_eps, pytorch_compatible=True
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            config.num_groups,
            out_channels,
            eps=config.norm_eps,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions with 1x1 convolutions."""

    def __init__(self, config: VisionConfig, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(
            config.num_groups, channels, eps=config.norm_eps, pytorch_compatible=True
        )
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        queries, keys, values = self.q(h), self.k(h), self.v(h)

        B, H, W, C = queries.shape
        queries = queries.reshape(B, H * W, C)
        keys = keys.reshape(B, H * W, C)
        values = values.reshape(B, H * W, C)

        # The reference uses plain matmuls rather than a fused attention kernel
        # to keep bitwise parity, which the code argmax is sensitive to.
        scores = (queries @ keys.transpose(0, 2, 1)) * C**-0.5
        scores = mx.softmax(scores, axis=-1)
        h = (scores @ values).reshape(B, H, W, C)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        # Asymmetric bottom/right padding, as in the original VQ-GAN encoder.
        x = mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(x)


class DownStage(nn.Module):
    def __init__(
        self,
        block: List[ResnetBlock],
        attn: List[AttnBlock],
        downsample: Optional[Downsample],
    ):
        super().__init__()
        self.block = block
        self.attn = attn
        if downsample is not None:
            self.downsample = downsample

    def __call__(self, x: mx.array) -> mx.array:
        for i, block in enumerate(self.block):
            x = block(x)
            if self.attn:
                x = self.attn[i](x)
        if hasattr(self, "downsample"):
            x = self.downsample(x)
        return x


class MidStage(nn.Module):
    def __init__(self, config: VisionConfig, channels: int):
        super().__init__()
        self.block_1 = ResnetBlock(config, channels, channels)
        self.attn_1 = AttnBlock(config, channels)
        self.block_2 = ResnetBlock(config, channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.block_2(self.attn_1(self.block_1(x)))


class Encoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        # Attention placement is decided statically from the reference
        # resolution, as in the original tokenizer.
        resolution = config.resolution
        multipliers = [1] + list(config.channel_multiplier)
        num_stages = len(config.channel_multiplier)

        self.down = []
        for level in range(num_stages):
            block_in = config.base_channels * multipliers[level]
            block_out = config.base_channels * config.channel_multiplier[level]
            block, attn = [], []
            for _ in range(config.num_res_blocks):
                block.append(ResnetBlock(config, block_in, block_out))
                block_in = block_out
                if resolution in config.attn_resolutions:
                    attn.append(AttnBlock(config, block_in))
            is_last = level == num_stages - 1
            downsample = None if is_last else Downsample(block_in)
            if not is_last:
                resolution //= 2
            self.down.append(DownStage(block, attn, downsample))

        self.mid = MidStage(config, block_in)
        self.norm_out = nn.GroupNorm(
            config.num_groups, block_in, eps=config.norm_eps, pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(
            block_in, config.latent_channels, kernel_size=3, padding=1
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for stage in self.down:
            x = stage(x)
        x = self.mid(x)
        return self.conv_out(swish(self.norm_out(x)))


class VectorQuantizer(nn.Module):
    """IBQ codebook lookup, which at inference is a similarity argmax."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, embed_dim) -> code ids (B, H, W)
        logits = x @ self.embedding.weight.T
        return mx.argmax(logits, axis=-1)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.encoder = Encoder(config)
        self.quantize = VectorQuantizer(config)
        self.quant_conv = nn.Conv2d(
            config.latent_channels, config.embed_dim, kernel_size=1
        )

    @property
    def spatial_scale_factor(self) -> int:
        return self.config.spatial_scale_factor

    def encode(self, pixel_values: mx.array) -> mx.array:
        """Tokenize NHWC images in ``[-1, 1]`` into a grid of codebook indices.

        Sides must be multiples of ``spatial_scale_factor``. Returns ids of
        shape ``(B, H // factor, W // factor)`` in ``[0, codebook_size)``.
        """
        pixel_values = pixel_values.astype(self.encoder.conv_in.weight.dtype)
        return self.quantize(self.quant_conv(self.encoder(pixel_values)))

    # A 3x3 convolution over the 3 input channels is (256, 3, 3, 3) in either
    # layout, so per-key shape checks cannot tell NCHW from NHWC. This key can:
    # it is (C, C, 3, 3) in PyTorch and (C, 3, 3, C) in MLX.
    LAYOUT_PROBE_KEY = "encoder.down.0.block.0.conv1.weight"

    def sanitize(self, weights):
        probe = weights.get(self.LAYOUT_PROBE_KEY)
        if probe is not None and probe.shape[1] == probe.shape[2]:
            # Already converted to MLX's NHWC layout.
            return weights
        # PyTorch conv2d weights are (out, in, kH, kW); MLX wants (out, kH, kW, in).
        return {
            k: (v.transpose(0, 2, 3, 1) if v.ndim == 4 else v)
            for k, v in weights.items()
        }
