"""Z-Image VAE (AutoencoderKL-style) matching quantized checkpoint weight keys.

Key structure:
- encoder.conv_in.conv2d.{weight,bias}
- encoder.down_blocks.N.resnets.M.{norm1,conv1,norm2,conv2,conv_shortcut}.{weight,bias}
- encoder.down_blocks.N.downsamplers.0.conv.{weight,bias}
- encoder.mid_block.resnets.M.{norm1,conv1,norm2,conv2}.{weight,bias}
- encoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.{weight,bias,scales}
- encoder.conv_norm_out.norm.{weight,bias}
- encoder.conv_out.conv2d.{weight,bias}
- decoder.conv_in.conv.{weight,bias}
- decoder.up_blocks.N.resnets.M.{norm1,conv1,norm2,conv2,conv_shortcut}.{weight,bias}
- decoder.up_blocks.N.upsamplers.0.conv.{weight,bias}
- decoder.mid_block.resnets.M.{norm1,conv1,norm2,conv2}.{weight,bias}
- decoder.mid_block.attentions.0.{group_norm,to_q,to_k,to_v,to_out.0}.{weight,bias,scales}
- decoder.conv_norm_out.norm.{weight,bias}
- decoder.conv_out.conv.{weight,bias}
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import ZImageVAEConfig


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class VAEAttention(nn.Module):
    """Matches mid_block.attentions.0.{group_norm, to_q, to_k, to_v, to_out.0}."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, dim, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        # List for .0. key path
        self.to_out = [nn.Linear(dim, dim)]
        self.scale = dim**-0.5

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        h = self.group_norm(x)
        q = self.to_q(h).reshape(B, H * W, 1, C)
        k = self.to_k(h).reshape(B, H * W, 1, C)
        v = self.to_v(h).reshape(B, H * W, 1, C)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        return x + self.to_out[0](out)


class Upsample(nn.Module):
    """Matches upsamplers.0.conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        return self.conv(x)


class Downsample(nn.Module):
    """Matches downsamplers.0.conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class MidBlock(nn.Module):
    """Matches mid_block.{resnets.0, attentions.0, resnets.1}."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.resnets = [ResnetBlock(channels), ResnetBlock(channels)]
        self.attentions = [VAEAttention(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class _EncoderConvIn(nn.Module):
    """Wrapper to match encoder.conv_in.conv2d key."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2d(x)


class _EncoderConvOut(nn.Module):
    """Wrapper to match encoder.conv_out.conv2d key."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv2d(x)


class _ConvNormOut(nn.Module):
    """Wrapper to match conv_norm_out.norm key."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x)


class _DecoderConvIn(nn.Module):
    """Wrapper to match decoder.conv_in.conv key."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class _DecoderConvOut(nn.Module):
    """Wrapper to match decoder.conv_out.conv key."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class _DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        add_downsample: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            self.resnets.append(
                ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            )
        if add_downsample:
            self.downsamplers = [Downsample(out_channels)]
        else:
            self.downsamplers = None

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        add_upsample: bool = True,
    ) -> None:
        super().__init__()
        self.resnets = []
        for i in range(num_layers):
            self.resnets.append(
                ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            )
        if add_upsample:
            self.upsamplers = [Upsample(out_channels)]
        else:
            self.upsamplers = None

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: ZImageVAEConfig) -> None:
        super().__init__()
        channels = config.block_out_channels
        self.conv_in = _EncoderConvIn(config.in_channels, channels[0])
        self.down_blocks = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            is_last = i == len(channels) - 1
            self.down_blocks.append(
                _DownBlock(in_ch, out_ch, config.layers_per_block, add_downsample=not is_last)
            )
            in_ch = out_ch
        self.mid_block = MidBlock(channels[-1])
        self.conv_norm_out = _ConvNormOut(channels[-1])
        self.conv_out = _EncoderConvOut(channels[-1], 2 * config.latent_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = nn.silu(self.conv_norm_out(x))
        return self.conv_out(x)


class Decoder(nn.Module):
    def __init__(self, config: ZImageVAEConfig) -> None:
        super().__init__()
        channels = config.block_out_channels
        reversed_channels = list(reversed(channels))
        self.conv_in = _DecoderConvIn(config.latent_channels, reversed_channels[0])
        self.mid_block = MidBlock(reversed_channels[0])
        self.up_blocks = []
        in_ch = reversed_channels[0]
        for i, out_ch in enumerate(reversed_channels):
            is_last = i == len(reversed_channels) - 1
            self.up_blocks.append(
                _UpBlock(
                    in_ch,
                    out_ch,
                    config.layers_per_block + 1,
                    add_upsample=not is_last,
                )
            )
            in_ch = out_ch
        self.conv_norm_out = _ConvNormOut(reversed_channels[-1])
        self.conv_out = _DecoderConvOut(reversed_channels[-1], config.out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = nn.silu(self.conv_norm_out(x))
        return self.conv_out(x)


class ZImageVAE(nn.Module):
    """AutoencoderKL for Z-Image."""

    def __init__(self, config: ZImageVAEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ZImageVAEConfig()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x: mx.array) -> mx.array:
        """x: [B, H, W, 3] in [-1, 1]. Returns mean latent."""
        h = self.encoder(x)
        mean, _ = mx.split(h, 2, axis=-1)
        return mean

    def decode(self, z: mx.array) -> mx.array:
        """z: [B, H, W, C_latent]. Returns [B, H, W, 3]."""
        return self.decoder(z)


__all__ = ["ZImageVAE"]
