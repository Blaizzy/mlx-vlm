"""Qwen-Image-2.1 3D-causal residual VAE (image inference path)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _conv2d(conv: nn.Conv2d, x: mx.array) -> mx.array:
    """Apply a channels-first ``[N, C, H, W]`` tensor through an MLX Conv2d."""
    x = x.transpose(0, 2, 3, 1)
    x = conv(x)
    return x.transpose(0, 3, 1, 2)


def _to_2d(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value[1:]) if len(value) == 3 else tuple(value)


class QwenImageCausalConv(nn.Conv2d):
    """Image specialization of the causal 3D conv: a padded per-frame 2D conv.

    Subclasses ``nn.Conv2d`` so its ``weight``/``bias`` live at the module's own
    key, matching the checkpoint (the reference likewise subclasses Conv2d).
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size, padding=0) -> None:
        super().__init__(in_dim, out_dim, _to_2d(kernel_size), padding=_to_2d(padding))

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, h, w = x.shape
        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = _conv2d(super().__call__, x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])
        return x.transpose(0, 2, 1, 3, 4)


class QwenImageRMSNorm(nn.Module):
    """Channel-wise L2 (RMS) norm matching the reference ``QwenImage21RMS_norm``."""

    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(
            mx.sum(x.astype(mx.float32) ** 2, axis=1, keepdims=True) + 1e-12
        )
        normalized = (x.astype(mx.float32) * norm).astype(x.dtype)
        gamma = self.gamma.reshape((1, -1) + (1,) * (x.ndim - 2))
        return normalized * self.scale * gamma


def _upsample_nearest(x: mx.array) -> mx.array:
    """2x nearest spatial upsample of ``[N, C, H, W]``."""
    n, c, h, w = x.shape
    x = x[:, :, :, None, :, None]
    x = mx.broadcast_to(x, (n, c, h, 2, w, 2))
    return x.reshape(n, c, h * 2, w * 2)


class _Passthrough(nn.Module):
    """Param-less placeholder so the conv sits at ``resample.1`` (matches ckpt)."""

    def __call__(self, x: mx.array) -> mx.array:
        return x


class QwenImageResample(nn.Module):
    """Spatial up/down resample (image path; the temporal conv is video-only).

    The pad/upsample op carries no parameters, so the convolution is stored at
    ``resample.1`` to match the reference's ``nn.Sequential`` layout. ``time_conv``
    exists only for the temporal (video) modes and is unused on the image path.
    """

    def __init__(
        self, dim: int, mode: str, upsample_out_dim: int | None = None
    ) -> None:
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode in ("upsample2d", "upsample3d"):
            self.resample = [
                _Passthrough(),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            ]
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = [
                _Passthrough(),
                nn.Conv2d(dim, dim, 3, stride=2, padding=0),
            ]
        else:
            self.resample = [_Passthrough()]
        if mode == "upsample3d":
            self.time_conv = QwenImageCausalConv(dim, dim * 2, 1)
        elif mode == "downsample3d":
            self.time_conv = QwenImageCausalConv(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, h, w = x.shape
        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if self.mode in ("upsample2d", "upsample3d"):
            x = _conv2d(self.resample[1], _upsample_nearest(x))
        elif self.mode in ("downsample2d", "downsample3d"):
            x = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1)])
            x = _conv2d(self.resample[1], x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])
        return x.transpose(0, 2, 1, 3, 4)


class QwenImageAvgDown3D(nn.Module):
    """Pixel-unshuffle + average downsample shortcut."""

    def __init__(
        self, in_dim: int, out_dim: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_dim * factor % out_dim != 0:
            raise ValueError("in_dim * factor must be divisible by out_dim")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor
        self.group_size = in_dim * factor // out_dim

    def __call__(self, x: mx.array) -> mx.array:
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - x.shape[2] % ft) % ft
        x = mx.pad(x, [(0, 0), (0, 0), (pad_t, 0), (0, 0), (0, 0)])
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.reshape(b, self.out_dim, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(axis=2)


class QwenImageDupUp3D(nn.Module):
    """Duplicate + pixel-shuffle upsample shortcut."""

    def __init__(
        self, in_dim: int, out_dim: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_dim * self.factor % in_dim != 0:
            raise ValueError("out_dim * factor must be divisible by in_dim")
        self.repeats = out_dim * self.factor // in_dim

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        ft, fs = self.factor_t, self.factor_s
        x = mx.repeat(x, self.repeats, axis=1)
        b, _, t, h, w = x.shape
        x = x.reshape(b, self.out_dim, ft, fs, fs, t, h, w)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, self.out_dim, t * ft, h * fs, w * fs)
        if first_chunk:
            x = x[:, :, ft - 1 :, :, :]
        return x


class QwenImageResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = QwenImageRMSNorm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim, images=False)
        self.conv2 = QwenImageCausalConv(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv(in_dim, out_dim, 1) if in_dim != out_dim else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.conv1(nn.silu(self.norm1(x)))
        x = self.conv2(nn.silu(self.norm2(x)))
        return x + h


class QwenImageAttentionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape
        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = _conv2d(self.to_qkv, x)
        qkv = qkv.reshape(b * t, 3 * c, h * w).transpose(0, 2, 1)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q[:, None]
        k = k[:, None]
        v = v[:, None]
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (c**0.5))
        out = out[:, 0].transpose(0, 2, 1).reshape(b * t, c, h, w)
        out = _conv2d(self.proj, out)
        out = out.reshape(b, t, c, h, w).transpose(0, 2, 1, 3, 4)
        return out + identity


class QwenImageMidBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1) -> None:
        super().__init__()
        resnets = [QwenImageResidualBlock(dim, dim)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim))
        self.attentions = attentions
        self.resnets = resnets

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class QwenImageResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = QwenImageAvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(QwenImageResidualBlock(dim, out_dim))
            dim = out_dim
        self.resnets = resnets
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = QwenImageResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def __call__(self, x: mx.array) -> mx.array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)


class QwenImageResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut = (
            QwenImageDupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2
            )
            if up_flag
            else None
        )
        resnets = []
        dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(dim, out_dim))
            dim = out_dim
        self.resnets = resnets
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = QwenImageResample(
                out_dim, mode=mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
        return x


class QwenImageEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temperal_downsample: tuple[bool, ...],
        is_residual: bool,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in (1,) + tuple(dim_mult)]
        self.conv_in = QwenImageCausalConv(in_channels, dims[0], 3, padding=1)
        blocks: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == len(dim_mult) - 1
            if is_residual:
                blocks.append(
                    QwenImageResidualDownBlock(
                        in_dim,
                        out_dim,
                        num_res_blocks,
                        temperal_downsample=(
                            temperal_downsample[i] if not last else False
                        ),
                        down_flag=not last,
                    )
                )
            else:
                cur = in_dim
                for _ in range(num_res_blocks):
                    blocks.append(QwenImageResidualBlock(cur, out_dim))
                    cur = out_dim
                if not last:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    blocks.append(QwenImageResample(out_dim, mode=mode))
        self.down_blocks = blocks
        self.mid_block = QwenImageMidBlock(dims[-1])
        self.norm_out = QwenImageRMSNorm(dims[-1], images=False)
        self.conv_out = QwenImageCausalConv(dims[-1], z_dim, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_out(nn.silu(self.norm_out(x)))
        return x


class QwenImageDecoder3d(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temperal_upsample: tuple[bool, ...],
        out_channels: int,
        is_residual: bool,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in (dim_mult[-1],) + tuple(dim_mult[::-1])]
        self.conv_in = QwenImageCausalConv(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0])
        blocks: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            if is_residual:
                blocks.append(
                    QwenImageResidualUpBlock(
                        in_dim,
                        out_dim,
                        num_res_blocks,
                        temperal_upsample=temperal_upsample[i] if up_flag else False,
                        up_flag=up_flag,
                    )
                )
            else:
                mode = None
                if up_flag:
                    mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                blocks.append(QwenImageUpBlock(in_dim, out_dim, num_res_blocks, mode))
        self.up_blocks = blocks
        self.norm_out = QwenImageRMSNorm(dims[-1], images=False)
        self.conv_out = QwenImageCausalConv(dims[-1], out_channels, 3, padding=1)

    def __call__(self, x: mx.array, first_chunk: bool = True) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            if isinstance(block, QwenImageResidualUpBlock):
                x = block(x, first_chunk=first_chunk)
            else:
                x = block(x)
        x = self.conv_out(nn.silu(self.norm_out(x)))
        return x


class QwenImageUpBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, num_res_blocks: int, upsample_mode: str | None
    ) -> None:
        super().__init__()
        resnets = []
        dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(dim, out_dim))
            dim = out_dim
        self.resnets = resnets
        self.upsamplers = (
            [QwenImageResample(out_dim, mode=upsample_mode)] if upsample_mode else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class QwenImageVAE(nn.Module):
    """AutoencoderKLQwenImage21 restricted to the single-frame image path."""

    def __init__(
        self,
        base_dim: int = 96,
        decoder_base_dim: int = 144,
        z_dim: int = 64,
        dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True, True),
        in_channels: int = 4,
        out_channels: int = 4,
        is_residual: bool = True,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        temperal_upsample = tuple(temperal_downsample[::-1])
        self.encoder = QwenImageEncoder3d(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_downsample=temperal_downsample,
            is_residual=is_residual,
        )
        self.quant_conv = QwenImageCausalConv(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_upsample=temperal_upsample,
            out_channels=out_channels,
            is_residual=is_residual,
        )

    def encode(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Encode ``[B, C, T, H, W]`` pixels to latent (mean, logvar)."""
        h = self.quant_conv(self.encoder(x))
        mean, logvar = mx.split(h, 2, axis=1)
        return mean, logvar

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent ``[B, z_dim, T, H, W]`` to pixels, clamped to [-1, 1]."""
        out = self.decoder(self.post_quant_conv(z), first_chunk=True)
        return mx.clip(out, -1.0, 1.0)


__all__ = ["QwenImageVAE"]
