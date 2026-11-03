"""Float32 image autoencoder with the native BFL checkpoint names (NHWC)."""

import mlx.core as mx
from mlx import nn

from ..flux2.latent import pack_latents, patchify_latents


def norm(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x):
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x)


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def __call__(self, x):
        h = self.norm(x)
        b, height, width, channels = h.shape
        q, k, v = [
            proj(h).reshape(b, 1, height * width, channels)
            for proj in (self.q, self.k, self.v)
        ]
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=channels**-0.5)
        return x + self.proj_out(h.reshape(x.shape))


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels)

    def __call__(self, x):
        return self.block_2(self.attn_1(self.block_1(x)))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def __call__(self, x):
        return self.conv(mx.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0))))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x):
        return self.conv(mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2))


class Level(nn.Module):
    def __init__(self, in_channels, out_channels, *, count, sampling=None):
        super().__init__()
        self.block = [
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(count)
        ]
        if sampling == "down":
            self.downsample = Downsample(out_channels)
        if sampling == "up":
            self.upsample = Upsample(out_channels)

    def __call__(self, x):
        for block in self.block:
            x = block(x)
        if hasattr(self, "downsample"):
            x = self.downsample(x)
        if hasattr(self, "upsample"):
            x = self.upsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (128, 256, 512, 512)
        self.quant_conv = nn.Conv2d(64, 64, 1)
        self.conv_in = nn.Conv2d(3, 128, 3, padding=1)
        self.down = [
            Level(
                channels[i - 1] if i else 128,
                c,
                count=2,
                sampling="down" if i < 3 else None,
            )
            for i, c in enumerate(channels)
        ]
        self.mid = MidBlock(512)
        self.norm_out = norm(512)
        self.conv_out = nn.Conv2d(512, 64, 3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        for level in self.down:
            x = level(x)
        return self.quant_conv(self.conv_out(nn.silu(self.norm_out(self.mid(x)))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (128, 256, 512, 512)
        self.post_quant_conv = nn.Conv2d(32, 32, 1)
        self.conv_in = nn.Conv2d(32, 512, 3, padding=1)
        self.mid = MidBlock(512)
        self.up = [
            Level(channels[min(i + 1, 3)], c, count=3, sampling="up" if i else None)
            for i, c in enumerate(channels)
        ]
        self.norm_out = norm(128)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

    def __call__(self, x):
        x = self.mid(self.conv_in(self.post_quant_conv(x)))
        for level in reversed(self.up):
            x = level(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class BatchNormStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.running_mean = mx.zeros((128,))
        self.running_var = mx.ones((128,))


class Flux3VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.bn = BatchNormStats()

    def encode(self, image):
        """NHWC pixels in [-1, 1] -> normalized (B, H/16*W/16, 128)."""
        mean = self.encoder(image.astype(mx.float32))[..., :32]
        z = patchify_latents(mean.transpose(0, 3, 1, 2))
        z = pack_latents(z)
        return (z - self.bn.running_mean) * mx.rsqrt(self.bn.running_var + 1e-4)

    def decode(self, tokens, *, height, width):
        b = tokens.shape[0]
        h, w = height // 16, width // 16
        z = (
            tokens.astype(mx.float32) * mx.sqrt(self.bn.running_var + 1e-4)
            + self.bn.running_mean
        )
        z = (
            z.reshape(b, h, w, 32, 2, 2)
            .transpose(0, 1, 4, 2, 5, 3)
            .reshape(b, h * 2, w * 2, 32)
        )
        return self.decoder(z)
