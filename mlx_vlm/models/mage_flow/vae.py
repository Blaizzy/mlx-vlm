from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=32,
        dims=channels,
        eps=1e-6,
        pytorch_compatible=True,
    )


def _nonlinearity(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.linear_1 = nn.Linear(frequency_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, timestep: mx.array, dtype) -> mx.array:
        half = self.frequency_size // 2
        frequencies = mx.exp(
            -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
        )
        args = timestep.reshape(-1, 1).astype(mx.float32) * frequencies[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.linear_2(nn.silu(self.linear_1(embedding.astype(dtype))))


class AdaLNModulation(nn.Module):
    def __init__(self, channels: int, chunks: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, chunks * channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.silu(x))


class DiCoBlock(nn.Module):
    def __init__(self, channels: int = 384, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.ca_conv = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, hidden, 1)
        self.conv5 = nn.Conv2d(hidden, channels, 1)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, affine=False)
        self.adaLN_modulation = AdaLNModulation(channels, 6)

    def __call__(self, inputs: mx.array, conditioning: mx.array) -> mx.array:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = mx.split(self.adaLN_modulation(conditioning), 6, axis=-1)
        x = self.norm1(inputs)
        x = x * (1.0 + scale_attn[:, None, None]) + shift_attn[:, None, None]
        x = nn.gelu(self.conv2(self.conv1(x)))
        channel_attention = mx.sigmoid(
            self.ca_conv(mx.mean(x, axis=(1, 2), keepdims=True))
        )
        x = self.conv3(x * channel_attention)
        x = inputs + gate_attn[:, None, None] * x
        residual = self.norm2(x)
        residual = (
            residual * (1.0 + scale_mlp[:, None, None]) + shift_mlp[:, None, None]
        )
        residual = self.conv5(nn.gelu(self.conv4(residual)))
        return x + gate_mlp[:, None, None] * residual


class EncoderDiCoBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.ca_conv = nn.Conv2d(channels, channels, 1)
        self.conv4 = nn.Conv2d(channels, hidden, 1)
        self.conv5 = nn.Conv2d(hidden, channels, 1)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)

    def __call__(self, inputs: mx.array) -> mx.array:
        x = nn.gelu(self.conv2(self.conv1(self.norm1(inputs))))
        x = x * mx.sigmoid(self.ca_conv(mx.mean(x, axis=(1, 2), keepdims=True)))
        x = inputs + self.conv3(x)
        return x + self.conv5(nn.gelu(self.conv4(self.norm2(x))))


class DConvEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_channels: int = 128,
        hidden_size: int = 384,
        head_size: int = 768,
        patch_size: int = 16,
        num_blocks: int = 21,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(3, head_size, patch_size, stride=patch_size)
        self.head_blocks = [EncoderDiCoBlock(head_size) for _ in range(2)]
        self.proj_down = nn.Conv2d(head_size, hidden_size, 1)
        self.z_proj = nn.Conv2d(latent_channels, hidden_size, 1)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, 1)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = [DiCoBlock(hidden_size) for _ in range(num_blocks)]
        self.norm_out = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj_out = nn.Conv2d(hidden_size, latent_channels * 2, 1)

    def __call__(self, image: mx.array) -> tuple[mx.array, mx.array]:
        batch, height, width, _ = image.shape
        latent = mx.zeros(
            (
                batch,
                height // self.patch_size,
                width // self.patch_size,
                self.latent_channels,
            ),
            dtype=image.dtype,
        )
        condition = self.patch_cond_embed(image)
        for block in self.head_blocks:
            condition = block(condition)
        condition = self.proj_down(condition)
        x = self.fuse_proj(mx.concatenate([condition, self.z_proj(latent)], axis=-1))
        temb = self.t_embedder(mx.zeros((batch,), dtype=image.dtype), image.dtype)
        for block in self.blocks:
            x = block(x, temb)
        mean, logvar = mx.split(self.proj_out(self.norm_out(x)), 2, axis=-1)
        return mean, mx.clip(logvar, -20.0, 10.0)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.conv1(_nonlinearity(self.norm1(x)))
        hidden = self.conv2(_nonlinearity(self.norm2(hidden)))
        return x + hidden


class LocalAttentionBlock(nn.Module):
    def __init__(self, channels: int = 384, patch_size: int = 32) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.norm = _group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def _patchify(self, x: mx.array) -> tuple[mx.array, tuple[int, ...]]:
        batch, height, width, channels = x.shape
        patch = self.patch_size
        padded_h = math.ceil(height / patch) * patch
        padded_w = math.ceil(width / patch) * patch
        if padded_h != height or padded_w != width:
            x = mx.pad(
                x,
                ((0, 0), (0, padded_h - height), (0, padded_w - width), (0, 0)),
                mode="edge",
            )
        rows, cols = padded_h // patch, padded_w // patch
        x = x.reshape(batch, rows, patch, cols, patch, channels)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(
            batch * rows * cols, patch * patch, channels
        )
        return x, (batch, height, width, padded_h, padded_w, rows, cols, channels)

    def _unpatchify(self, x: mx.array, shape: tuple[int, ...]) -> mx.array:
        batch, height, width, padded_h, padded_w, rows, cols, channels = shape
        patch = self.patch_size
        x = x.reshape(batch, rows, cols, patch, patch, channels)
        x = x.transpose(0, 1, 3, 2, 4, 5).reshape(batch, padded_h, padded_w, channels)
        return x[:, :height, :width]

    def __call__(self, x: mx.array) -> mx.array:
        hidden = self.norm(x)
        query, shape = self._patchify(self.q(hidden))
        key, _ = self._patchify(self.k(hidden))
        value, _ = self._patchify(self.v(hidden))
        scores = query.astype(mx.float32) @ key.astype(mx.float32).swapaxes(-1, -2)
        scores = mx.softmax(scores * (self.channels**-0.5), axis=-1)
        hidden = (scores @ value.astype(mx.float32)).astype(value.dtype)
        hidden = self._unpatchify(hidden, shape)
        return x + self.proj_out(hidden)


class CoDDecoder(nn.Module):
    def __init__(self, channels: int = 384, latent_channels: int = 128) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, channels, 3, padding=1)
        self.block = [
            ResnetBlock(channels),
            LocalAttentionBlock(channels),
            ResnetBlock(channels),
            LocalAttentionBlock(channels),
            ResnetBlock(channels),
        ]
        self.norm_out = _group_norm(channels)
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, latent: mx.array) -> mx.array:
        hidden = self.conv_in(latent)
        for block in self.block:
            hidden = block(hidden)
        return self.conv_out(_nonlinearity(self.norm_out(hidden)))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, channels: int = 384, patch_size: int = 16) -> None:
        super().__init__()
        self.proj1 = nn.Conv2d(3, 128, patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(128 + channels, channels, 1)

    def __call__(self, image: mx.array, condition: mx.array) -> mx.array:
        return self.proj2(mx.concatenate([self.proj1(image), condition], axis=-1))


class NerfEmbedder(nn.Module):
    def __init__(self, input_channels: int = 35, max_freqs: int = 8) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.max_freqs = max_freqs
        self.linear = nn.Linear(input_channels + max_freqs**2, 32)

    def __call__(self, x: mx.array) -> mx.array:
        patch_size = int(math.sqrt(x.shape[1]))
        positions = mx.linspace(0, 1, patch_size, dtype=mx.float32)
        pos_y, pos_x = mx.meshgrid(positions, positions, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        frequencies = mx.linspace(0, self.max_freqs, self.max_freqs, dtype=mx.float32)
        fx = frequencies[None, :, None]
        fy = frequencies[None, None, :]
        coefficients = 1.0 / (1.0 + fx * fy)
        dct = (
            mx.cos(math.pi * pos_x * fx) * mx.cos(math.pi * pos_y * fy) * coefficients
        ).reshape(1, patch_size**2, self.max_freqs**2)
        dct = mx.broadcast_to(dct.astype(x.dtype), (x.shape[0], *dct.shape[1:]))
        return self.linear(mx.concatenate([x, dct], axis=-1))


class MLPResBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linear_1 = nn.Linear(channels, channels)
        self.linear_2 = nn.Linear(channels, channels)
        self.adaLN_modulation = AdaLNModulation(channels, 3)

    def __call__(self, x: mx.array, condition: mx.array) -> mx.array:
        shift, scale, gate = mx.split(self.adaLN_modulation(condition), 3, axis=-1)
        hidden = self.in_ln(x) * (1.0 + scale) + shift
        hidden = self.linear_2(nn.silu(self.linear_1(hidden)))
        return x + gate * hidden


class DecoderMLP(nn.Module):
    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(384, patch_size**2 * 32)
        self.input_proj = nn.Linear(32, 32)
        self.res_blocks = [MLPResBlock(32) for _ in range(3)]

    def __call__(self, x: mx.array, condition: mx.array) -> mx.array:
        x = self.input_proj(x)
        condition = self.cond_embed(condition).reshape(
            condition.shape[0], self.patch_size**2, 32
        )
        for block in self.res_blocks:
            x = block(x, condition)
        return x


class FinalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(32, eps=1e-6)
        self.linear = nn.Linear(32, 3)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(self.norm(x))


class YEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = CoDDecoder()


class DConvDenoiser(nn.Module):
    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.t_embedder = TimestepEmbedder(384)
        self.y_embedder_x = nn.Conv2d(384, 32 * patch_size**2, 1)
        self.x_embedder = NerfEmbedder()
        self.s_embedder = BottleneckPatchEmbed()
        self.blocks = [DiCoBlock(384) for _ in range(21)]
        self.dec_net = DecoderMLP(patch_size)
        self.final_layer = FinalLayer()
        self.y_embedder = YEmbedder()

    def __call__(self, condition: mx.array) -> mx.array:
        batch, grid_h, grid_w, _ = condition.shape
        patch = self.patch_size
        height, width = grid_h * patch, grid_w * patch
        noise = mx.zeros((batch, height, width, 3), dtype=condition.dtype)
        temb = self.t_embedder(
            mx.zeros((batch,), dtype=condition.dtype), condition.dtype
        )
        spatial = self.s_embedder(noise, condition)
        for block in self.blocks:
            spatial = block(spatial, temb)
        spatial = spatial.reshape(batch * grid_h * grid_w, 384)

        noise_patches = noise.reshape(batch, grid_h, patch, grid_w, patch, 3).transpose(
            0, 1, 3, 5, 2, 4
        )
        noise_patches = noise_patches.reshape(
            batch, grid_h * grid_w, 3, patch**2
        ).transpose(0, 1, 3, 2)
        cond_patches = (
            self.y_embedder_x(condition)
            .reshape(batch, grid_h * grid_w, 32, patch**2)
            .transpose(0, 1, 3, 2)
        )
        x = mx.concatenate([noise_patches, cond_patches], axis=-1)
        x = x.reshape(batch * grid_h * grid_w, patch**2, 35)
        x = self.x_embedder(x)
        x = self.dec_net(x, spatial)
        x = self.final_layer(x)
        x = x.reshape(batch, grid_h, grid_w, patch, patch, 3)
        return x.transpose(0, 1, 3, 2, 4, 5).reshape(batch, height, width, 3)


class MageVAE(nn.Module):
    latent_channels = 128
    downsample_factor = 16

    def __init__(self, *, include_encoder: bool = True) -> None:
        super().__init__()
        self.dconv_encoder = DConvEncoder() if include_encoder else None
        self.decoder_model = DConvDenoiser()

    def encode(
        self,
        image: mx.array,
        *,
        sample_posterior: bool = True,
        key: mx.array | None = None,
    ) -> mx.array:
        if self.dconv_encoder is None:
            raise RuntimeError("MageVAE was loaded without encoder weights")
        if image.shape[1] % 16 or image.shape[2] % 16:
            raise ValueError(
                "image height and width must be multiples of 16, got "
                f"{image.shape[1:3]}"
            )
        mean, logvar = self.dconv_encoder(image)
        if not sample_posterior:
            return mean
        return mean + mx.exp(0.5 * logvar) * mx.random.normal(
            mean.shape, key=key, dtype=mean.dtype
        )

    def decode(self, latent: mx.array) -> mx.array:
        condition = self.decoder_model.y_embedder.decoder(latent)
        return self.decoder_model(condition)


__all__ = ["MageVAE"]
