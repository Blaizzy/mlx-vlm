from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import MiniMaxH3VideoVAEConfig


@dataclass(frozen=True, slots=True)
class MiniMaxH3VideoVAEOutput:
    sample: mx.array


class MiniMaxH3DiagonalGaussianDistribution:
    def __init__(self, parameters: mx.array) -> None:
        self.parameters = parameters
        self.mean, self.logvar = mx.split(parameters, 2, axis=1)
        self.logvar = mx.clip(self.logvar, -30.0, 20.0)

    def mode(self) -> mx.array:
        return self.mean

    def sample(self, key: mx.array | None = None) -> mx.array:
        noise = mx.random.normal(self.mean.shape, key=key)
        return self.mean + mx.exp(0.5 * self.logvar) * noise


def _reflect_pad_axis(
    hidden_states: mx.array,
    axis: int,
    before: int,
    after: int,
) -> mx.array:
    if before == 0 and after == 0:
        return hidden_states
    axis = axis % hidden_states.ndim
    length = hidden_states.shape[axis]
    if before >= length or after >= length:
        raise ValueError("reflect padding must be smaller than its input dimension")
    indices = [*range(before, 0, -1), *range(length)]
    indices.extend(range(length - 2, length - after - 2, -1))
    return mx.take(hidden_states, mx.array(indices, dtype=mx.int32), axis=axis)


class MiniMaxH3VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode
        scale = math.sqrt(1 / (in_channels * math.prod(kernel_size)))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.spatial_padding:
            if self.spatial_padding_mode != "reflect":
                raise ValueError("only reflect spatial padding is supported")
            hidden_states = _reflect_pad_axis(
                hidden_states, -2, self.spatial_padding, self.spatial_padding
            )
            hidden_states = _reflect_pad_axis(
                hidden_states, -1, self.spatial_padding, self.spatial_padding
            )
        if self.temporal_padding:
            hidden_states = mx.pad(
                hidden_states,
                ((0, 0), (0, 0), (self.temporal_padding, 0), (0, 0), (0, 0)),
            )
        values = hidden_states.transpose(0, 2, 3, 4, 1)
        values = mx.conv3d(values, self.weight, stride=self.stride)
        values = values + self.bias
        return values.transpose(0, 4, 1, 2, 3)


class MiniMaxH3VideoGroupNorm(nn.Module):
    def __init__(self, num_groups: int, channels: int, eps: float) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, channels, frames, height, width = hidden_states.shape
        values = hidden_states.transpose(0, 2, 3, 4, 1).reshape(
            batch * frames, height, width, channels
        )
        group_size = channels // self.num_groups
        values = values.reshape(batch * frames, -1, self.num_groups, group_size)
        values = values.transpose(0, 2, 1, 3).reshape(
            batch * frames, self.num_groups, -1
        )
        values = mx.fast.layer_norm(
            values,
            weight=None,
            bias=None,
            eps=self.eps,
        )
        values = values.reshape(batch * frames, self.num_groups, -1, group_size)
        values = values.transpose(0, 2, 1, 3).reshape(
            batch * frames, height, width, channels
        )
        values = values * self.weight + self.bias
        return values.reshape(batch, frames, height, width, channels).transpose(
            0, 4, 1, 2, 3
        )


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, eps: float):
        super().__init__()
        self.norm1 = MiniMaxH3VideoGroupNorm(groups, in_channels, eps)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels, out_channels, 3, spatial_padding=1, temporal_padding=2
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(groups, out_channels, eps)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels, out_channels, 3, spatial_padding=1, temporal_padding=2
        )
        self.conv_shortcut = (
            MiniMaxH3VideoCausalConv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.conv1(nn.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(nn.silu(self.norm2(hidden_states)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    def __init__(self, channels: int, temporal_stride: int, spatial_stride: int):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.conv = MiniMaxH3VideoCausalConv3d(
            channels,
            channels,
            3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            temporal_padding=2,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.spatial_stride == 2:
            hidden_states = _reflect_pad_axis(hidden_states, -2, 0, 1)
            hidden_states = _reflect_pad_axis(hidden_states, -1, 0, 1)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_factor: int,
        spatial_factor: int,
        groups: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.resnets = [
            MiniMaxH3VideoResnetBlock3d(
                in_channels if index == 0 else out_channels,
                out_channels,
                groups,
                eps,
            )
            for index in range(num_layers)
        ]
        self.downsamplers = (
            [MiniMaxH3VideoDownsample3d(out_channels, temporal_factor, spatial_factor)]
            if temporal_factor * spatial_factor > 1
            else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for downsampler in self.downsamplers or []:
            hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__()
        channels = config.block_out_channels
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            config.in_channels, channels[0], 3, spatial_padding=1, temporal_padding=2
        )
        block_in = (channels[0], *channels[:-1])
        self.down_blocks = [
            MiniMaxH3VideoDownBlock3d(
                block_in[index],
                channels[index],
                config.layers_per_block,
                config.temporal_downsample_factors[index],
                config.spatial_downsample_factors[index],
                config.norm_num_groups,
                config.norm_eps,
            )
            for index in range(len(channels))
        ]
        self.norm_out = MiniMaxH3VideoGroupNorm(
            config.norm_num_groups, channels[-1], config.norm_eps
        )
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            channels[-1],
            2 * config.latent_channels,
            3,
            spatial_padding=1,
            temporal_padding=2,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_in(hidden_states)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        return self.conv_out(nn.silu(self.norm_out(hidden_states)))


def _rms_norm(hidden_states: mx.array, eps: float) -> mx.array:
    values = hidden_states.astype(mx.float32)
    values = values * mx.rsqrt(mx.mean(values * values, axis=-1, keepdims=True) + eps)
    return values.astype(hidden_states.dtype)


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / mx.power(
            self.theta,
            mx.arange(self.dim // 6, dtype=mx.float32) * (6.0 / self.dim),
        )
        angles = 2.0 * math.pi * position_ids[..., None] * inv_freq
        angles = angles.reshape(*position_ids.shape[:-1], -1)
        angles = mx.tile(angles, (1, 1, 2))[:, :, None]
        return mx.cos(angles), mx.sin(angles)


def _apply_video_rope(
    hidden_states: mx.array, cos: mx.array, sin: mx.array
) -> mx.array:
    rotary_dim = cos.shape[-1]
    rotary, passthrough = (
        hidden_states[..., :rotary_dim],
        hidden_states[..., rotary_dim:],
    )
    first, second = mx.split(rotary, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    rotary = rotary * cos.astype(rotary.dtype) + rotated * sin.astype(rotary.dtype)
    return mx.concatenate([rotary, passthrough], axis=-1)


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_k = nn.Linear(dim, inner_dim, bias=True)
        self.to_v = nn.Linear(dim, inner_dim, bias=True)
        self.to_out = [nn.Linear(inner_dim, dim, bias=True), nn.Dropout(0.0)]
        self.eps = eps

    def __call__(
        self,
        hidden_states: mx.array,
        rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(
            batch, length, self.heads, self.dim_head
        )
        key = self.to_k(hidden_states).reshape(batch, length, self.heads, self.dim_head)
        value = self.to_v(hidden_states).reshape(
            batch, length, self.heads, self.dim_head
        )
        query = _rms_norm(query, self.eps)
        key = _rms_norm(key, self.eps)
        if rotary_emb is not None:
            query = _apply_video_rope(query, *rotary_emb)
            key = _apply_video_rope(key, *rotary_emb)
        output = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.dim_head**-0.5,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out[1](self.to_out[0](output))


class _VideoSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 2 * hidden_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states, gate = mx.split(self.proj(hidden_states), 2, axis=-1)
        return hidden_states * nn.silu(gate)


class _Identity(nn.Module):
    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states


class MiniMaxH3VideoFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.net = [
            _VideoSwiGLU(dim, dim * mult),
            _Identity(),
            nn.Linear(dim * mult, dim, bias=True),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mult: int, eps: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = MiniMaxH3VideoAttention(dim, heads, dim_head, eps)
        self.scale1 = mx.zeros((dim,))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = MiniMaxH3VideoFeedForward(dim, mult)
        self.scale2 = mx.zeros((dim,))

    def __call__(
        self, hidden_states: mx.array, rotary_emb: tuple[mx.array, mx.array]
    ) -> mx.array:
        norm = self.norm1(hidden_states.astype(mx.float32)).astype(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(norm, rotary_emb) * self.scale1
        norm = self.norm2(hidden_states.astype(mx.float32)).astype(hidden_states.dtype)
        return hidden_states + self.ff(norm) * self.scale2


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    def __init__(self, config: MiniMaxH3VideoVAEConfig) -> None:
        super().__init__()
        dim = config.decoder_num_attention_heads * config.decoder_attention_head_dim
        self.patch_size = config.spatial_compression_ratio
        self.patch_size_t = config.temporal_compression_ratio
        self.out_channels = config.out_channels
        self.num_register_tokens = config.decoder_num_register_tokens
        rope_dim = int(
            config.decoder_attention_head_dim * config.decoder_rope_dim_ratio
        )
        self.rope = MiniMaxH3VideoRotaryPosEmbed(rope_dim, config.decoder_rope_theta)
        self.proj_in = nn.Linear(config.latent_channels, dim, bias=True)
        self.register_tokens = mx.zeros(
            (1, config.decoder_num_register_tokens, dim), dtype=mx.float32
        )
        self.transformer_blocks = [
            MiniMaxH3VideoTransformerBlock(
                dim,
                config.decoder_num_attention_heads,
                config.decoder_attention_head_dim,
                config.decoder_ffn_mult,
                config.decoder_norm_eps,
            )
            for _ in range(config.decoder_num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=config.decoder_norm_eps)
        self.proj_out = nn.Linear(
            dim,
            config.out_channels * self.patch_size_t * self.patch_size * self.patch_size,
            bias=True,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1).reshape(
            batch, frames * height * width, channels
        )
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        registers = mx.broadcast_to(
            self.register_tokens,
            (batch, self.num_register_tokens, hidden_states.shape[-1]),
        )
        cls_token = mx.zeros_like(hidden_states[:, :1])
        hidden_states = mx.concatenate([hidden_states, registers, cls_token], axis=1)

        grids = [
            2.0 * ((mx.arange(size, dtype=mx.float32) + 0.5) / size) - 1.0
            for size in (frames, height, width)
        ]
        mesh = mx.meshgrid(*grids, indexing="ij")
        position_ids = mx.stack(mesh, axis=-1).reshape(-1, 3)
        position_ids = mx.broadcast_to(position_ids[None], (batch, num_patches, 3))
        suffix = mx.zeros((batch, self.num_register_tokens + 1, 3), mx.float32)
        rotary_emb = self.rope(mx.concatenate([position_ids, suffix], axis=1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)
        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]

        patch, patch_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.reshape(
            batch,
            frames,
            height,
            width,
            self.out_channels,
            patch_t,
            patch,
            patch,
        )
        hidden_states = hidden_states.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.reshape(
            batch,
            self.out_channels,
            frames * patch_t,
            height * patch,
            width * patch,
        )


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(
        self, config: MiniMaxH3VideoVAEConfig | None = None, **config_kwargs
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise ValueError("pass a config or keyword fields, not both")
        self.config = config or MiniMaxH3VideoVAEConfig(**config_kwargs)
        config = self.config
        self.encoder = MiniMaxH3VideoEncoder3d(config)
        self.quant_conv = MiniMaxH3VideoCausalConv3d(
            2 * config.latent_channels, 2 * config.latent_channels, 1
        )
        self.post_quant_conv = MiniMaxH3VideoCausalConv3d(
            config.latent_channels, config.latent_channels, 1
        )
        self.decoder = MiniMaxH3VideoViTDecoder3d(config)
        ratio = config.temporal_compression_ratio
        self.frame_pre_padding = (-config.clip_length) % ratio
        self.tokens_chunk_size = math.ceil(config.clip_length / ratio)
        self.token_overlap = (-config.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * ratio - self.frame_pre_padding, 0)
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = (
            tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        )
        self.tile_sample_min_overlap_width = (
            tile_sample_min_overlap_width or self.tile_sample_min_overlap_width
        )

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def _split_tiles(
        self, length: int, tile_size: int, min_overlap: int
    ) -> tuple[list[int], list[int], list[int]]:
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.config.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.config.spatial_compression_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    def _stitch_tiles(
        self,
        tiles: list[list[mx.array]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> mx.array:
        result_rows = []
        for row_index, row in enumerate(tiles):
            result_row = []
            for column_index, tile in enumerate(row):
                if row_index > 0:
                    tile = self._blend(
                        tiles[row_index - 1][column_index],
                        tile,
                        height_overlaps[row_index - 1],
                        3,
                    )
                if column_index > 0:
                    tile = self._blend(
                        row[column_index - 1],
                        tile,
                        width_overlaps[column_index - 1],
                        4,
                    )
                if row_index < len(tiles) - 1:
                    tile = tile[:, :, :, : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[:, :, :, :, : -width_overlaps[column_index]]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=4))
        return mx.concatenate(result_rows, axis=3)

    def _encode_clip(self, hidden_states: mx.array) -> mx.array:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(hidden_states))
        height, width = hidden_states.shape[-2:]
        y_starts, y_lengths, y_overlaps = self._split_tiles(
            height,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        x_starts, x_lengths, x_overlaps = self._split_tiles(
            width,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        tiles = [
            [
                self.quant_conv(
                    self.encoder(
                        hidden_states[
                            ...,
                            y_start : y_start + y_length,
                            x_start : x_start + x_length,
                        ]
                    )
                )
                for x_start, x_length in zip(x_starts, x_lengths)
            ]
            for y_start, y_length in zip(y_starts, y_lengths)
        ]
        ratio = self.config.spatial_compression_ratio
        return self._stitch_tiles(
            tiles,
            [overlap // ratio for overlap in y_overlaps],
            [overlap // ratio for overlap in x_overlaps],
        )

    def _decode_clip(self, latents: mx.array) -> mx.array:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(latents))
        ratio = self.config.spatial_compression_ratio
        height, width = latents.shape[-2] * ratio, latents.shape[-1] * ratio
        y_starts, y_lengths, y_overlaps = self._split_tiles(
            height,
            self.tile_sample_min_height,
            self.tile_sample_min_overlap_height,
        )
        x_starts, x_lengths, x_overlaps = self._split_tiles(
            width,
            self.tile_sample_min_width,
            self.tile_sample_min_overlap_width,
        )
        tiles = [
            [
                self.decoder(
                    self.post_quant_conv(
                        latents[
                            ...,
                            y_start // ratio : y_start // ratio + y_length // ratio,
                            x_start // ratio : x_start // ratio + x_length // ratio,
                        ]
                    )
                )
                for x_start, x_length in zip(x_starts, x_lengths)
            ]
            for y_start, y_length in zip(y_starts, y_lengths)
        ]
        return self._stitch_tiles(tiles, y_overlaps, x_overlaps)

    def _encode(self, hidden_states: mx.array) -> mx.array:
        clip_length = self.config.clip_length
        num_frames = hidden_states.shape[2]
        if num_frames % clip_length:
            padding = (-num_frames) % clip_length
            hidden_states = mx.concatenate(
                [hidden_states, mx.repeat(hidden_states[:, :, -1:], padding, axis=2)],
                axis=2,
            )
        moments = mx.concatenate(
            [
                self._encode_clip(hidden_states[:, :, index : index + clip_length])
                for index in range(0, hidden_states.shape[2], clip_length)
            ],
            axis=2,
        )
        if self.config.token_drop:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    @staticmethod
    def _blend(left: mx.array, right: mx.array, extent: int, axis: int) -> mx.array:
        extent = min(left.shape[axis], right.shape[axis], extent)
        positions = mx.arange(extent, dtype=right.dtype)
        shape = [1] * right.ndim
        shape[axis] = extent
        left_weight = (1.0 - positions / extent).reshape(shape)
        right_weight = (positions / extent).reshape(shape)
        left_slice = [slice(None)] * left.ndim
        right_slice = [slice(None)] * right.ndim
        left_slice[axis] = slice(-extent, None)
        right_slice[axis] = slice(0, extent)
        blended = (
            left[tuple(left_slice)] * left_weight
            + right[tuple(right_slice)] * right_weight
        )
        if extent == right.shape[axis]:
            return blended
        right_slice[axis] = slice(extent, None)
        return mx.concatenate([blended, right[tuple(right_slice)]], axis=axis)

    def _decode(self, latents: mx.array) -> mx.array:
        chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        ratio = self.config.temporal_compression_ratio
        chunk_frames = chunk_size * ratio
        num_tokens = latents.shape[2] + token_drop
        pad_tokens = (-num_tokens) % chunk_size
        num_chunks = (num_tokens + pad_tokens) // chunk_size - int(token_drop > 0)
        if pad_tokens:
            latents = mx.concatenate(
                [latents, mx.repeat(latents[:, :, -1:], pad_tokens, axis=2)], axis=2
            )
        chunks: list[mx.array] = []
        overlap = None
        for index in range(num_chunks):
            start = index * chunk_size
            clip = self._decode_clip(
                latents[:, :, start : start + chunk_size + self.token_overlap]
            )
            for offset in range(int(token_drop > 0) + 1):
                frame_start = offset * chunk_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if offset == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, 2)
                    chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            chunks.append(overlap)
        decoded = mx.concatenate(chunks, axis=2)
        if pad_tokens:
            intra_tail = self.config.clip_length % ratio
            before_pad = latents.shape[2] - pad_tokens
            pad_frames = sum(
                (
                    intra_tail
                    if intra_tail and (before_pad + index) % chunk_size == 0
                    else ratio
                )
                for index in range(pad_tokens)
            )
            decoded = decoded[:, :, :-pad_frames]
        return decoded

    def encode(self, hidden_states: mx.array) -> MiniMaxH3DiagonalGaussianDistribution:
        return MiniMaxH3DiagonalGaussianDistribution(self._encode(hidden_states))

    def decode(self, latents: mx.array) -> MiniMaxH3VideoVAEOutput:
        return MiniMaxH3VideoVAEOutput(self._decode(latents))

    def __call__(self, hidden_states: mx.array) -> MiniMaxH3VideoVAEOutput:
        return self.decode(self.encode(hidden_states).mode())

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return {
            key: value.transpose(0, 2, 3, 4, 1) if value.ndim == 5 else value
            for key, value in weights.items()
        }


__all__ = [
    "MiniMaxH3DiagonalGaussianDistribution",
    "MiniMaxH3VideoVAE",
    "MiniMaxH3VideoVAEOutput",
]
