import mlx.core as mx
import mlx.nn as nn

from .config import VisionTokenizerConfig


class ResnetBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int | None = None, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            32, self.out_channels, eps=1e-6, pytorch_compatible=True
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

    @staticmethod
    def _silu(hidden_states: mx.array) -> mx.array:
        return hidden_states * mx.sigmoid(hidden_states)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.conv1(self._silu(self.norm1(hidden_states)))
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.dropout(self._silu(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, pytorch_compatible=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch, height, width, channels = query.shape
        query = query.reshape(batch, height * width, channels)
        key = key.reshape(batch, height * width, channels).transpose(0, 2, 1)
        attention = mx.softmax((query @ key) * channels**-0.5, axis=-1)
        value = value.reshape(batch, height * width, channels)
        hidden_states = (attention @ value).reshape(batch, height, width, channels)
        return residual + self.proj_out(hidden_states)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.pad(hidden_states, ((0, 0), (0, 1), (0, 1), (0, 0)))
        return self.conv(hidden_states)


class DownBlock(nn.Module):
    def __init__(self, block, attn, downsample=None):
        super().__init__()
        self.block = block
        self.attn = attn
        if downsample is not None:
            self.downsample = downsample


class MidBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels, dropout)
        self.attn_1 = AttentionBlock(channels)
        self.block_2 = ResnetBlock(channels, channels, dropout)


class Encoder(nn.Module):
    def __init__(self, config: VisionTokenizerConfig):
        super().__init__()
        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        self.conv_in = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        current_resolution = config.resolution
        input_multipliers = (1,) + tuple(config.channel_multiplier)
        self.down = []
        block_out = config.base_channels
        for level in range(self.num_resolutions):
            block_in = config.base_channels * input_multipliers[level]
            block_out = config.base_channels * config.channel_multiplier[level]
            blocks = []
            attention = []
            for _ in range(config.num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, config.dropout))
                block_in = block_out
                if current_resolution in config.attn_resolutions:
                    attention.append(AttentionBlock(block_in))

            downsample = None
            if level != self.num_resolutions - 1:
                downsample = Downsample(block_in)
                current_resolution //= 2
            self.down.append(DownBlock(blocks, attention, downsample))

        self.mid = MidBlock(block_out, config.dropout)
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(
            block_out, config.latent_channels, kernel_size=3, padding=1
        )

    def __call__(self, pixel_values: mx.array) -> mx.array:
        hidden_states = self.conv_in(pixel_values)
        for level, down in enumerate(self.down):
            for block_index, block in enumerate(down.block):
                hidden_states = block(hidden_states)
                if down.attn:
                    hidden_states = down.attn[block_index](hidden_states)
            if level != self.num_resolutions - 1:
                hidden_states = down.downsample(hidden_states)

        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * mx.sigmoid(hidden_states)
        return self.conv_out(hidden_states)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        config: VisionTokenizerConfig,
        max_positions_per_chunk: int = 512,
    ):
        super().__init__()
        if max_positions_per_chunk <= 0:
            raise ValueError("max_positions_per_chunk must be positive")
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        self.max_positions_per_chunk = max_positions_per_chunk

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size, height, width, _ = hidden_states.shape
        column_chunk_size = min(width, self.max_positions_per_chunk)
        rows_per_chunk = max(1, self.max_positions_per_chunk // column_chunk_size)
        num_chunks = (
            batch_size
            * ((height + rows_per_chunk - 1) // rows_per_chunk)
            * ((width + column_chunk_size - 1) // column_chunk_size)
        )
        batch_codes = []

        # Preserve the reference einsum and argmax axes while bounding the
        # temporary score tensor. Spatial chunks are independent, so no
        # cross-chunk maximum or tie bookkeeping is needed.
        for batch_index in range(batch_size):
            row_codes = []
            sample = hidden_states[batch_index : batch_index + 1]
            for row_start in range(0, height, rows_per_chunk):
                column_codes = []
                row_end = min(row_start + rows_per_chunk, height)
                for column_start in range(0, width, column_chunk_size):
                    column_end = min(column_start + column_chunk_size, width)
                    block = sample[:, row_start:row_end, column_start:column_end, :]
                    logits = mx.einsum("bhwd,nd->bnhw", block, self.embedding.weight)
                    codes = mx.argmax(logits, axis=1)
                    if num_chunks > 1:
                        mx.eval(codes)  # Release score storage before the next chunk.
                    column_codes.append(codes)
                row_codes.append(mx.concatenate(column_codes, axis=2))
            batch_codes.append(mx.concatenate(row_codes, axis=1))
        return mx.concatenate(batch_codes, axis=0)


def ensure_channels_last(pixel_values: mx.array, in_channels: int) -> mx.array:
    if pixel_values.shape[-1] == in_channels:
        return pixel_values
    if pixel_values.shape[1] == in_channels:
        return pixel_values.transpose(0, 2, 3, 1)
    raise ValueError(f"pixel_values must have {in_channels} channels")


class VisionTokenizer(nn.Module):
    def __init__(self, config: VisionTokenizerConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.encoder = Encoder(config)
        self.quantize = VectorQuantizer(config)
        self.quant_conv = nn.Conv2d(
            config.latent_channels, config.embed_dim, kernel_size=1
        )
        self.vision_spatial_factor = config.spatial_scale_factor

    def encode(self, pixel_values: mx.array) -> mx.array:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a rank-4 NCHW or NHWC array")
        channel_last = ensure_channels_last(pixel_values, self.config.in_channels)

        height, width = channel_last.shape[1:3]
        factor = self.vision_spatial_factor
        if height <= 0 or width <= 0 or height % factor or width % factor:
            raise ValueError(
                "Image height and width must be positive multiples of the vision "
                f"tokenizer's spatial factor ({factor}), got {height}x{width}."
            )
        if (
            self.encoder.conv_in.weight.dtype != mx.float32
            or self.quantize.embedding.weight.dtype != mx.float32
        ):
            raise ValueError(
                "The Apertus 1.5 vision tokenizer weights must remain in float32."
            )

        hidden_states = self.encoder(channel_last.astype(mx.float32))
        hidden_states = self.quant_conv(hidden_states)
        return self.quantize(hidden_states)
