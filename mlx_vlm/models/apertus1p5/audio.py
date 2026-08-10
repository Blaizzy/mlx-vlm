"""WavTokenizer audio tokenizer (encode path) for Apertus 1.5.

Ports the SEANet encoder and single Euclidean VQ codebook of the WavTokenizer
neural codec bundled with the Apertus 1.5 checkpoints. Only encoding (waveform
to discrete codes) is implemented: the language model consumes audio as input
tokens and never emits them, so the Vocos-style decoder is not needed and its
weights are dropped in the model's ``sanitize``.

Weight normalization is fused into plain convolution weights at load time, so
the modules here hold ordinary ``nn.Conv1d`` layers. All arrays use the MLX
channel-last convention ``(batch, length, channels)`` internally.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import AudioTokenizerConfig


def _pad1d(
    hidden_states: mx.array, padding_left: int, padding_right: int, mode: str
) -> mx.array:
    """Pad along the length axis, replicating the reference semantics.

    Reflect padding on inputs shorter than the pad width first zero-extends
    the signal so the reflection has enough context, then trims the extra.
    """
    if mode != "reflect":
        return mx.pad(hidden_states, [(0, 0), (padding_left, padding_right), (0, 0)])

    length = hidden_states.shape[1]
    max_pad = max(padding_left, padding_right)
    extra_pad = 0
    if length <= max_pad:
        extra_pad = max_pad - length + 1
        hidden_states = mx.pad(hidden_states, [(0, 0), (0, extra_pad), (0, 0)])
        length = hidden_states.shape[1]

    parts = []
    if padding_left:
        indices = mx.arange(padding_left, 0, -1)
        parts.append(mx.take(hidden_states, indices, axis=1))
    parts.append(hidden_states)
    if padding_right:
        indices = mx.arange(length - 2, length - 2 - padding_right, -1)
        parts.append(mx.take(hidden_states, indices, axis=1))
    padded = mx.concatenate(parts, axis=1) if len(parts) > 1 else hidden_states
    end = padded.shape[1] - extra_pad
    return padded[:, :end]


class AudioConv1d(nn.Module):
    """Conv1d with EnCodec-style asymmetric (or causal) padding."""

    def __init__(
        self,
        config: AudioTokenizerConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.causal = config.use_causal_conv
        self.pad_mode = "constant" if config.pad_mode == "zero" else config.pad_mode
        self.conv_stride = stride
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.effective_kernel_size - stride

    def _extra_padding(self, length: int) -> int:
        # ceil((length - kernel + padding_total) / stride + 1) - 1 frames,
        # padded back up to the ideal input length for the final frame.
        numerator = length - self.effective_kernel_size + self.padding_total
        n_frames = -(-numerator // self.conv_stride)
        ideal_length = (
            n_frames * self.conv_stride
            + self.effective_kernel_size
            - self.padding_total
        )
        return ideal_length - length

    def __call__(self, hidden_states: mx.array) -> mx.array:
        extra_padding = self._extra_padding(hidden_states.shape[1])
        if self.causal:
            hidden_states = _pad1d(
                hidden_states, self.padding_total, extra_padding, self.pad_mode
            )
        else:
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            hidden_states = _pad1d(
                hidden_states,
                padding_left,
                padding_right + extra_padding,
                self.pad_mode,
            )
        return self.conv(hidden_states)


class AudioResnetBlock(nn.Module):
    """SEANet residual block: ELU + conv (bottleneck), with a conv shortcut."""

    def __init__(self, config: AudioTokenizerConfig, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for index, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if index == 0 else hidden
            out_channels = dim if index == len(kernel_sizes) - 1 else hidden
            block.append(nn.ELU())
            block.append(
                AudioConv1d(
                    config, in_channels, out_channels, kernel_size, dilation=dilation
                )
            )
        self.block = block
        if config.use_conv_shortcut:
            self.shortcut = AudioConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return self.shortcut(residual) + hidden_states


class AudioLSTM(nn.Module):
    """Stacked LSTM with a residual connection, in convolutional layout."""

    def __init__(self, config: AudioTokenizerConfig, dimension: int):
        super().__init__()
        self.lstm = [
            nn.LSTM(dimension, dimension) for _ in range(config.num_lstm_layers)
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # (batch, length, channels) already matches the MLX LSTM NLD layout.
        outputs = hidden_states
        for layer in self.lstm:
            outputs = layer(outputs)[0]
        return outputs + hidden_states


class AudioEncoder(nn.Module):
    """SEANet encoder as used by WavTokenizer.

    Layer indices (initial conv, residual blocks, ELUs, strided downsampling
    convs, LSTM, final conv) intentionally mirror the reference ModuleList so
    checkpoint weight names map one-to-one.
    """

    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        layers = [
            AudioConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(
                    AudioResnetBlock(
                        config, current_scale, [config.dilation_growth_rate**j, 1]
                    )
                )
            layers.append(nn.ELU())
            layers.append(
                AudioConv1d(
                    config,
                    current_scale,
                    current_scale * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )
            scaling *= 2
        layers.append(AudioLSTM(config, scaling * config.num_filters))
        layers.append(nn.ELU())
        layers.append(
            AudioConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        )
        self.layers = layers

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class AudioEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.embed = mx.zeros((config.codebook_size, config.codebook_dim))

    def encode(self, hidden_states: mx.array) -> mx.array:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        # Same formula and axis order as the reference; the squared-input term
        # is constant across codes but kept so near-ties resolve identically.
        scaled_states = mx.sum(flat * flat, axis=1, keepdims=True)
        embed = self.embed.T
        distances = -(
            scaled_states
            - 2 * flat @ embed
            + mx.sum(embed * embed, axis=0, keepdims=True)
        )
        codes = mx.argmax(distances, axis=-1)
        return codes.reshape(shape[:-1])


class AudioVectorQuantization(nn.Module):
    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.codebook = AudioEuclideanCodebook(config)

    def encode(self, hidden_states: mx.array) -> mx.array:
        return self.codebook.encode(hidden_states)


class AudioTokenizer(nn.Module):
    """WavTokenizer encode path: mono waveform to discrete codebook indices."""

    def __init__(self, config: AudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.encoder = AudioEncoder(config)
        self.quantizer = AudioVectorQuantization(config)
        self.hop_length = config.hop_length

    def encode(self, input_values: mx.array) -> mx.array:
        """Encode waveforms of shape ``(batch, 1, num_samples)`` into codes.

        Returns an integer array of shape ``(batch, ceil(num_samples / hop))``.
        """
        if input_values.ndim != 3 or input_values.shape[1] != 1:
            raise ValueError(
                "input_values must have shape (batch, 1, num_samples), got "
                f"{input_values.shape}."
            )
        if input_values.shape[-1] == 0:
            raise ValueError("input_values must contain at least one sample.")
        first_conv = self.encoder.layers[0]
        if (
            first_conv.conv.weight.dtype != mx.float32
            or self.quantizer.codebook.embed.dtype != mx.float32
        ):
            raise ValueError(
                "The Apertus 1.5 audio tokenizer weights must remain in float32."
            )

        # (batch, 1, length) -> channel-last (batch, length, 1)
        hidden_states = input_values.astype(mx.float32).transpose(0, 2, 1)
        embeddings = self.encoder(hidden_states)
        return self.quantizer.encode(embeddings)
