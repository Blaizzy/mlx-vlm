"""Apertus 1.5 audio tokenizer: the WavTokenizer encoder.

WavTokenizer's encoder is the SEANet encoder used by EnCodec, followed by a
single Euclidean codebook. Apertus 1.5 consumes audio and emits text, so only
the encode path is ported — the Vocos decoder backbone and the ISTFT head that
make up most of the checkpoint are dropped when sanitizing.

Everything here runs in NLC, MLX's native convolution layout.

Like the image codes, audio code assignment is an argmax, so ``Model`` keeps
these weights in ``float32`` and vetoes quantization for this submodule.
"""

import math
import re
from typing import List

import mlx.core as mx
import mlx.nn as nn

from .config import AudioConfig


def _pad1d(x: mx.array, left: int, right: int, mode: str) -> mx.array:
    """Pad the time axis of an ``(B, L, C)`` array, reflect mode included."""
    if left == 0 and right == 0:
        return x
    if mode != "reflect":
        return mx.pad(x, ((0, 0), (left, right), (0, 0)))

    # Reflect padding needs at least as many samples as the padding width; pad
    # with zeros first when the clip is shorter, then trim them back off.
    extra = 0
    if x.shape[1] <= max(left, right):
        extra = max(left, right) - x.shape[1] + 1
        x = mx.pad(x, ((0, 0), (0, extra), (0, 0)))

    length = x.shape[1]
    parts = []
    if left > 0:
        parts.append(x[:, 1 : left + 1][:, ::-1])
    parts.append(x)
    if right > 0:
        parts.append(x[:, length - right - 1 : length - 1][:, ::-1])
    out = mx.concatenate(parts, axis=1) if len(parts) > 1 else x
    if extra:
        out = out[:, : out.shape[1] - extra]
    return out


class WavConv1d(nn.Module):
    """Conv1d with the asymmetric padding EnCodec/WavTokenizer use."""

    def __init__(
        self,
        config: AudioConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        # Effective kernel size with dilation.
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.kernel_size - stride

    def _extra_padding(self, length: int) -> int:
        """Right padding that makes the last convolution window land exactly."""
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    def __call__(self, x: mx.array) -> mx.array:
        extra = self._extra_padding(x.shape[1])
        if self.causal:
            x = _pad1d(x, self.padding_total, extra, self.pad_mode)
        else:
            # Asymmetric padding, required for odd strides.
            right = self.padding_total // 2
            left = self.padding_total - right
            x = _pad1d(x, left, right + extra, self.pad_mode)
        return self.conv(x)


class WavLSTM(nn.Module):
    def __init__(self, config: AudioConfig, dimension: int):
        super().__init__()
        self.lstm = [
            nn.LSTM(dimension, dimension) for _ in range(config.num_lstm_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for lstm in self.lstm:
            h = lstm(h)[0]
        return h + x


class WavResnetBlock(nn.Module):
    """SEANet residual block."""

    def __init__(self, config: AudioConfig, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden
            # The ELUs occupy list slots, which keeps the checkpoint's
            # `block.1` / `block.3` convolution indices aligned.
            block.append(nn.ELU())
            block.append(
                WavConv1d(
                    config, in_channels, out_channels, kernel_size, dilation=dilation
                )
            )
        self.block = block
        if config.use_conv_shortcut:
            self.shortcut = WavConv1d(config, dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = x
        for layer in self.block:
            h = layer(h)
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        return shortcut + h


class AudioEncoder(nn.Module):
    """SEANet encoder as used by WavTokenizer."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        layers = [
            WavConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            current = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(
                    WavResnetBlock(config, current, [config.dilation_growth_rate**j, 1])
                )
            layers.append(nn.ELU())
            layers.append(
                WavConv1d(
                    config, current, current * 2, kernel_size=ratio * 2, stride=ratio
                )
            )
            scaling *= 2

        layers.append(WavLSTM(config, scaling * config.num_filters))
        layers.append(nn.ELU())
        layers.append(
            WavConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        )
        self.layers = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class EuclideanCodebook(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed = mx.zeros((config.codebook_size, config.codebook_dim))

    def encode(self, x: mx.array) -> mx.array:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        embed = self.embed.T
        distance = -(
            flat.square().sum(axis=1, keepdims=True)
            - 2 * (flat @ embed)
            + embed.square().sum(axis=0, keepdims=True)
        )
        return mx.argmax(distance, axis=-1).reshape(*shape[:-1])


class VectorQuantization(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.codebook = EuclideanCodebook(config)

    def encode(self, x: mx.array) -> mx.array:
        return self.codebook.encode(x)


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.encoder = AudioEncoder(config)
        self.quantizer = VectorQuantization(config)

    @property
    def hop_length(self) -> int:
        return self.config.hop_length

    def encode(self, input_values: mx.array) -> mx.array:
        """Tokenize a mono 24 kHz waveform into discrete codes.

        Args:
            input_values: ``(B, L, 1)`` waveform samples.

        Returns:
            Code ids of shape ``(B, ceil(L / hop_length))``.
        """
        input_values = input_values.astype(self.encoder.layers[0].conv.weight.dtype)
        return self.quantizer.encode(self.encoder(input_values))

    def num_codes(self, num_samples: int) -> int:
        return -(-num_samples // self.hop_length)

    # `parametrizations.weight.original0` is the weight-norm magnitude `g` and
    # `original1` the direction `v`; PyTorch materialises `g * v / ||v||`.
    WEIGHT_NORM_SUFFIX = ".parametrizations.weight.original"
    # EMA statistics kept for codebook training; the lookup only needs `embed`.
    CODEBOOK_BUFFERS = (
        "quantizer.codebook.cluster_size",
        "quantizer.codebook.embed_avg",
        "quantizer.codebook.inited",
    )
    # The Vocos backbone and ISTFT head synthesise audio, which Apertus 1.5
    # never does. They are the bulk of the WavTokenizer checkpoint.
    DECODER_PREFIXES = ("backbone.", "head.")

    def sanitize(self, weights):
        sanitized = {}
        weight_norm_pairs = {}
        lstm_biases = {}
        for k, v in weights.items():
            if k in self.CODEBOOK_BUFFERS or k.startswith(self.DECODER_PREFIXES):
                continue
            if self.WEIGHT_NORM_SUFFIX in k:
                prefix, index = k.rsplit(self.WEIGHT_NORM_SUFFIX, 1)
                weight_norm_pairs.setdefault(prefix, {})[index] = v
                continue
            # PyTorch stacks LSTM layers inside one module and splits the bias
            # in two; MLX uses one module per layer with a single bias.
            match = re.fullmatch(r"(.*\.lstm)\.(weight|bias)_(ih|hh)_l(\d+)", k)
            if match:
                prefix, kind, gate, layer = match.groups()
                if kind == "weight":
                    name = "Wx" if gate == "ih" else "Wh"
                    sanitized[f"{prefix}.{layer}.{name}"] = v
                else:
                    key = f"{prefix}.{layer}.bias"
                    lstm_biases[key] = lstm_biases.get(key, 0) + v
                continue
            sanitized[k] = v

        sanitized.update(lstm_biases)

        for prefix, pair in weight_norm_pairs.items():
            magnitude, direction = pair.get("0"), pair.get("1")
            if magnitude is None or direction is None:
                raise ValueError(
                    f"Incomplete weight-norm parametrization for '{prefix}': "
                    "both `original0` and `original1` are required."
                )
            norm = mx.sqrt(direction.square().sum(axis=(1, 2), keepdims=True))
            sanitized[f"{prefix}.weight"] = magnitude * direction / norm

        if not weight_norm_pairs:
            # No parametrizations means this is an mlx-vlm conversion, whose
            # convolution weights are already in NLC layout.
            return sanitized
        # PyTorch conv1d weights are (out, in, k); MLX wants (out, k, in).
        return {
            k: (v.transpose(0, 2, 1) if k.endswith("conv.weight") else v)
            for k, v in sanitized.items()
        }
