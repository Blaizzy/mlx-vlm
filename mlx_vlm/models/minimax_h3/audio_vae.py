from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import MiniMaxH3AudioVAEConfig


@dataclass(frozen=True, slots=True)
class MiniMaxH3AudioVAEOutput:
    sample: mx.array


class MiniMaxH3AudioDiagonalGaussianDistribution:
    def __init__(self, mean: mx.array, logs: mx.array) -> None:
        self.mean = mean
        self.logs = logs
        self.std = mx.exp(logs)

    def mode(self) -> mx.array:
        return self.mean

    def sample(self, key: mx.array | None = None) -> mx.array:
        return self.mean + self.std * mx.random.normal(self.mean.shape, key=key)


class MiniMaxH3AudioConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, hidden_states: mx.array) -> mx.array:
        values = mx.conv1d(
            hidden_states.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if "bias" in self:
            values = values + self.bias
        return values.transpose(0, 2, 1)


class MiniMaxH3AudioWeightNormConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        # Diffusers stores ``weight_g`` and ``weight_v``. The H3 loader folds
        # them once so inference does not retain a duplicate convolution
        # kernel or recompute its norm on every call.
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, hidden_states: mx.array) -> mx.array:
        values = mx.conv1d(
            hidden_states.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if "bias" in self:
            values = values + self.bias
        return values.transpose(0, 2, 1)


class MiniMaxH3AudioWeightNormConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        self.bias = mx.zeros((out_channels,))
        self.stride = stride
        self.padding = padding

    def __call__(self, hidden_states: mx.array) -> mx.array:
        values = mx.conv_transpose1d(
            hidden_states.transpose(0, 2, 1),
            self.weight,
            stride=self.stride,
            padding=self.padding,
        )
        return (values + self.bias).transpose(0, 2, 1)


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> mx.array:
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    def bessel_i0(values: mx.array) -> mx.array:
        argument = values * values / 4.0
        result = mx.ones_like(argument)
        term = mx.ones_like(argument)
        # The H3 filters have beta below 8; this converges to float64 precision
        # with ample margin while keeping the construction entirely in MLX.
        for index in range(1, 33):
            term = term * argument / (index * index)
            result = result + term
        return result

    with mx.stream(mx.cpu):
        positions = mx.arange(kernel_size, dtype=mx.float64)
        normalized = 2.0 * positions / (kernel_size - 1) - 1.0
        beta_value = mx.array(beta, dtype=mx.float64)
        window = bessel_i0(beta_value * mx.sqrt(1.0 - normalized * normalized))
        window = window / bessel_i0(beta_value)
        if kernel_size % 2 == 0:
            time = mx.arange(-half_size, half_size, dtype=mx.float64) + 0.5
        else:
            time = mx.arange(kernel_size, dtype=mx.float64) - half_size
        sinc_argument = 2 * cutoff * time
        pi_argument = math.pi * sinc_argument
        sinc = mx.where(
            mx.abs(pi_argument) < 1e-12,
            1.0,
            mx.sin(pi_argument) / pi_argument,
        )
        filter_values = 2 * cutoff * window * sinc
        filter_values = filter_values / mx.sum(filter_values)
        filter_values = filter_values.astype(mx.float32).reshape(1, 1, kernel_size)
        mx.eval(filter_values)
    return filter_values


class MiniMaxH3AudioSnake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return hidden_states + mx.sin(self.alpha * hidden_states) ** 2 / (
            self.alpha + 1e-9
        )


class MiniMaxH3AudioSnakeBeta(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)[None, :, None]
        beta = mx.exp(self.beta)[None, :, None]
        return hidden_states + mx.sin(alpha * hidden_states) ** 2 / (beta + 1e-9)


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    def __init__(
        self, cutoff: float, half_width: float, stride: int, kernel_size: int
    ) -> None:
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        channels = hidden_states.shape[1]
        hidden_states = mx.pad(
            hidden_states,
            ((0, 0), (0, 0), (self.pad_left, self.pad_right)),
            mode="edge",
        )
        kernel = mx.broadcast_to(
            self.filter.transpose(0, 2, 1),
            (channels, self.filter.shape[-1], 1),
        )
        values = mx.conv1d(
            hidden_states.transpose(0, 2, 1),
            kernel,
            stride=self.stride,
            groups=channels,
        )
        return values.transpose(0, 2, 1)


class MiniMaxH3AudioUpSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=kernel_size,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        channels = hidden_states.shape[1]
        hidden_states = mx.pad(
            hidden_states,
            ((0, 0), (0, 0), (self.pad, self.pad)),
            mode="edge",
        )
        kernel = mx.broadcast_to(
            self.filter.transpose(0, 2, 1),
            (channels, self.filter.shape[-1], 1),
        )
        values = self.ratio * mx.conv_transpose1d(
            hidden_states.transpose(0, 2, 1),
            kernel,
            stride=self.stride,
            groups=channels,
        )
        values = values.transpose(0, 2, 1)
        return values[..., self.pad_left : -self.pad_right]


class MiniMaxH3AudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int) -> None:
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.lowpass(hidden_states)


class MiniMaxH3AudioActivation1d(nn.Module):
    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(hidden_states)))


class MiniMaxH3AudioResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.block = [
            MiniMaxH3AudioSnake1d(dim),
            MiniMaxH3AudioWeightNormConv1d(
                dim,
                dim,
                7,
                dilation=dilation,
                padding=3 * dilation,
            ),
            MiniMaxH3AudioSnake1d(dim),
            MiniMaxH3AudioWeightNormConv1d(dim, dim, 1),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        for layer in self.block:
            residual = layer(residual)
        pad = (hidden_states.shape[-1] - residual.shape[-1]) // 2
        if pad > 0:
            hidden_states = hidden_states[..., pad:-pad]
        return hidden_states + residual


class MiniMaxH3AudioEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int) -> None:
        super().__init__()
        channels = dim // 2
        self.block = [
            MiniMaxH3AudioResidualUnit(channels, 1),
            MiniMaxH3AudioResidualUnit(channels, 3),
            MiniMaxH3AudioResidualUnit(channels, 9),
            MiniMaxH3AudioSnake1d(channels),
            MiniMaxH3AudioWeightNormConv1d(
                channels,
                dim,
                2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return hidden_states


class MiniMaxH3AudioEncoder(nn.Module):
    def __init__(self, dim: int, rates: tuple[int, ...], latent_dim: int) -> None:
        super().__init__()
        self.block: list[nn.Module] = [
            MiniMaxH3AudioWeightNormConv1d(1, dim, 7, padding=3)
        ]
        for rate in rates:
            dim *= 2
            self.block.append(MiniMaxH3AudioEncoderBlock(dim, rate))
        self.block.extend(
            [
                MiniMaxH3AudioSnake1d(dim),
                MiniMaxH3AudioWeightNormConv1d(dim, latent_dim, 3, padding=1),
            ]
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return hidden_states


class MiniMaxH3AudioGeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.norm(hidden_states)
        return self.w2(nn.gelu_approx(self.w0(hidden_states)) * self.w1(hidden_states))


def _adaptive_avg_pool_last(hidden_states: mx.array, output_size: int) -> mx.array:
    input_size = hidden_states.shape[-1]
    outputs = []
    for index in range(output_size):
        start = math.floor(index * input_size / output_size)
        end = math.ceil((index + 1) * input_size / output_size)
        outputs.append(mx.mean(hidden_states[..., start:end], axis=-1))
    return mx.stack(outputs, axis=-1)


class MiniMaxH3AudioCausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = mx.zeros((in_dim,))
        self.v_bias = mx.zeros((in_dim,))
        self.zero_k_bias = mx.zeros((in_dim,))
        self.proj = nn.Linear(out_dim, out_dim)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, length, _ = hidden_states.shape
        bias = mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias])
        qkv = (self.qkv(hidden_states) + bias).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = [qkv[:, :, index] for index in range(3)]
        attended = mx.fast.scaled_dot_product_attention(
            query.transpose(0, 2, 1, 3),
            key.transpose(0, 2, 1, 3),
            value.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
            mask="causal",
        )
        attended = attended.transpose(0, 2, 1, 3).mean(axis=2)
        return self.proj(_adaptive_avg_pool_last(attended, self.out_dim))


class MiniMaxH3AudioAttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(out_dim, out_dim * 2)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(self.norm3(hidden_states)) + self.attn(
            self.norm1(hidden_states)
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3AudioAMPBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            MiniMaxH3AudioWeightNormConv1d(
                channels,
                channels,
                kernel_size,
                dilation=value,
                padding=(kernel_size * value - value) // 2,
            )
            for value in dilation
        ]
        self.convs2 = [
            MiniMaxH3AudioWeightNormConv1d(
                channels, channels, kernel_size, padding=(kernel_size - 1) // 2
            )
            for _ in dilation
        ]
        self.activations = [
            MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
            for _ in range(2 * len(dilation))
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for index, (conv1, conv2) in enumerate(zip(self.convs1, self.convs2)):
            residual = conv1(self.activations[2 * index](hidden_states))
            residual = conv2(self.activations[2 * index + 1](residual))
            hidden_states = hidden_states + residual
        return hidden_states


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    def __init__(self, config: MiniMaxH3AudioVAEConfig) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.decoder_rates)
        self.conv_pre = MiniMaxH3AudioWeightNormConv1d(
            config.latent_dim, config.decoder_dim, 7, padding=3
        )
        self.ups = [
            [
                MiniMaxH3AudioWeightNormConvTranspose1d(
                    config.decoder_dim // (2**index),
                    config.decoder_dim // (2 ** (index + 1)),
                    kernel,
                    rate,
                    padding=(kernel - rate) // 2,
                )
            ]
            for index, (rate, kernel) in enumerate(
                zip(config.decoder_rates, config.decoder_kernel_sizes)
            )
        ]
        self.resblocks = []
        for index in range(self.num_upsamples):
            channels = config.decoder_dim // (2 ** (index + 1))
            for kernel, dilation in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    MiniMaxH3AudioAMPBlock(channels, kernel, dilation)
                )
        self.activation_post = MiniMaxH3AudioActivation1d(
            MiniMaxH3AudioSnakeBeta(channels)
        )
        self.conv_post = MiniMaxH3AudioWeightNormConv1d(
            channels, 1, 7, padding=3, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_pre(hidden_states)
        for index in range(self.num_upsamples):
            hidden_states = self.ups[index][0](hidden_states)
            residual = sum(
                (
                    self.resblocks[index * self.num_kernels + offset](hidden_states)
                    for offset in range(self.num_kernels)
                ),
                start=mx.zeros_like(hidden_states),
            )
            hidden_states = residual / self.num_kernels
        hidden_states = self.conv_post(self.activation_post(hidden_states))
        return mx.clip(hidden_states, -1.0, 1.0)


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(
        self, config: MiniMaxH3AudioVAEConfig | None = None, **config_kwargs
    ) -> None:
        super().__init__()
        if config is not None and config_kwargs:
            raise ValueError("pass a config or keyword fields, not both")
        self.config = config or MiniMaxH3AudioVAEConfig(**config_kwargs)
        config = self.config
        self.encoder = MiniMaxH3AudioEncoder(
            config.encoder_dim, config.encoder_rates, config.latent_dim
        )
        self.pre_block = MiniMaxH3AudioAttnProjection(
            config.latent_dim, config.latent_channels, config.num_attention_heads
        )
        self.mean_proj = MiniMaxH3AudioConv1d(
            config.latent_channels, config.latent_channels, 1
        )
        self.logs_proj = MiniMaxH3AudioConv1d(
            config.latent_channels, config.latent_channels, 1
        )
        self.dec_in_proj = MiniMaxH3AudioConv1d(
            config.latent_channels, config.latent_dim, 1
        )
        self.decoder = MiniMaxH3AudioBigVGANDecoder(config)

    def encode(self, sample: mx.array) -> MiniMaxH3AudioDiagonalGaussianDistribution:
        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError(
                f"sample must have shape (batch, 1, samples), got {sample.shape}"
            )
        right_pad = math.ceil(sample.shape[-1] / self.config.hop_length)
        right_pad = right_pad * self.config.hop_length - sample.shape[-1]
        if right_pad:
            sample = mx.pad(sample, ((0, 0), (0, 0), (0, right_pad)))
        hidden_states = self.encoder(sample.astype(self.encoder.block[0].weight.dtype))
        hidden_states = self.pre_block(hidden_states.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )
        mean = self.mean_proj(hidden_states).astype(mx.float32)
        logs = self.logs_proj(hidden_states).astype(mx.float32)
        return MiniMaxH3AudioDiagonalGaussianDistribution(mean, logs)

    def decode(self, latents: mx.array) -> MiniMaxH3AudioVAEOutput:
        if latents.ndim != 3:
            raise ValueError(
                f"latents must have shape (batch, channels, frames), got {latents.shape}"
            )
        decoded = self.decoder(self.dec_in_proj(latents))
        return MiniMaxH3AudioVAEOutput(decoded.astype(mx.float32))

    def __call__(self, sample: mx.array) -> MiniMaxH3AudioVAEOutput:
        return self.decode(self.encode(sample).mode())

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        converted: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith("weight_g"):
                continue
            if key.endswith("weight_v"):
                scale_key = f"{key[:-8]}weight_g"
                if scale_key not in weights:
                    raise ValueError(
                        f"missing weight-normalization tensor: {scale_key}"
                    )
                scale = weights[scale_key]
                norm = mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
                value = value * (scale / norm)
                key = f"{key[:-8]}weight"
            if value.ndim == 3 and key.endswith("weight"):
                value = (
                    value.transpose(1, 2, 0)
                    if key.startswith("decoder.ups.")
                    else value.transpose(0, 2, 1)
                )
            converted[key] = value
        return converted


__all__ = [
    "MiniMaxH3AudioDiagonalGaussianDistribution",
    "MiniMaxH3AudioVAE",
    "MiniMaxH3AudioVAEOutput",
    "kaiser_sinc_filter1d",
]
