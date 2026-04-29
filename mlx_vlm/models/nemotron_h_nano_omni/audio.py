import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import SoundConfig


class SquaredReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x) ** 2


class SoundProjection(nn.Module):
    def __init__(self, config: SoundConfig, llm_hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.projection_hidden_size,
            bias=config.projection_bias,
        )
        self.activation = SquaredReLU()
        self.linear2 = nn.Linear(
            config.projection_hidden_size,
            llm_hidden_size,
            bias=config.projection_bias,
        )

    def __call__(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.linear2(hidden_states)


class ParakeetEncoderRelPositionalEncoding(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings

    def __call__(self, hidden_states):
        seq_length = hidden_states.shape[1]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds "
                f"max_position_embeddings={self.max_position_embeddings}."
            )

        positions = mx.arange(seq_length - 1, -seq_length, -1, dtype=mx.float32)
        inv_freq = 1.0 / (
            10000.0
            ** (mx.arange(0, self.hidden_size, 2, dtype=mx.float32) / self.hidden_size)
        )
        freqs = positions[:, None] * inv_freq[None, :]
        pos_embed = mx.stack([mx.sin(freqs), mx.cos(freqs)], axis=-1).reshape(
            2 * seq_length - 1, self.hidden_size
        )
        pos_embed = mx.broadcast_to(
            pos_embed[None, :, :],
            (hidden_states.shape[0], pos_embed.shape[0], pos_embed.shape[1]),
        )
        return pos_embed.astype(hidden_states.dtype)


class ParakeetEncoderFeedForward(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(self, hidden_states):
        return self.linear2(self.activation(self.linear1(hidden_states)))


class ParakeetEncoderConvolutionModule(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size
        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.convolution_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.norm = nn.BatchNorm(channels)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.convolution_bias,
        )

    def __call__(self, hidden_states, attention_mask=None):
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.glu(hidden_states, axis=-1)

        if attention_mask is not None:
            all_masked_rows = mx.all(mx.logical_not(attention_mask), axis=2)
            all_masked_rows = mx.squeeze(all_masked_rows, axis=1)
            hidden_states = mx.where(all_masked_rows[..., None], 0.0, hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.pointwise_conv2(hidden_states)


class ParakeetEncoderAttention(nn.Module):
    def __init__(self, config: SoundConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.relative_k_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.bias_u = mx.zeros((config.num_attention_heads, self.head_dim))
        self.bias_v = mx.zeros((config.num_attention_heads, self.head_dim))

    def _rel_shift(self, attention_scores):
        batch_size, num_heads, query_length, position_length = attention_scores.shape
        attention_scores = mx.pad(attention_scores, [(0, 0), (0, 0), (0, 0), (1, 0)])
        attention_scores = attention_scores.reshape(
            batch_size, num_heads, position_length + 1, query_length
        )
        attention_scores = attention_scores[:, :, 1:, :]
        return attention_scores.reshape(
            batch_size, num_heads, query_length, position_length
        )

    def __call__(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (
            batch_size,
            seq_length,
            self.config.num_attention_heads,
            self.head_dim,
        )

        query_states = (
            self.q_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        )
        key_states = (
            self.k_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        )
        value_states = (
            self.v_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        )

        query_states_with_bias_u = query_states + self.bias_u[None, :, None, :]
        query_states_with_bias_v = query_states + self.bias_v[None, :, None, :]

        relative_key_states = self.relative_k_proj(position_embeddings).reshape(
            batch_size, -1, self.config.num_attention_heads, self.head_dim
        )

        matrix_bd = mx.matmul(
            query_states_with_bias_v,
            relative_key_states.transpose(0, 2, 3, 1),
        )
        matrix_bd = self._rel_shift(matrix_bd)[..., :seq_length] * self.scaling

        if attention_mask is not None:
            matrix_bd = mx.where(
                attention_mask,
                matrix_bd,
                mx.array(mx.finfo(matrix_bd.dtype).min, dtype=matrix_bd.dtype),
            )

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states_with_bias_u,
            key_states,
            value_states,
            scale=self.scaling,
            mask=matrix_bd,
        )
        if attention_mask is not None:
            valid_queries = mx.any(attention_mask, axis=-1)
            attn_output = attn_output * valid_queries[..., None].astype(
                attn_output.dtype
            )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, -1
        )
        return self.o_proj(attn_output)


class ParakeetEncoderSubsamplingConv2D(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.padding = (self.kernel_size - 1) // 2
        self.num_layers = int(math.log2(config.subsampling_factor))

        self.layers = [
            nn.Conv2d(
                1,
                self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.ReLU(),
        ]
        for _ in range(self.num_layers - 1):
            self.layers.extend(
                [
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        groups=self.channels,
                    ),
                    nn.Conv2d(self.channels, self.channels, kernel_size=1),
                    nn.ReLU(),
                ]
            )

        out_length = config.num_mel_bins // (self.stride**self.num_layers)
        self.linear = nn.Linear(
            config.subsampling_conv_channels * out_length,
            config.hidden_size,
            bias=True,
        )

    def _get_output_length(self, input_lengths, conv_layer):
        if getattr(conv_layer, "stride", (1, 1)) == (1, 1):
            return input_lengths

        padding = conv_layer.padding[0]
        kernel_size = conv_layer.weight.shape[1]
        stride = conv_layer.stride[0]
        lengths = mx.floor(
            (input_lengths.astype(mx.float32) + 2 * padding - kernel_size) / stride
            + 1.0
        )
        return lengths.astype(mx.int32)

    def __call__(self, input_features, attention_mask=None):
        hidden_states = input_features[..., None]
        current_lengths = (
            mx.sum(attention_mask, axis=-1).astype(mx.int32)
            if attention_mask is not None
            else None
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if isinstance(layer, nn.Conv2d) and attention_mask is not None:
                current_lengths = self._get_output_length(current_lengths, layer)
                current_seq_length = hidden_states.shape[1]
                channel_mask = (
                    mx.arange(current_seq_length)[None, :] < current_lengths[:, None]
                )
                hidden_states = hidden_states * channel_mask[:, :, None, None].astype(
                    hidden_states.dtype
                )

        hidden_states = hidden_states.transpose(0, 1, 3, 2).reshape(
            hidden_states.shape[0], hidden_states.shape[1], -1
        )
        return self.linear(hidden_states)


class ParakeetEncoderBlock(nn.Module):
    def __init__(self, config: SoundConfig, layer_idx: int):
        super().__init__()
        self.feed_forward1 = ParakeetEncoderFeedForward(config)
        self.self_attn = ParakeetEncoderAttention(config, layer_idx)
        self.conv = ParakeetEncoderConvolutionModule(config)
        self.feed_forward2 = ParakeetEncoderFeedForward(config)
        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states

        normalized_hidden_states = self.norm_self_att(hidden_states)
        hidden_states = hidden_states + self.self_attn(
            normalized_hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )

        hidden_states = hidden_states + self.conv(
            self.norm_conv(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden_states)
        )
        return self.norm_out(hidden_states)


class ParakeetEncoder(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        self.config = config
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0
        self.subsampling = ParakeetEncoderSubsamplingConv2D(config)
        self.encode_positions = ParakeetEncoderRelPositionalEncoding(config)
        self.layers = [
            ParakeetEncoderBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]

    def _get_subsampling_output_length(self, input_lengths):
        kernel_size = self.config.subsampling_conv_kernel_size
        stride = self.config.subsampling_conv_stride
        num_layers = int(math.log2(self.config.subsampling_factor))
        add_pad = ((kernel_size - 1) // 2) * 2 - kernel_size
        lengths = input_lengths
        for _ in range(num_layers):
            lengths = mx.floor((lengths.astype(mx.float32) + add_pad) / stride + 1.0)
        return lengths.astype(mx.int32)

    def _get_output_attention_mask(self, attention_mask, target_length=None):
        output_lengths = self._get_subsampling_output_length(
            mx.sum(attention_mask, axis=-1)
        )
        max_length = (
            target_length if target_length is not None else mx.max(output_lengths)
        )
        max_length = (
            int(max_length.item()) if isinstance(max_length, mx.array) else max_length
        )
        return mx.arange(max_length)[None, :] < output_lengths[:, None]

    def __call__(self, input_features, attention_mask=None):
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        output_mask = None
        if attention_mask is not None:
            output_mask = self._get_output_attention_mask(
                attention_mask, target_length=hidden_states.shape[1]
            )
            attention_mask = (
                output_mask[:, None, :, None] & output_mask[:, None, None, :]
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        return hidden_states, output_mask


class SoundEncoder(nn.Module):
    def __init__(self, config: SoundConfig):
        super().__init__()
        self.config = config
        self.encoder = ParakeetEncoder(config)

    def __call__(self, input_features, attention_mask=None):
        hidden_states, _ = self.encoder(input_features, attention_mask)
        return hidden_states

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size


class SoundFeatureExtractor:
    def __init__(self, config: SoundConfig):
        self.config = config
        self.sampling_rate = config.sampling_rate

        try:
            from mlx_audio.dsp import hanning, mel_filters, stft
        except ImportError as exc:
            raise ImportError(
                "Nemotron Omni audio preprocessing requires mlx-audio. "
                "Install mlx-audio or make it importable on PYTHONPATH."
            ) from exc

        self._hanning = hanning
        self._mel_filters = mel_filters
        self._stft = stft

    def _to_waveform(self, audio):
        if isinstance(audio, mx.array):
            waveform = audio
        elif isinstance(audio, np.ndarray):
            waveform = mx.array(audio)
        elif isinstance(audio, (list, tuple)):
            waveform = mx.array(np.asarray(audio, dtype=np.float32))
        elif isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            waveform = load_audio(audio, self.sampling_rate, dtype=mx.float32)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")

        if waveform.ndim > 1:
            waveform = mx.mean(waveform, axis=-1)
        return waveform.astype(mx.float32)

    def _log_mel_spectrogram(self, waveform):
        original_dtype = waveform.dtype
        if self.config.preemphasis is not None:
            waveform = mx.concatenate(
                [
                    waveform[:1],
                    waveform[1:] - self.config.preemphasis * waveform[:-1],
                ],
                axis=0,
            )

        window = self._hanning(self.config.win_length, periodic=False)
        if window.shape[0] < self.config.n_fft:
            left = (self.config.n_fft - window.shape[0]) // 2
            right = self.config.n_fft - window.shape[0] - left
            window = mx.concatenate(
                [
                    mx.zeros((left,), dtype=window.dtype),
                    window,
                    mx.zeros((right,), dtype=window.dtype),
                ]
            )

        spec = self._stft(
            waveform,
            self.config.n_fft,
            self.config.hop_length,
            self.config.n_fft,
            window,
            pad_mode="constant",
        )
        spec = mx.square(mx.abs(spec)).astype(original_dtype)
        filters = self._mel_filters(
            self.config.sampling_rate,
            self.config.n_fft,
            self.config.num_mel_bins,
            norm="slaney",
            mel_scale="slaney",
        )
        mel = filters.astype(spec.dtype) @ spec.T
        mel = mx.log(mel + mx.array(2**-24, dtype=mel.dtype)).T
        return mel

    def __call__(self, audio: Sequence):
        if not isinstance(audio, (list, tuple)):
            audio = [audio]

        features = []
        full_lengths = []
        valid_lengths = []
        for clip in audio:
            waveform = self._to_waveform(clip)
            mel = self._log_mel_spectrogram(waveform)
            valid_length = waveform.shape[0] // self.config.hop_length
            valid_length = min(valid_length, mel.shape[0])
            mask = mx.arange(mel.shape[0]) < valid_length
            mask_f = mask[:, None].astype(mel.dtype)
            denom = max(valid_length, 1)
            mean = mx.sum(mel * mask_f, axis=0) / denom
            variance_denom = max(valid_length - 1, 1)
            variance = mx.sum(((mel - mean) ** 2) * mask_f, axis=0) / variance_denom
            mel = ((mel - mean) / (mx.sqrt(variance) + 1e-5)) * mask_f
            features.append(mel)
            full_lengths.append(mel.shape[0])
            valid_lengths.append(valid_length)

        max_length = max(full_lengths)
        padded = []
        masks = []
        for mel, full_length, valid_length in zip(
            features, full_lengths, valid_lengths
        ):
            pad = max_length - full_length
            if pad:
                mel = mx.pad(mel, [(0, pad), (0, 0)])
            padded.append(mel)
            masks.append(mx.arange(max_length) < valid_length)

        return (
            mx.stack(padded, axis=0),
            mx.stack(masks, axis=0).astype(mx.int32),
            mx.array(full_lengths, dtype=mx.int32),
        )


def sanitize_audio_weights(weights):
    sanitized = {}
    for key, value in weights.items():
        if key.startswith("sound_encoder.encoder.feature_extractor."):
            continue
        if key.endswith(".num_batches_tracked"):
            continue
        if key.startswith("sound_encoder.encoder."):
            if key.endswith(".weight") and value.ndim == 3:
                value = value.transpose(0, 2, 1)
            elif key.endswith(".weight") and value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
        sanitized[key] = value
    return sanitized
