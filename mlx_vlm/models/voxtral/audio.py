import json
import math
from pathlib import Path
from typing import Iterable, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

from .config import AudioConfig


def _conv1d_out_length(length: int, kernel: int, stride: int, padding: int) -> int:
    return (length + 2 * padding - kernel) // stride + 1


def _expected_mel_frames(config: AudioConfig) -> int:
    conv1_stride = 1
    conv2_stride = 2
    return config.max_source_positions * conv1_stride * conv2_stride


def _audio_tokens_per_chunk(config: AudioConfig) -> int:
    group_size = config.intermediate_size // config.hidden_size
    if group_size <= 0 or config.intermediate_size % config.hidden_size != 0:
        raise ValueError("audio_config.intermediate_size must be a multiple of hidden_size")
    input_len = _expected_mel_frames(config)
    conv2_out = _conv1d_out_length(
        _conv1d_out_length(input_len, 3, 1, 1),
        3,
        2,
        1,
    )
    if conv2_out % group_size != 0:
        raise ValueError(
            "audio_config.max_source_positions must yield a conv2 length divisible by group_size"
        )
    return conv2_out // group_size


class VoxtralFeatureExtractor:
    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        dither: float = 0.0,
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_fft = n_fft
        self.padding_value = padding_value
        self.dither = dither
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    @classmethod
    def from_pretrained(cls, model_path: Path) -> "VoxtralFeatureExtractor":
        config_path = Path(model_path) / "preprocessor_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"preprocessor_config.json not found at {config_path}")
        config = json.loads(config_path.read_text())
        return cls(**config)

    def _log_mel(self, waveform: np.ndarray) -> np.ndarray:
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            dither=self.dither,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.astype(np.float32)

    def _pad_to_multiple(self, audio: np.ndarray, multiple: int) -> np.ndarray:
        if multiple <= 0:
            return audio
        pad_len = (-len(audio)) % multiple
        if pad_len == 0:
            return audio
        return np.pad(audio, (0, pad_len), constant_values=self.padding_value)

    def _chunk_audio(self, audio: np.ndarray, chunk_samples: int) -> List[np.ndarray]:
        if len(audio) == 0:
            return [np.zeros((chunk_samples,), dtype=np.float32)]
        chunks = []
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start : start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(
                    chunk, (0, chunk_samples - len(chunk)), constant_values=self.padding_value
                )
            chunks.append(chunk.astype(np.float32))
        return chunks

    def __call__(
        self,
        raw_speech: Iterable[np.ndarray],
        sampling_rate: Optional[int] = None,
        padding: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        max_source_positions: Optional[int] = None,
        **kwargs,
    ):
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate {self.sampling_rate}, got {sampling_rate}"
            )

        pad_to_multiple_of = (
            self.n_samples if pad_to_multiple_of is None else pad_to_multiple_of
        )
        max_source_positions = (
            self.nb_max_frames
            if max_source_positions is None
            else max_source_positions
        )

        features = []
        for audio in raw_speech:
            audio = np.asarray(audio, dtype=np.float32)
            if padding:
                audio = self._pad_to_multiple(audio, pad_to_multiple_of)
            chunks = self._chunk_audio(audio, self.n_samples)
            for chunk in chunks:
                mel = self._log_mel(chunk)
                if mel.shape[1] != max_source_positions:
                    if mel.shape[1] > max_source_positions:
                        mel = mel[:, :max_source_positions]
                    else:
                        pad_len = max_source_positions - mel.shape[1]
                        mel = np.pad(mel, ((0, 0), (0, pad_len)), constant_values=0.0)
                features.append(mel)
        return {"input_features": np.stack(features, axis=0)}


class VoxtralAttention(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states) * self.scaling
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn_scores = q @ k.swapaxes(-2, -1)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = mx.softmax(attn_scores, axis=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
        return self.out_proj(attn_output)


class VoxtralEncoderLayer(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn = VoxtralAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        if config.activation_function == "gelu":
            self.activation_fn = nn.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.activation_function}")

    def __call__(self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, config.hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions, config.hidden_size)
        self.layers = [VoxtralEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, input_features: mx.array, attention_mask: Optional[mx.array] = None):
        expected_len = _expected_mel_frames(self.config)
        if input_features.shape[-1] != expected_len:
            raise ValueError(
                f"Voxtral expects mel length {expected_len}, got {input_features.shape[-1]}"
            )

        # MLX conv1d expects (B, L, C) while Whisper features are (B, C, L).
        input_features = input_features.astype(self.conv1.weight.dtype).transpose(0, 2, 1)
        hidden_states = nn.gelu(self.conv1(input_features))
        hidden_states = nn.gelu(self.conv2(hidden_states))

        pos_embeds = self.embed_positions.weight[: hidden_states.shape[1]]
        hidden_states = (hidden_states + pos_embeds) * self.embed_scale

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    @staticmethod
    def sanitize(weights):
        def _maybe_transpose(weight):
            if weight.ndim != 3:
                return weight
            # HF conv1d weights: (out, in, k); MLX expects (out, k, in)
            return weight.transpose(0, 2, 1)

        remapped = {}
        for key, value in weights.items():
            if key.endswith("audio_tower.conv1.weight") or key.endswith(
                "audio_tower.conv2.weight"
            ):
                remapped[key] = _maybe_transpose(value)
            else:
                remapped[key] = value
        return remapped


__all__ = [
    "AudioModel",
    "VoxtralFeatureExtractor",
    "_audio_tokens_per_chunk",
]
