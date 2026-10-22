"""Speech to prefix embeddings, as in the reference ``audio.py``.

A 16 kHz mono clip (int16 PCM or float samples, cut to 30 s, padded to 0.5 s) ->
NeMo log-mel features (128 bins every 10 ms, always float32) -> 17-layer FastConformer
with 8x subsampling -> adapter -> residual: one prefix embedding per 80 ms of audio.
The front end and the conformer are Nemotron-H Nano Omni's ``SoundFeatureExtractor``
and ``ParakeetEncoder`` (the same NeMo maths); ``sanitize_audio`` maps the checkpoint's
NeMo names and torch conv layout onto them.
"""

from io import BytesIO
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..nemotron_h_nano_omni.audio import ParakeetEncoder, SoundFeatureExtractor
from ..nemotron_h_nano_omni.config import AudioConfig as ParakeetConfig

SAMPLE_RATE, MIN_SAMPLES, MAX_SECONDS = 16000, 8000, 30

RENAMES = (
    ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
    ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
    ("self_attn.linear_q.", "self_attn.q_proj."),
    ("self_attn.linear_k.", "self_attn.k_proj."),
    ("self_attn.linear_v.", "self_attn.v_proj."),
    ("self_attn.linear_out.", "self_attn.o_proj."),
    ("self_attn.linear_pos.", "self_attn.relative_k_proj."),
    ("self_attn.pos_bias_u", "self_attn.bias_u"),
    ("self_attn.pos_bias_v", "self_attn.bias_v"),
    ("conv.batch_norm.", "conv.norm."),
)


def waveform(audio):
    """Mono 16 kHz int16 PCM or float samples, or an audio file (path, URL or binary
    file object, any rate, downmixed) -> float32 (N,), 0.5 s to 30 s long. Samples
    are padded with silence, as in the reference, even when there are none; a file
    that fails to decode or decodes to no samples raises ValueError."""
    if isinstance(audio, (str, Path)) or hasattr(audio, "read"):
        from ...utils import load_audio

        try:
            if hasattr(audio, "read") and not isinstance(audio, BytesIO):
                audio = BytesIO(audio.read())  # mlx-audio reads paths and BytesIO
            audio = load_audio(audio, sr=SAMPLE_RATE)
        except Exception as error:
            raise ValueError(f"Failed to load audio: {error}") from error
        if audio.size == 0:
            raise ValueError("Failed to load audio: the file has no samples")
    if isinstance(audio, mx.array) and audio.dtype != mx.int16:
        audio = audio.astype(mx.float32)
    x = np.asarray(audio)
    if x.ndim != 1:
        raise ValueError("audio must be mono: a 1-D array of 16 kHz samples")
    x = x[: MAX_SECONDS * SAMPLE_RATE]
    scale = 32768.0 if x.dtype == np.int16 else 1.0
    x = x.astype(np.float32) / np.float32(scale)
    return mx.array(np.pad(x, (0, max(0, MIN_SAMPLES - len(x)))))


def sanitize_audio(weights):
    """Checkpoint ``audio.*`` keys -> ``Audio`` names, MLX layout; idempotent."""
    out = {}
    for key, value in weights.items():
        if key.startswith("audio."):
            if key.endswith("num_batches_tracked"):
                continue
            for old, new in RENAMES:
                key = key.replace(old, new)
            # torch (out, in/groups, *kernel) -> MLX (out, *kernel, in/groups); kernels
            # are square, 1-D convs pointwise (kernel 1) or depthwise (in/groups 1)
            if value.ndim == 4 and value.shape[2] == value.shape[3]:
                value = value.transpose(0, 2, 3, 1)
            elif value.ndim == 3 and value.shape[-1 if "pointwise" in key else 1] == 1:
                value = value.transpose(0, 2, 1)
        out[key] = value
    return out


class Adapter(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear_1 = nn.Linear(d_in, d_out)
        self.linear_2 = nn.Linear(d_out, d_out)

    def __call__(self, x):
        return self.linear_2(nn.gelu(self.linear_1(self.norm(x))))


class Residual(nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, width)
        self.up = nn.Linear(width, d)

    def __call__(self, x):
        return x + self.up(nn.gelu(self.down(self.ln(x))))


class Audio(nn.Module):
    def __init__(self, config, out):
        super().__init__()
        encoder = ParakeetConfig(
            hidden_size=config.d_model,
            num_attention_heads=config.n_heads,
            num_hidden_layers=config.n_layers,
            intermediate_size=config.d_model * config.ff_expansion_factor,
            attention_bias=True,
            convolution_bias=True,
            conv_kernel_size=config.conv_kernel_size,
            subsampling_conv_channels=config.subsampling_conv_channels,
            num_mel_bins=config.feat_in,
        )
        self.frontend = SoundFeatureExtractor(encoder)
        self.encoder = ParakeetEncoder(encoder)
        self.adapter = Adapter(config.d_model, out)
        self.residual = Residual(out, config.residual_width)

    def __call__(self, audio):
        """One clip -> (1, P, D) prefix embeddings, P = the encoder's valid frames."""
        mel, mask, _ = self.frontend([waveform(audio)])
        x, valid = self.encoder(mel.astype(self.adapter.norm.weight.dtype), mask)
        return self.residual(self.adapter(x[:, : int(valid.sum())]))
