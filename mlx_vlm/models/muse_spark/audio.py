"""Causal waveform pyramid used by Muse Spark's audio encoder."""

import mlx.core as mx
from mlx import nn

from .config import AudioConfig


class AudioConvCache:
    """Left context for streaming chunks aligned to samples_per_token."""

    def __init__(self):
        self.states = {}

    def pad(self, x, key, length):
        if length == 0:
            return x
        previous = self.states.get(key)
        if previous is None:
            previous = mx.zeros((x.shape[0], length, x.shape[-1]), dtype=x.dtype)
        if previous.shape != (x.shape[0], length, x.shape[-1]):
            raise ValueError("Audio cache batch size changed; use a new cache.")
        padded = mx.concatenate([previous, x], axis=1)
        self.states[key] = padded[:, -length:]
        return padded


class AudioPatchEmbedding(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        channels = 1 if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        width = config.hidden_sizes[layer_idx]
        self.patch_size = config.patch_sizes[layer_idx]
        self.proj = nn.Linear(channels * self.patch_size, width)
        self.norm = nn.RMSNorm(width, eps=config.rms_norm_eps)

    def __call__(self, x):
        batch, length, channels = x.shape
        x = x.reshape(batch, length // self.patch_size, self.patch_size * channels)
        return self.norm(self.proj(x))


class AudioMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        width = config.hidden_sizes[layer_idx]
        intermediate = int(width * config.mlp_ratio)
        self.fc1 = nn.Linear(width, intermediate)
        self.fc2 = nn.Linear(intermediate, width)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class AudioConvBlock(nn.Module):
    def __init__(self, config, layer_idx, block_idx):
        super().__init__()
        width = config.hidden_sizes[layer_idx]
        self.conv = nn.Conv1d(width, width, config.conv_kernel_size, groups=width)
        self.norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.mlp = AudioMLP(config, layer_idx)
        self._left_pad = config.conv_kernel_size - 1
        self._cache_key = (layer_idx, block_idx)

    def __call__(self, x, padding_cache=None):
        residual = x
        if padding_cache is None:
            x = mx.pad(x, [(0, 0), (self._left_pad, 0), (0, 0)])
        else:
            x = padding_cache.pad(x, self._cache_key, self._left_pad)
        return residual + self.mlp(self.norm(self.conv(x)))


class AudioLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.patch_embedder = AudioPatchEmbedding(config, layer_idx)
        self.conv_blocks = [
            AudioConvBlock(config, layer_idx, idx)
            for idx in range(config.depths[layer_idx])
        ]

    def __call__(self, x, padding_cache=None):
        x = self.patch_embedder(x)
        for block in self.conv_blocks:
            x = block(x, padding_cache)
        return x


class AudioModel(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.layers = [
            AudioLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_sizes[-1], eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_sizes[-1], config.out_hidden_size)
        self.post_proj_norm = nn.RMSNorm(
            config.out_hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, input_values, padding_cache=None):
        if input_values.ndim != 2 or input_values.shape[1] == 0:
            raise ValueError(
                "input_values must be a nonempty (batch, samples) waveform."
            )
        if input_values.shape[1] % self.config.samples_per_token:
            raise ValueError(
                f"Audio length must be a multiple of {self.config.samples_per_token}."
            )
        dtype = self.layers[0].patch_embedder.proj.weight.dtype
        x = input_values.astype(dtype)[..., None]
        for layer in self.layers:
            x = layer(x, padding_cache)
        return self.post_proj_norm(self.proj(self.norm(x)))
