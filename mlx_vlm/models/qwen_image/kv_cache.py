"""Per-edit cache for the fixed text and reference-image prefix."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(slots=True)
class QwenImageKVLayerCache:
    key: mx.array | None = None
    value: mx.array | None = None

    def store(self, key: mx.array, value: mx.array, prefix_len: int) -> None:
        # Compact the prefix instead of retaining strided views of the full K/V.
        self.key = mx.contiguous(key[:, :, :prefix_len])
        self.value = mx.contiguous(value[:, :, :prefix_len])

    def get(self) -> tuple[mx.array, mx.array]:
        if self.key is None or self.value is None:
            raise ValueError("Qwen-Image KV cache has not been populated")
        return self.key, self.value


class QwenImageKVCache:
    """One conditioning branch of one edit; never share across prompts or models."""

    def __init__(self, num_layers: int) -> None:
        self.layers = [QwenImageKVLayerCache() for _ in range(num_layers)]
        self.prefix_len = 0
        self.target_shape = None
        self.input_shape = None
        self.dtype = None
        self.cos = self.sin = self.mask = None

    def arrays(self) -> list[mx.array]:
        arrays = [value for layer in self.layers for value in (layer.key, layer.value)]
        arrays.extend([self.cos, self.sin, self.mask])
        return [value for value in arrays if value is not None]

    def clear(self) -> None:
        for layer in self.layers:
            layer.key = layer.value = None
        self.prefix_len = 0
        self.target_shape = self.input_shape = self.dtype = None
        self.cos = self.sin = self.mask = None
