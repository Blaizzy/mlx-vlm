from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(slots=True)
class Flux2KVLayerCache:
    k_ref: mx.array | None = None
    v_ref: mx.array | None = None

    def store(self, k_ref: mx.array, v_ref: mx.array) -> None:
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[mx.array, mx.array]:
        if self.k_ref is None or self.v_ref is None:
            raise RuntimeError("Flux2 KV cache has not been populated")
        return self.k_ref, self.v_ref

    def clear(self) -> None:
        self.k_ref = None
        self.v_ref = None


class Flux2KVCache:
    def __init__(self, *, num_double_layers: int, num_single_layers: int):
        self.double_block_caches = [
            Flux2KVLayerCache() for _ in range(num_double_layers)
        ]
        self.single_block_caches = [
            Flux2KVLayerCache() for _ in range(num_single_layers)
        ]
        self.num_ref_tokens = 0

    def get_double(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.double_block_caches[layer_idx]

    def get_single(self, layer_idx: int) -> Flux2KVLayerCache:
        return self.single_block_caches[layer_idx]

    def arrays(self) -> list[mx.array]:
        arrays = []
        for layer_cache in (*self.double_block_caches, *self.single_block_caches):
            if layer_cache.k_ref is not None:
                arrays.append(layer_cache.k_ref)
            if layer_cache.v_ref is not None:
                arrays.append(layer_cache.v_ref)
        return arrays

    def clear(self) -> None:
        for layer_cache in (*self.double_block_caches, *self.single_block_caches):
            layer_cache.clear()
        self.num_ref_tokens = 0
