from __future__ import annotations

import math
from bisect import bisect_right
from types import SimpleNamespace
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import ArraysCache, BatchKVCache, KVCache, QuantizedKVCache, dynamic_roll
from ..qwen3_5.language import LanguageModel as Qwen3_5LanguageModel
from ..qwen3_5.language import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    _create_qwen3_5_attention_mask,
    _create_qwen3_5_ssm_mask,
    _extract_row_cache,
    _pad_row_time,
    _qwen3_5_left_padding_info,
    _restore_batch_padding_metadata,
)
from ..qwen3_5.speculative_verifier import Qwen3_5ExactSpeculativeVerifier
from ..qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock
from .config import ModelConfig, TextConfig
from .qsa_kernel import qsa_sparse_attention


def _append_indexer_positions(
    cached: Optional[mx.array], position_ids: mx.array
) -> mx.array:
    if cached is None:
        return position_ids
    if cached.ndim == 3 and position_ids.ndim == 2:
        position_ids = mx.broadcast_to(
            position_ids[None],
            (cached.shape[0], *position_ids.shape),
        )
    elif cached.ndim == 2 and position_ids.ndim == 3:
        cached = mx.broadcast_to(cached[None], (position_ids.shape[0], *cached.shape))
    elif cached.ndim != position_ids.ndim:
        raise ValueError(
            "QSA position IDs must be 2-D text positions or 3-D MRoPE positions, "
            f"got cached={cached.shape} and current={position_ids.shape}."
        )
    return mx.concatenate([cached, position_ids], axis=-1)


class QSAKVCache(KVCache):
    """KV cache with the raw indexer keys and multimodal positions used by QSA."""

    # Hybrid/TurboQuant caches do not currently expose a way to carry the
    # indexer's unprojected keys. Uniform quantization uses the specialized
    # QSAQuantizedKVCache below; other schemes leave this cache in float.
    preserve_auxiliary_kv_state = True

    def __init__(self):
        super().__init__()
        self.index_keys = None
        self.index_position_ids = None
        self.index_block_keys = None
        self.index_block_ratio = None

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids, position_ids
            )
        return self.index_keys, self.index_position_ids

    @property
    def state(self):
        if self.keys is None:
            return (
                None,
                None,
                self.index_keys,
                self.index_position_ids,
                self.index_block_keys,
                self.index_block_ratio,
            )
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
            self.index_keys,
            self.index_position_ids,
            self.index_block_keys,
            self.index_block_ratio,
        )

    @state.setter
    def state(self, value):
        if len(value) == 4:
            self.keys, self.values, self.index_keys, self.index_position_ids = value
            self.index_block_keys = None
            self.index_block_ratio = None
        else:
            (
                self.keys,
                self.values,
                self.index_keys,
                self.index_position_ids,
                self.index_block_keys,
                self.index_block_ratio,
            ) = value
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    def clear_index_blocks(self):
        self.index_block_keys = None
        self.index_block_ratio = None

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, : self.offset]
            self.index_position_ids = self.index_position_ids[..., : self.offset]
        self.clear_index_blocks()
        return n

    def extract(self, idx):
        cache = QSAKVCache()
        if self.keys is not None:
            cache.keys = mx.contiguous(self.keys[idx : idx + 1])
            cache.values = mx.contiguous(self.values[idx : idx + 1])
            cache.offset = self.offset
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
        if self.index_block_keys is not None:
            cache.index_block_keys = mx.contiguous(self.index_block_keys[idx : idx + 1])
            cache.index_block_ratio = self.index_block_ratio
        return cache

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if self.index_position_ids.ndim == 3:
                self.index_position_ids = self.index_position_ids[:, batch_indices]
            else:
                self.index_position_ids = self.index_position_ids[batch_indices]
        if self.index_block_keys is not None:
            self.index_block_keys = self.index_block_keys[batch_indices]

    def to_batch(self, left_padding):
        """Convert a singleton QSA cache, including its indexer state."""

        batch = BatchQSAKVCache(left_padding)
        padding = mx.array(left_padding)
        if self.empty() and self.index_keys is None:
            return batch
        if padding.size != 1:
            raise ValueError(
                "A warm QSA cache can only seed one batch row, got "
                f"left_padding={padding.tolist()}"
            )
        pad = int(padding.item())
        if not self.empty():
            keys, values = self.state[:2]
            if pad:
                keys = mx.pad(keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
                values = mx.pad(values, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            batch.kv_cache.state = (
                keys,
                values,
                mx.array([self.offset], dtype=mx.int32),
                padding.astype(mx.int32),
            )
        if self.index_keys is not None:
            index_keys = self.index_keys[:, : self.offset]
            positions = self.index_position_ids[..., : self.offset]
            if pad:
                index_keys = mx.pad(index_keys, [(0, 0), (pad, 0), (0, 0)])
                positions = mx.pad(
                    positions,
                    (
                        [(0, 0), (0, 0), (pad, 0)]
                        if positions.ndim == 3
                        else [(0, 0), (pad, 0)]
                    ),
                )
            batch.index_keys = index_keys
            batch.index_position_ids = positions
            batch.index_offset = index_keys.shape[1]
        if pad == 0 and self.index_block_keys is not None:
            batch.index_block_keys = self.index_block_keys
            batch.index_block_ratio = self.index_block_ratio
        return batch

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return BatchQSAKVCache.merge(caches, prefix_lens=prefix_lens)

    def prefix_cache_merge(self, rows, prefix_lens):
        return self.merge(rows, prefix_lens=prefix_lens)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        base = super().to_quantized(group_size=group_size, bits=bits)
        cache = QSAQuantizedKVCache(group_size=group_size, bits=bits)
        cache.keys = base.keys
        cache.values = base.values
        cache.offset = base.offset
        cache.index_keys = self.index_keys
        cache.index_position_ids = self.index_position_ids
        cache.index_block_keys = self.index_block_keys
        cache.index_block_ratio = self.index_block_ratio
        return cache

    @property
    def nbytes(self):
        size = super().nbytes
        if self.index_keys is not None:
            size += self.index_keys.nbytes + self.index_position_ids.nbytes
        if self.index_block_keys is not None:
            size += self.index_block_keys.nbytes
        return size


class BatchQSAKVCache:
    """Batch KV cache that keeps QSA keys and text/MRoPE positions aligned."""

    step = BatchKVCache.step

    def __init__(self, left_padding):
        self.kv_cache = BatchKVCache(left_padding)
        self.index_keys = None
        self.index_position_ids = None
        self.index_offset = 0
        self.index_block_keys = None
        self.index_block_ratio = None

    def clear_index_blocks(self):
        self.index_block_keys = None
        self.index_block_ratio = None

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = value

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    @left_padding.setter
    def left_padding(self, value):
        self.kv_cache.left_padding = value

    @property
    def _idx(self):
        return self.kv_cache._idx

    def update_and_fetch(self, keys, values):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate(
                [self.index_keys[:, : self.index_offset], keys], axis=1
            )
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids[..., : self.index_offset], position_ids
            )
        self.index_offset = self.index_keys.shape[1]
        return self.index_keys, self.index_position_ids

    def prepare(self, **kwargs):
        right_padding = kwargs.get("right_padding")
        if right_padding is not None and any(int(x) for x in right_padding):
            self.clear_index_blocks()
        self.kv_cache.prepare(**kwargs)

    def finalize(self):
        right_padding = getattr(self.kv_cache, "_right_padding", None)
        self.kv_cache.finalize()
        if right_padding is None or self.index_keys is None:
            return
        self.clear_index_blocks()
        self.index_keys = dynamic_roll(self.index_keys, right_padding, axis=1)
        if self.index_position_ids.ndim == 3:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding[None], axis=2
            )
        else:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding, axis=1
            )

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        min_left = int(self.left_padding[batch_indices].min().item())
        self.kv_cache.filter(batch_indices)
        self.clear_index_blocks()
        if self.index_keys is None:
            return
        self.index_keys = self.index_keys[batch_indices]
        if self.index_position_ids.ndim == 3:
            self.index_position_ids = self.index_position_ids[:, batch_indices]
        else:
            self.index_position_ids = self.index_position_ids[batch_indices]
        if min_left > 0:
            self.index_keys = self.index_keys[:, min_left:]
            self.index_position_ids = self.index_position_ids[..., min_left:]
            self.index_offset -= min_left

    @staticmethod
    def _promote_positions(positions, sample_positions):
        if positions.ndim == sample_positions.ndim:
            return positions
        if positions.ndim == 2 and sample_positions.ndim == 3:
            return mx.broadcast_to(
                positions[None], (sample_positions.shape[0], *positions.shape)
            )
        raise ValueError(
            "QSA batch position IDs must be 2-D text positions or compatible "
            f"3-D MRoPE positions, got {positions.shape} and "
            f"{sample_positions.shape}."
        )

    @classmethod
    def _pad_index(cls, cache, target, sample_keys, sample_positions):
        length = 0 if cache.index_keys is None else cache.index_offset
        left = target - length
        if cache.index_keys is None:
            keys = mx.zeros(
                (cache.offset.shape[0], 0, sample_keys.shape[-1]),
                dtype=sample_keys.dtype,
            )
            if sample_positions.ndim == 3:
                positions = mx.zeros(
                    (sample_positions.shape[0], cache.offset.shape[0], 0),
                    dtype=sample_positions.dtype,
                )
            else:
                positions = mx.zeros(
                    (cache.offset.shape[0], 0), dtype=sample_positions.dtype
                )
        else:
            keys = cache.index_keys[:, :length]
            positions = cache.index_position_ids[..., :length]
            positions = cls._promote_positions(positions, sample_positions)
        if left:
            keys = mx.pad(keys, [(0, 0), (left, 0), (0, 0)])
            positions = mx.pad(
                positions,
                (
                    [(0, 0), (0, 0), (left, 0)]
                    if positions.ndim == 3
                    else [(0, 0), (left, 0)]
                ),
            )
        return keys, positions

    @staticmethod
    def _samples(*caches):
        sample_keys = next(
            (cache.index_keys for cache in caches if cache.index_keys is not None),
            None,
        )
        positions = [
            cache.index_position_ids
            for cache in caches
            if cache.index_position_ids is not None
        ]
        sample_positions = max(positions, key=lambda value: value.ndim, default=None)
        return sample_keys, sample_positions

    def extend(self, other):
        if not isinstance(other, BatchQSAKVCache):
            raise TypeError(f"Cannot extend BatchQSAKVCache with {type(other)}")
        sample_keys, sample_positions = self._samples(self, other)
        if sample_keys is not None:
            target = max(self.index_offset, other.index_offset)
            left = self._pad_index(self, target, sample_keys, sample_positions)
            right = self._pad_index(other, target, sample_keys, sample_positions)
        self.kv_cache.extend(other.kv_cache)
        self.clear_index_blocks()
        if sample_keys is not None:
            self.index_keys = mx.concatenate([left[0], right[0]], axis=0)
            position_axis = 1 if sample_positions.ndim == 3 else 0
            self.index_position_ids = mx.concatenate(
                [left[1], right[1]], axis=position_axis
            )
            self.index_offset = target

    def extract(self, idx):
        cache = QSAKVCache()
        if not self.kv_cache.empty():
            base = self.kv_cache.extract(idx)
            cache.keys, cache.values, cache.offset = (
                base.keys,
                base.values,
                base.offset,
            )
        if self.index_keys is not None:
            padding = int(self.left_padding[idx].item())
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, padding : self.index_offset]
            )
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[
                        :, idx : idx + 1, padding : self.index_offset
                    ]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1, padding : self.index_offset]
                )
            if padding == 0 and self.index_block_keys is not None:
                cache.index_block_keys = mx.contiguous(
                    self.index_block_keys[idx : idx + 1]
                )
                cache.index_block_ratio = self.index_block_ratio
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        caches = list(caches)
        out = cls([0] * len(caches))
        if not caches:
            return out
        if prefix_lens is not None and len(prefix_lens) != len(caches):
            raise ValueError("prefix_lens must have one entry per QSA cache")
        if not all(isinstance(cache, QSAKVCache) for cache in caches):
            types = ", ".join(type(cache).__name__ for cache in caches)
            raise TypeError(
                f"Cannot merge non-QSA caches into BatchQSAKVCache: {types}"
            )
        out.kv_cache = BatchKVCache.merge(caches)
        sample_keys, sample_positions = cls._samples(*caches)
        if sample_keys is None:
            return out
        target = max(cache.offset for cache in caches)
        rows = [
            cls._pad_index(
                SimpleNamespace(
                    index_keys=cache.index_keys,
                    index_position_ids=cache.index_position_ids,
                    index_offset=cache.offset,
                    offset=mx.array([cache.offset]),
                ),
                target,
                sample_keys,
                sample_positions,
            )
            for cache in caches
        ]
        out.index_keys = mx.concatenate([row[0] for row in rows], axis=0)
        position_axis = 1 if sample_positions.ndim == 3 else 0
        out.index_position_ids = mx.concatenate(
            [row[1] for row in rows], axis=position_axis
        )
        out.index_offset = target
        summaries = [cache.index_block_keys for cache in caches]
        ratios = [cache.index_block_ratio for cache in caches]
        if (
            all(summary is not None for summary in summaries)
            and len(set(ratios)) == 1
            and len({cache.offset for cache in caches}) == 1
            and len({summary.shape[2] for summary in summaries}) == 1
        ):
            out.index_block_keys = mx.concatenate(summaries, axis=0)
            out.index_block_ratio = ratios[0]
        return out

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return self.kv_cache.is_trimmable()

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        self.clear_index_blocks()
        return trimmed

    @property
    def state(self):
        kv_state = (
            (None, None, self.kv_cache.offset, self.kv_cache.left_padding)
            if self.kv_cache.empty()
            else self.kv_cache.state
        )
        return (
            kv_state,
            (
                None
                if self.index_keys is None
                else self.index_keys[:, : self.index_offset]
            ),
            (
                None
                if self.index_position_ids is None
                else self.index_position_ids[..., : self.index_offset]
            ),
            self.index_block_keys,
            self.index_block_ratio,
        )

    @state.setter
    def state(self, value):
        if len(value) == 3:
            kv_state, self.index_keys, self.index_position_ids = value
            self.index_block_keys = None
            self.index_block_ratio = None
        else:
            (
                kv_state,
                self.index_keys,
                self.index_position_ids,
                self.index_block_keys,
                self.index_block_ratio,
            ) = value
        left_padding = (
            [0]
            if kv_state is None or len(kv_state) < 4 or kv_state[3] is None
            else kv_state[3]
        )
        self.kv_cache = BatchKVCache(left_padding)
        if kv_state is None or kv_state[0] is None:
            if kv_state is not None and kv_state[2] is not None:
                self.kv_cache.offset = kv_state[2]
        else:
            self.kv_cache.state = kv_state
        self.index_offset = 0 if self.index_keys is None else self.index_keys.shape[1]

    @classmethod
    def from_state(cls, state, meta_state):
        cache = cls.__new__(cls)
        cache.state = state
        cache.meta_state = meta_state
        return cache

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, value):
        if value:
            raise ValueError("BatchQSAKVCache has no separate metadata state")

    @property
    def nbytes(self):
        extra = 0
        if self.index_keys is not None:
            extra = self.index_keys.nbytes + self.index_position_ids.nbytes
        if self.index_block_keys is not None:
            extra += self.index_block_keys.nbytes
        return self.kv_cache.nbytes + extra


class QSAQuantizedKVCache(QuantizedKVCache):
    """Uniformly quantized QSA cache that retains float indexer state."""

    preserve_auxiliary_kv_state = True

    def __init__(self, group_size: int = 64, bits: int = 8):
        super().__init__(group_size=group_size, bits=bits)
        self.index_keys = None
        self.index_position_ids = None
        self.index_block_keys = None
        self.index_block_ratio = None

    def update_indexer(self, keys: mx.array, position_ids: mx.array):
        if self.index_keys is None:
            self.index_keys = keys
            self.index_position_ids = position_ids
        else:
            self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
            self.index_position_ids = _append_indexer_positions(
                self.index_position_ids, position_ids
            )
        return self.index_keys, self.index_position_ids

    @property
    def state(self):
        if self.keys is None:
            keys, values = None, None
        else:
            keys, values = super().state
        return (
            keys,
            values,
            self.index_keys,
            self.index_position_ids,
            self.index_block_keys,
            self.index_block_ratio,
        )

    @state.setter
    def state(self, value):
        if len(value) == 4:
            self.keys, self.values, self.index_keys, self.index_position_ids = value
            self.index_block_keys = None
            self.index_block_ratio = None
        else:
            (
                self.keys,
                self.values,
                self.index_keys,
                self.index_position_ids,
                self.index_block_keys,
                self.index_block_ratio,
            ) = value
        self.offset = 0 if self.keys is None else self.keys[0].shape[2]

    def clear_index_blocks(self):
        self.index_block_keys = None
        self.index_block_ratio = None

    def trim(self, n):
        n = min(self.offset, n)
        super().trim(n)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[:, : self.offset]
            self.index_position_ids = self.index_position_ids[..., : self.offset]
        self.clear_index_blocks()
        return n

    def extract(self, idx):
        cache = QSAQuantizedKVCache(self.group_size, self.bits)
        if self.keys is not None:
            cache.keys = tuple(mx.contiguous(x[idx : idx + 1]) for x in self.keys)
            cache.values = tuple(mx.contiguous(x[idx : idx + 1]) for x in self.values)
            cache.offset = self.offset
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(self.index_keys[idx : idx + 1])
            if self.index_position_ids.ndim == 3:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[:, idx : idx + 1]
                )
            else:
                cache.index_position_ids = mx.contiguous(
                    self.index_position_ids[idx : idx + 1]
                )
        if self.index_block_keys is not None:
            cache.index_block_keys = mx.contiguous(self.index_block_keys[idx : idx + 1])
            cache.index_block_ratio = self.index_block_ratio
        return cache

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = tuple(x[batch_indices] for x in self.keys)
            self.values = tuple(x[batch_indices] for x in self.values)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if self.index_position_ids.ndim == 3:
                self.index_position_ids = self.index_position_ids[:, batch_indices]
            else:
                self.index_position_ids = self.index_position_ids[batch_indices]
        if self.index_block_keys is not None:
            self.index_block_keys = self.index_block_keys[batch_indices]

    @property
    def nbytes(self):
        size = 0 if self.keys is None else super().nbytes
        if self.index_keys is not None:
            size += self.index_keys.nbytes + self.index_position_ids.nbytes
        if self.index_block_keys is not None:
            size += self.index_block_keys.nbytes
        return size


class Qwen4ExpRMSNorm(nn.Module):
    """Qwen4 RMSNorm, whose checkpoint weights are centered at zero."""

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size:
            raise ValueError(f"{dim=} must be divisible by {group_size=}")
        self.weight = mx.zeros(dim)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        y = x.astype(mx.float32)
        if self.group_size is not None:
            y = y.reshape(*y.shape[:-1], -1, self.group_size)
            weight = self.weight.reshape(-1, self.group_size)
        else:
            weight = self.weight
        y = y * mx.rsqrt(mx.mean(mx.square(y), axis=-1, keepdims=True) + self.eps)
        y = y * (1.0 + weight.astype(mx.float32))
        return y.reshape(x.shape).astype(dtype)


class Qwen4ExpRMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float, activation: str):
        super().__init__()
        self.eps = eps
        self.activation = activation
        self.weight = mx.ones(dim)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        dtype = x.dtype
        y = mx.fast.rms_norm(x, self.weight, self.eps).astype(mx.float32)
        gate = gate.astype(mx.float32)
        if self.activation == "sigmoid":
            gate = mx.sigmoid(gate)
        else:
            gate = nn.silu(gate)
        return (y * gate).astype(dtype)


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            activation=config.output_gate_type or config.hidden_act,
        )

    def _normalize_qk(self, q: mx.array, k: mx.array):
        # Transformers/FLA uses L2 normalization (epsilon after the sum),
        # followed by the usual 1/sqrt(head_dim) query scaling.
        scale = q.shape[-1] ** -0.5
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        return q * scale, k


class Qwen4ExpQSAIndexer(nn.Module):
    """Select compressed key blocks using Qwen Sparse Attention scores."""

    def __init__(self, config: TextConfig, rotary_emb):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.kv_heads = config.indexer_kv_heads
        self.head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_emb = rotary_emb
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _default_position_ids(batch: int, start: int | mx.array, length: int):
        steps = mx.arange(length, dtype=mx.int32)
        if isinstance(start, mx.array) and start.ndim > 0:
            return start[:batch, None].astype(mx.int32) + steps[None]
        if isinstance(start, mx.array):
            start = int(start.item())
        return mx.broadcast_to((start + steps)[None], (batch, length))

    def _apply_rope(self, x: mx.array, position_ids: mx.array) -> mx.array:
        # MRoPE's helper applies the same partial rotary transform to both
        # operands, so use a throwaway second operand for indexer-only states.
        rotated, _ = self.rotary_emb.apply_rotary(x, x, position_ids, unsqueeze_dim=1)
        return rotated

    def __call__(
        self,
        hidden_states: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
    ) -> Optional[mx.array]:
        return self.from_projected(
            self.index_qk_proj(hidden_states), cache, position_ids
        )

    def select(
        self,
        hidden_states: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
    ) -> Optional[SimpleNamespace]:
        return self.select_from_projected(
            self.index_qk_proj(hidden_states), cache, position_ids
        )

    def from_projected(
        self,
        qk: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
    ) -> Optional[mx.array]:
        selection = self.select_from_projected(qk, cache, position_ids)
        return None if selection is None else self.build_mask(selection)

    def select_from_projected(
        self,
        qk: mx.array,
        cache: Optional[QSAKVCache],
        position_ids: Optional[mx.array],
    ) -> Optional[SimpleNamespace]:
        batch, seq_len, _ = qk.shape
        past_len = cache.offset if cache is not None else 0
        past_index_len = getattr(cache, "index_offset", past_len)
        if position_ids is None:
            position_ids = self._default_position_ids(batch, past_len, seq_len)

        qk = qk.reshape(batch, seq_len, self.n_heads + self.kv_heads, self.head_dim)
        query = qk[:, :, : self.n_heads]
        raw_keys = qk[:, :, self.n_heads :].squeeze(2)
        query = self.q_layernorm(query).transpose(0, 2, 1, 3)

        if cache is not None:
            raw_keys, full_position_ids = cache.update_indexer(raw_keys, position_ids)
        else:
            full_position_ids = position_ids

        key_len = raw_keys.shape[1]
        max_complete_blocks = key_len // self.compress_ratio
        if max_complete_blocks <= self.block_topk:
            return None

        query = self._apply_rope(query, position_ids)
        left_padding = getattr(cache, "left_padding", None)
        padding_info = _qwen3_5_left_padding_info(cache)
        zero_padding = padding_info is None or padding_info[1] == 0
        if not isinstance(left_padding, mx.array) or left_padding.ndim == 0:
            left_padding = mx.zeros((batch,), dtype=mx.int32)
        else:
            left_padding = left_padding[:batch].astype(mx.int32)

        block_ids = mx.arange(max_complete_blocks, dtype=mx.int32)
        block_starts = left_padding[:, None] + block_ids[None] * self.compress_ratio
        cached_block_keys = getattr(cache, "index_block_keys", None)
        cached_block_ratio = getattr(cache, "index_block_ratio", None)
        can_reuse_blocks = (
            zero_padding
            and isinstance(cached_block_keys, mx.array)
            and cached_block_keys.ndim == 4
            and cached_block_keys.shape[0] == batch
            and cached_block_keys.shape[1] == 1
            and cached_block_keys.shape[-1] == self.head_dim
            and cached_block_keys.shape[2] <= max_complete_blocks
            and cached_block_ratio == self.compress_ratio
        )
        first_new_block = cached_block_keys.shape[2] if can_reuse_blocks else 0
        new_block_ids = mx.arange(first_new_block, max_complete_blocks, dtype=mx.int32)
        if zero_padding:
            # Use a reshape/view for batches without left padding to avoid
            # gathering the full key history. Retain already normalized and
            # rotated complete blocks on the cache, so each subsequent prefill
            # chunk only summarizes newly completed blocks.
            new_token_start = first_new_block * self.compress_ratio
            complete_key_len = max_complete_blocks * self.compress_ratio
            pooled_keys = raw_keys[:, new_token_start:complete_key_len].reshape(
                batch,
                max_complete_blocks - first_new_block,
                self.compress_ratio,
                self.head_dim,
            )
        else:
            # Compressed blocks group the tokens visible to each row. With
            # left padding they begin at different physical cache columns.
            block_token_indices = (
                block_starts[..., None]
                + mx.arange(self.compress_ratio, dtype=mx.int32)[None, None]
            )
            safe_token_indices = mx.minimum(block_token_indices, key_len - 1)
            flat_token_indices = safe_token_indices.reshape(batch, -1)
            flat_token_indices = mx.broadcast_to(
                flat_token_indices[..., None],
                (*flat_token_indices.shape, self.head_dim),
            )
            pooled_keys = mx.take_along_axis(raw_keys, flat_token_indices, axis=1)
            pooled_keys = pooled_keys.reshape(
                batch, max_complete_blocks, self.compress_ratio, self.head_dim
            )
        if max_complete_blocks > first_new_block:
            pooled_keys = mx.expand_dims(
                self.k_layernorm(
                    mx.mean(pooled_keys.astype(mx.float32), axis=2).astype(
                        raw_keys.dtype
                    )
                ),
                axis=1,
            )
            if zero_padding:
                block_position_ids = full_position_ids[
                    ..., new_block_ids * self.compress_ratio
                ]
            elif full_position_ids.ndim == 3:
                safe_block_starts = mx.minimum(block_starts, key_len - 1)
                position_indices = mx.broadcast_to(
                    safe_block_starts[None],
                    (full_position_ids.shape[0], *safe_block_starts.shape),
                )
                block_position_ids = mx.take_along_axis(
                    full_position_ids, position_indices, axis=2
                )
            else:
                safe_block_starts = mx.minimum(block_starts, key_len - 1)
                block_position_ids = mx.take_along_axis(
                    full_position_ids, safe_block_starts, axis=1
                )
            pooled_keys = self._apply_rope(pooled_keys, block_position_ids)
            if can_reuse_blocks and first_new_block > 0:
                pooled_keys = mx.concatenate([cached_block_keys, pooled_keys], axis=2)
        else:
            pooled_keys = cached_block_keys

        if zero_padding and cache is not None:
            cache.index_block_keys = pooled_keys
            cache.index_block_ratio = self.compress_ratio
        elif cache is not None:
            clear_index_blocks = getattr(cache, "clear_index_blocks", None)
            if callable(clear_index_blocks):
                clear_index_blocks()

        # Score in float32, as the reference does: which blocks win is a discrete
        # choice, and rounding the products flips the ones near the cut-off.
        scores = query.astype(mx.float32) @ pooled_keys.astype(mx.float32).transpose(
            0, 1, 3, 2
        )
        scores = mx.sum(mx.maximum(scores, 0), axis=1)
        scores = scores / math.sqrt(self.head_dim)

        query_ends = past_index_len + mx.arange(seq_len, dtype=mx.int32) + 1
        visible_counts = mx.maximum(query_ends[None] - left_padding[:, None], 0)
        complete_counts = visible_counts // self.compress_ratio
        valid_blocks = block_ids[None, None] < complete_counts[..., None]
        scores = mx.where(valid_blocks, scores, -mx.inf)
        selected_blocks = mx.argpartition(scores, kth=-self.block_topk, axis=-1)[
            ..., -self.block_topk :
        ]

        return SimpleNamespace(
            selected_blocks=selected_blocks,
            query_ends=mx.broadcast_to(query_ends[None], (batch, seq_len)),
            complete_counts=complete_counts,
            left_padding=left_padding,
            key_len=key_len,
            zero_padding=zero_padding,
            all_sparse=(
                zero_padding
                and int(past_index_len) + 1
                >= (self.block_topk + 1) * self.compress_ratio
            ),
        )

    def build_mask(self, selection: SimpleNamespace) -> mx.array:
        selected_blocks = selection.selected_blocks
        query_ends = selection.query_ends
        complete_counts = selection.complete_counts
        left_padding = selection.left_padding
        key_len = selection.key_len
        batch, seq_len, _ = selected_blocks.shape

        # Scatter the winning blocks directly onto the token axis. Comparing
        # every token against every pick would cost seq_len * key_len *
        # block_topk bytes per prefill step.
        selected_token_indices = (
            left_padding[:, None, None, None]
            + selected_blocks[..., None] * self.compress_ratio
            + mx.arange(self.compress_ratio, dtype=mx.int32)[None, None, None]
        ).reshape(batch, seq_len, -1)
        valid_selected_tokens = selected_token_indices < key_len
        selected_token_indices = mx.where(
            valid_selected_tokens, selected_token_indices, key_len
        )
        selected_tokens = mx.put_along_axis(
            mx.zeros((batch, seq_len, key_len + 1), dtype=mx.bool_),
            selected_token_indices,
            valid_selected_tokens,
            axis=-1,
        )[..., :key_len]

        token_indices = mx.arange(key_len)
        tail_starts = left_padding[:, None] + complete_counts * self.compress_ratio
        tail = (token_indices[None, None, :] >= tail_starts[..., None]) & (
            token_indices[None, None, :] < query_ends[..., None]
        )
        causal = (token_indices[None, None, :] >= left_padding[:, None, None]) & (
            token_indices[None, None, :] < query_ends[..., None]
        )
        use_sparse = complete_counts > self.block_topk
        selected_tokens = mx.where(
            use_sparse[..., None], selected_tokens | tail, causal
        )
        return selected_tokens[:, None]


class Qwen4ExpAttention(Qwen3_5Attention):
    def __init__(self, config: TextConfig):
        super().__init__(config)
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpQSAIndexer(config, self.rotary_emb)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
        position_embeddings: Optional[tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        selection = self.indexer.select(x, cache, position_ids)
        if selection is None:
            return super().__call__(
                x,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        standard_causal_mask = mask is None or (
            isinstance(mask, str) and mask == "causal"
        )
        if isinstance(mask, str) and not standard_causal_mask:
            # Qwen3.5 owns specialized string modes such as
            # ``left_padded_decode``. The indexer state is already updated;
            # delegate only the Q/K/V attention path to the parent.
            return super().__call__(
                x,
                mask=mask,
                cache=cache,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        use_sparse_kernel = selection.all_sparse and standard_causal_mask
        qsa_mask = None if use_sparse_kernel else self.indexer.build_mask(selection)
        if qsa_mask is not None:
            if standard_causal_mask:
                mask = qsa_mask
            elif isinstance(mask, mx.array):
                if mask.dtype == mx.bool_:
                    mask = mask & qsa_mask
                else:
                    sparse_bias = mx.where(qsa_mask, 0.0, -mx.inf).astype(mask.dtype)
                    mask = mask + sparse_bias

        B, L, _ = x.shape
        q_proj_output, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries, keys, values, gate, mask = self._prepare_projected_qkv(
            q_proj_output,
            keys,
            values,
            cache,
            position_ids,
            position_embeddings,
            None if use_sparse_kernel else mask,
        )
        output = None
        if use_sparse_kernel:
            output = qsa_sparse_attention(
                queries,
                keys,
                values,
                selection.selected_blocks,
                selection.query_ends,
                scale=self.scale,
                block_size=self.indexer.compress_ratio,
            )
        if output is None:
            if qsa_mask is None:
                qsa_mask = self.indexer.build_mask(selection)
                mask = qsa_mask
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


class Qwen4ExpGatedResidual(nn.Module):
    def __init__(self, config: TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.input_mix_weight_down = nn.Linear(
            hc_hidden_size, config.hc_lowrank, bias=False
        )
        self.input_mix_weight_up = nn.Linear(
            config.hc_lowrank, hc_hidden_size, bias=False
        )
        if use_combine:
            self.block_inject_weight = nn.Linear(
                hc_hidden_size, self.hc_count, bias=False
            )

    def __call__(self, hyper_input: mx.array):
        normed = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normed.reshape(*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed_input = mx.mean(mix * streams, axis=-2)
        if "block_inject_weight" not in self:
            return mixed_input
        injection_weights = 2 * mx.sigmoid(
            self.block_inject_weight(normed) / self.hc_count
        )
        return mixed_input, hyper_input, injection_weights


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(
    unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return mx.array(multipliers, dtype=mx.int64)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class ShardedEmbedding(nn.Module):
    """Embedding kept in checkpoint-sized row shards to avoid a 100 GB join."""

    def __init__(self, num_embeddings: int, dims: int, num_shards: int):
        super().__init__()
        if num_shards <= 0 or num_shards > num_embeddings:
            raise ValueError("num_shards must be in [1, num_embeddings]")
        base, remainder = divmod(num_embeddings, num_shards)
        self.shard_sizes = tuple(
            base + (1 if index < remainder else 0) for index in range(num_shards)
        )
        self.shards = [nn.Embedding(size, dims) for size in self.shard_sizes]
        offsets = [0]
        for size in self.shard_sizes:
            offsets.append(offsets[-1] + size)
        self.shard_offsets = tuple(offsets)
        self.dims = dims

    def __call__(self, indices: mx.array) -> mx.array:
        flat = indices.reshape(-1)
        # One tiny host sync avoids scheduling gathers against all 128 giant
        # PLE shards for every token.
        mx.eval(flat)
        host_indices = [int(index) for index in flat.tolist()]
        if not host_indices:
            return self.shards[0](flat).reshape(*indices.shape, self.dims)
        if any(index < 0 or index >= self.shard_offsets[-1] for index in host_indices):
            raise IndexError("embedding index is outside the sharded vocabulary")

        shard_indices = [
            bisect_right(self.shard_offsets, index) - 1 for index in host_indices
        ]
        result = None
        for shard_index in sorted(set(shard_indices)):
            positions_list = [
                position
                for position, current_shard in enumerate(shard_indices)
                if current_shard == shard_index
            ]
            local_indices = [
                host_indices[position] - self.shard_offsets[shard_index]
                for position in positions_list
            ]
            positions = mx.array(positions_list, dtype=mx.int32)
            values = self.shards[shard_index](mx.array(local_indices, dtype=mx.int32))
            if result is None:
                result = mx.zeros((len(host_indices), self.dims), dtype=values.dtype)
            result = result.at[positions].add(values)
        return result.reshape(*indices.shape, self.dims)


class Qwen4ExpNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: TextConfig,
        embedding_dim: int,
        layer_idx: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.seed = config.seed
        eos = config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos

        head_vocab_sizes = []
        head_offsets = []
        total_vocab_size = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(
                config.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            head_vocab_sizes.append(size)
            head_offsets.append(total_vocab_size)
            total_vocab_size += size

        self.layer_multipliers = _build_layer_multipliers(
            self.unigram_vocab_size,
            self.ngram_size,
            ple_layer_index,
            self.seed,
        )
        self.ngram_heads_vocab_sizes = mx.array(head_vocab_sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(head_offsets, dtype=mx.int64)
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(total_vocab_size / divisor) * divisor
        if config.ple_storage:
            from .ple_storage import QuantizedMMapNGramEmbedding

            manifest = config.ple_storage.get("manifest")
            if not manifest:
                raise ValueError("ple_storage requires a manifest path")
            self.ngram_embedding = QuantizedMMapNGramEmbedding(
                manifest, cache_rows=config.ple_storage.get("cache_rows")
            )
            if self.ngram_embedding.row_count != padded_vocab_size:
                raise ValueError(
                    "external PLE row count does not match model configuration: "
                    f"{self.ngram_embedding.row_count} != {padded_vocab_size}"
                )
            row_width = embedding_dim // self.ngram_heads
            if self.ngram_embedding.row_width != row_width:
                raise ValueError(
                    "external PLE row width does not match model configuration"
                )
        else:
            self.ngram_embedding = ShardedEmbedding(
                padded_vocab_size,
                embedding_dim // self.ngram_heads,
                config.split_ngram_parts,
            )

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int):
        if shift == 0:
            return token_ids
        batch, seq_len = token_ids.shape
        positions = mx.arange(seq_len, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((batch, 1), -1, dtype=mx.int64), previous_eos_inclusive[:, :-1]],
            axis=1,
        )
        segment_start = previous_eos + 1
        position_in_segment = positions[None] - segment_start
        source_positions = positions - shift
        gather_positions = mx.broadcast_to(
            mx.maximum(source_positions, 0)[None], (batch, seq_len)
        )
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None] >= 0)
        return mx.where(valid, shifted, self.eos_token_id)

    def __call__(self, input_ids: mx.array, cache: Optional[ArraysCache]):
        input_ids = input_ids.astype(mx.int64)
        batch = input_ids.shape[0]
        if cache is not None and cache[3] is not None:
            previous_context = cache[3]
        else:
            previous_context = mx.full(
                (batch, self.context_len), self.eos_token_id, dtype=mx.int64
            )

        token_history = mx.concatenate([previous_context, input_ids], axis=-1)
        if cache is not None:
            cache[3] = mx.contiguous(token_history[:, -self.context_len :])

        shifted_tokens = [
            self._shift_right_ignore_eos(token_history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = mx.bitwise_xor(
                    mixed_ids,
                    shifted_tokens[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ngram_ids = mixed_ids[..., None] % sizes[None, None]
            blocks.append(ngram_ids + offsets[None, None])

        ngram_ids = mx.concatenate(blocks, axis=-1)[:, -input_ids.shape[1] :]
        embeddings = self.ngram_embedding(ngram_ids)
        return embeddings.reshape(*embeddings.shape[:-2], -1)


class Qwen4ExpPLELayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config, config.ple_embed_dim, layer_idx, ple_layer_index
        )
        self.key_proj = nn.Linear(config.ple_embed_dim, hc_hidden_size, bias=False)
        self.value_proj = nn.Linear(config.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_query = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm_conv = Qwen4ExpRMSNorm(
            hc_hidden_size,
            group_size=self.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.conv_dilation = config.ngram_size
        self.short_conv_state_len = (
            config.ple_conv_kernel_size - 1
        ) * self.conv_dilation
        self.conv1d = nn.Conv1d(
            hc_hidden_size,
            hc_hidden_size,
            kernel_size=config.ple_conv_kernel_size,
            dilation=self.conv_dilation,
            groups=hc_hidden_size,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache: Optional[ArraysCache]):
        batch = x.shape[0]
        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros(
                (batch, self.short_conv_state_len, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[2] = mx.contiguous(conv_input[:, -self.short_conv_state_len :])
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Optional[ArraysCache],
        mask: Optional[mx.array],
    ):
        embeddings = self.ple_embedding(input_ids, cache)
        keys = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        values = self.value_proj(embeddings)
        queries = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated_values = mx.sigmoid(gate) * values[..., None, :]
        gated_values = gated_values.reshape(*hidden_states.shape)
        normed = self.norm_conv(gated_values)
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            gated_values = mx.where(mask[..., None], gated_values, 0)
            normed = mx.where(mask[..., None], normed, 0)
        return gated_values + self._short_conv(normed, cache)


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = Qwen4ExpGatedDeltaNet(config)
        else:
            self.self_attn = Qwen4ExpAttention(config)
        self.mlp = Qwen3_5MoeSparseMoeBlock(config)
        ple_index = (
            config.ple_layer_ids.index(layer_idx + 1)
            if layer_idx + 1 in config.ple_layer_ids
            else None
        )
        if ple_index is not None:
            self.ple = Qwen4ExpPLELayer(config, layer_idx, ple_index)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        position_ids: Optional[mx.array],
    ):
        if "ple" in self:
            hidden_states = hidden_states + self.ple(
                hidden_states, input_ids, cache, mask
            )

        mixed, hyper_input, injection_weights = self.attn_hyper_connection(
            hidden_states
        )
        if self.is_linear:
            branch = self.linear_attn(mixed, mask=mask, cache=cache)
        else:
            branch = self.self_attn(
                mixed, mask=mask, cache=cache, position_ids=position_ids
            )
        injection = branch[..., None, :] * injection_weights[..., None]
        hidden_states = hyper_input + injection.reshape(*hyper_input.shape)

        mixed, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
        branch = self.mlp(mixed)
        injection = branch[..., None, :] * injection_weights[..., None]
        return hyper_input + injection.reshape(*hyper_input.shape)


class Qwen4ExpModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.args = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen4ExpDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)
        self.ssm_idx = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), 0
        )
        self.fa_idx = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), 0
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        position_ids: Optional[mx.array] = None,
        capture_layer_ids=None,
        hidden_sink=None,
        **kwargs,
    ):
        del kwargs
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )
        if cache is None:
            cache = [None] * len(self.layers)

        # Ragged prefill has row-specific recurrent, PLE, RoPE, and QSA
        # semantics. Process each row through the singleton-equivalent path,
        # merge the resulting caches, and continue decode as a batch.
        fa_cache = cache[self.fa_idx]
        if (
            hidden_states.shape[0] > 1
            and hidden_states.shape[1] > 1
            and hidden_sink is None
            and fa_cache is not None
            and hasattr(fa_cache, "extract")
            and hasattr(fa_cache.__class__, "merge")
            and isinstance(getattr(fa_cache, "offset", None), mx.array)
            and fa_cache.offset.ndim > 0
        ):
            query_left_padding = mx.minimum(
                mx.maximum(-fa_cache.offset, 0), hidden_states.shape[1]
            )
            cache_left_padding = getattr(fa_cache, "left_padding", None)
            has_left_padding = (
                isinstance(cache_left_padding, mx.array)
                and cache_left_padding.ndim > 0
                and int(cache_left_padding.max().item()) > 0
            )
            if has_left_padding or int(query_left_padding.max().item()) > 0:
                row_outputs = []
                row_caches = [[] for _ in cache]
                batch_offsets = []
                for cache_entry in cache:
                    offsets = getattr(cache_entry, "offset", None)
                    if (
                        isinstance(offsets, mx.array)
                        and offsets.ndim > 0
                        and offsets.size >= hidden_states.shape[0]
                    ):
                        batch_offsets.append(offsets[: hidden_states.shape[0]])
                    else:
                        batch_offsets.append(None)

                for row, pad in enumerate(query_left_padding.tolist()):
                    pad = min(max(int(pad), 0), hidden_states.shape[1])
                    current_cache = []
                    for cache_entry in cache:
                        if cache_entry is None:
                            current_cache.append(None)
                        elif isinstance(cache_entry, BatchQSAKVCache):
                            current_cache.append(cache_entry.extract(row))
                        else:
                            current_cache.append(_extract_row_cache(cache_entry, row))
                    if pad == hidden_states.shape[1]:
                        row_outputs.append(mx.zeros_like(hidden_states[row : row + 1]))
                        for index, cache_entry in enumerate(current_cache):
                            layer = self.layers[index]
                            if (
                                isinstance(cache_entry, ArraysCache)
                                and "ple" in layer
                                and cache_entry[3] is None
                            ):
                                embedding = layer.ple.ple_embedding
                                cache_entry[3] = mx.full(
                                    (1, embedding.context_len),
                                    embedding.eos_token_id,
                                    dtype=mx.int64,
                                )
                            row_caches[index].append(cache_entry)
                        continue

                    row_position_ids = None
                    if position_ids is not None:
                        if position_ids.ndim == 2:
                            row_position_ids = position_ids[row : row + 1, pad:]
                        else:
                            row_position_ids = position_ids[:, row : row + 1, pad:]
                    row_mask = mask
                    if isinstance(mask, mx.array) and mask.ndim == 2:
                        row_mask = mask[row : row + 1, pad:]
                    row_output = self(
                        inputs[row : row + 1, pad:],
                        inputs_embeds=hidden_states[row : row + 1, pad:],
                        mask=row_mask,
                        cache=current_cache,
                        position_ids=row_position_ids,
                    )
                    if pad > 0:
                        row_output = _pad_row_time(
                            row_output, pad, hidden_states.shape[1]
                        )
                    row_outputs.append(row_output)
                    for index, cache_entry in enumerate(current_cache):
                        row_caches[index].append(cache_entry)

                for index, entries in enumerate(row_caches):
                    if cache[index] is None:
                        continue
                    if hasattr(cache[index].__class__, "merge"):
                        cache[index] = _restore_batch_padding_metadata(
                            cache[index].__class__.merge(entries),
                            batch_offsets[index],
                            hidden_states.shape[1],
                        )
                return mx.concatenate(row_outputs, axis=0)

        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        fa_mask = _create_qwen3_5_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(hidden_states, cache[self.ssm_idx])
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            ssm_mask = mask

        capture = set(capture_layer_ids or [])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(
                hidden_states,
                inputs,
                mask=layer_mask,
                cache=layer_cache,
                position_ids=position_ids,
            )
            if hidden_sink is not None and index in capture:
                hidden_sink.append(self.hyper_connection_mixer(hidden_states))

        if hidden_sink is not None and capture_layer_ids == []:
            # Native Qwen4 MTP consumes the complete hyper-connection state,
            # before the final 4-stream mixer.  An explicit empty capture list
            # is reserved for that final pre-mixer capture; ordinary layer
            # captures above retain their public, mixed hidden-state contract.
            hidden_sink.append(hidden_states)
        return self.hyper_connection_mixer(hidden_states)


class Qwen4ExpExactSpeculativeVerifier(Qwen3_5ExactSpeculativeVerifier):
    """Batched verifier with Qwen4's singleton-equivalent dense operations."""

    @staticmethod
    def _normalize_gated_delta_qk(layer, q, k):
        return layer._normalize_qk(q, k)

    def _hyper_connection(self, module, hidden_states):
        normed = module.hc_norm(hidden_states)
        mix = nn.silu(
            self._linear(module.input_mix_weight_down, normed) / module.hc_count
        )
        mix = mx.sigmoid(self._linear(module.input_mix_weight_up, mix))
        mix = mix.reshape(*mix.shape[:-1], module.hc_count, module.hidden_size)
        streams = normed.reshape(
            *normed.shape[:-1], module.hc_count, module.hidden_size
        )
        mixed = mx.mean(mix * streams, axis=-2)
        if "block_inject_weight" not in module:
            return mixed
        injection = 2 * mx.sigmoid(
            self._linear(module.block_inject_weight, normed) / module.hc_count
        )
        return mixed, hidden_states, injection

    @staticmethod
    def _inject(branch, hyper_input, injection_weights):
        injection = branch[..., None, :] * injection_weights[..., None]
        return hyper_input + injection.reshape(*hyper_input.shape)

    def _ple(self, module, hidden_states, input_ids, cache, mask):
        embeddings = module.ple_embedding(input_ids, cache)
        keys = module.norm_key(self._linear(module.key_proj, embeddings)).reshape(
            *hidden_states.shape[:-1], module.hc_count, module.hidden_size
        )
        values = self._linear(module.value_proj, embeddings)
        queries = module.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], module.hc_count, module.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            module.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated_values = (mx.sigmoid(gate) * values[..., None, :]).reshape(
            *hidden_states.shape
        )
        normed = module.norm_conv(gated_values)
        if mask is not None and isinstance(mask, mx.array) and mask.ndim == 2:
            gated_values = mx.where(mask[..., None], gated_values, 0)
            normed = mx.where(mask[..., None], normed, 0)
        return gated_values + module._short_conv(normed, cache)

    def _qsa_mask(self, attention, hidden_states, cache, position_ids, mask):
        projected = self._linear(attention.indexer.index_qk_proj, hidden_states)
        qsa_mask = attention.indexer.from_projected(projected, cache, position_ids)
        if qsa_mask is None:
            return mask
        if mask is None or (isinstance(mask, str) and mask == "causal"):
            return qsa_mask
        if isinstance(mask, mx.array):
            if mask.dtype == mx.bool_:
                return mask & qsa_mask
            sparse_bias = mx.where(qsa_mask, 0.0, -mx.inf).astype(mask.dtype)
            return mask + sparse_bias
        return mask

    def _layer(
        self,
        layer,
        hidden,
        input_ids,
        mask,
        cache,
        position_ids,
        gdn_sink,
    ):
        if "ple" in layer:
            hidden = hidden + self._ple(layer.ple, hidden, input_ids, cache, mask)

        mixed, hyper_input, injection = self._hyper_connection(
            layer.attn_hyper_connection, hidden
        )
        if layer.is_linear:
            branch = self._gated_delta(layer.linear_attn, mixed, mask, cache, gdn_sink)
        else:
            attention_mask = self._qsa_mask(
                layer.self_attn, mixed, cache, position_ids, mask
            )
            branch = self._attention(
                layer.self_attn,
                mixed,
                attention_mask,
                cache,
                position_ids,
                None,
            )
        hidden = self._inject(branch, hyper_input, injection)

        mixed, hyper_input, injection = self._hyper_connection(
            layer.mlp_hyper_connection, hidden
        )
        branch = self._feed_forward(layer.mlp, mixed)
        return self._inject(branch, hyper_input, injection)

    def _model(
        self,
        model,
        inputs,
        cache,
        inputs_embeds,
        position_ids,
        gdn_sink,
    ):
        hidden = model.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        hidden = mx.tile(hidden, (1, 1, model.args.hc_count))
        if cache is None:
            cache = [None] * len(model.layers)
        fa_mask = _create_qwen3_5_attention_mask(hidden, cache[model.fa_idx])
        ssm_mask = _create_qwen3_5_ssm_mask(hidden, cache[model.ssm_idx])
        for layer, layer_cache in zip(model.layers, cache):
            layer_mask = ssm_mask if layer.is_linear else fa_mask
            hidden = self._layer(
                layer,
                hidden,
                inputs,
                layer_mask,
                layer_cache,
                position_ids,
                gdn_sink,
            )
        return hidden

    def __call__(
        self,
        language_model,
        inputs,
        *,
        cache=None,
        inputs_embeds=None,
        position_ids=None,
        skip_logits=False,
    ):
        gdn_sink = []
        hidden = self._model(
            language_model.model,
            inputs,
            cache,
            inputs_embeds,
            position_ids,
            gdn_sink,
        )
        logits_hidden = self._hyper_connection(
            language_model.model.hyper_connection_mixer, hidden
        )
        if skip_logits:
            logits = None
        elif language_model.args.tie_word_embeddings:
            logits = self._embedding_as_linear(
                language_model.model.embed_tokens, logits_hidden
            )
        else:
            logits = self._linear(language_model.lm_head, logits_hidden)
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden],
            gdn_states=gdn_sink,
            shared_kv_states={},
        )


_QWEN4_EXACT_SPECULATIVE_VERIFIER = Qwen4ExpExactSpeculativeVerifier()


class LanguageModel(Qwen3_5LanguageModel):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        nn.Module.__init__(self)
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        return_hidden = bool(kwargs.get("return_hidden", False))
        if return_hidden and kwargs.get("capture_layer_ids") is None:
            # Force Qwen4ExpModel to capture its final pre-mixer HC state.  The
            # inherited wrapper also appends the mixed output; keep only the
            # first entry so MTP always sees [B, L, hc_count * hidden_size].
            kwargs["capture_layer_ids"] = []
        output = super().__call__(inputs, inputs_embeds, mask, cache, **kwargs)
        if return_hidden and output.hidden_states:
            output.hidden_states = [output.hidden_states[0]]
        return output

    def _mtp_logits_hidden(self, hidden: mx.array) -> mx.array:
        hc_width = self.args.hc_count * self.args.hidden_size
        if hidden.shape[-1] == hc_width:
            return self.model.hyper_connection_mixer(hidden)
        return hidden

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return super().speculative_logits_from_hidden(self._mtp_logits_hidden(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array):
        return super().speculative_argmax_from_hidden(self._mtp_logits_hidden(hidden))

    def speculative_draft_hidden(self, hidden: mx.array) -> mx.array:
        expected = self.args.hc_count * self.args.hidden_size
        if hidden.ndim != 3 or hidden.shape[-1] != expected:
            raise ValueError(
                "Qwen4-Exp MTP expects target hidden shape "
                "[batch, tokens, hc_count * hidden_size]."
            )
        return hidden

    def fused_greedy_decode(
        self,
        inputs: mx.array,
        cache=None,
        logits_processors=None,
        **kwargs,
    ):
        if (
            self.args.tie_word_embeddings
            or not _QWEN4_EXACT_SPECULATIVE_VERIFIER.can_quantized_head(self.lm_head)
            or "bias" in self.lm_head
        ):
            return None

        token_mask = None
        if logits_processors:
            if not self.supports_fused_greedy_logits_processors(logits_processors):
                return None
            token_mask = mx.concatenate(
                [
                    processors[0].prepare_next_token_mask(token)
                    for processors, token in zip(
                        logits_processors, inputs[:, -1].tolist()
                    )
                ],
                axis=0,
            )
            token_mask = _QWEN4_EXACT_SPECULATIVE_VERIFIER.pad_token_mask(
                token_mask, self.lm_head.weight.shape[0]
            )

        output = self(
            inputs,
            cache=cache,
            return_hidden=True,
            skip_logits=True,
            **kwargs,
        )
        hidden = self._mtp_logits_hidden(output.hidden_states[-1])
        sampled = _QWEN4_EXACT_SPECULATIVE_VERIFIER.quantized_argmax(
            self.lm_head, hidden, token_mask=token_mask
        )
        if sampled is not None:
            return sampled
        if token_mask is not None:
            raise RuntimeError("masked fused greedy decode became unsupported")
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    @staticmethod
    def _snapshot_speculative_cache(caches):
        snapshots = []
        for entry in caches:
            if isinstance(entry, ArraysCache):
                snapshots.append(
                    (
                        "arrays",
                        list(entry.state),
                        getattr(entry, "_left_padding", None),
                        getattr(entry, "_left_padding_advance", 0),
                        getattr(entry, "_lengths", None),
                        getattr(entry, "_lengths_advance", 0),
                    )
                )
            else:
                state = entry.state
                if isinstance(state, list):
                    state = list(state)
                elif isinstance(state, tuple):
                    state = tuple(state)
                snapshots.append(("cache", state, entry.meta_state))
        return snapshots

    @staticmethod
    def _restore_speculative_cache(caches, snapshots):
        for entry, snapshot in zip(caches, snapshots):
            if snapshot[0] == "arrays":
                (
                    _,
                    state,
                    left_padding,
                    left_padding_advance,
                    lengths,
                    lengths_advance,
                ) = snapshot
                entry.state = list(state)
                entry._left_padding = left_padding
                entry._left_padding_advance = left_padding_advance
                entry._lengths = lengths
                entry._lengths_advance = lengths_advance
            else:
                _, state, meta_state = snapshot
                entry.state = state
                entry.meta_state = meta_state

    def _speculative_verify(self, inputs: mx.array, cache, sampler=None):
        snapshot = self._snapshot_speculative_cache(cache)
        batch, length = inputs.shape
        cache_entry = cache[self.model.fa_idx]
        cache_offset = getattr(cache_entry, "offset", 0)
        if isinstance(cache_offset, mx.array) and cache_offset.ndim > 0:
            offsets = cache_offset[:batch].astype(mx.int64)
        else:
            offsets = mx.full((batch,), int(cache_offset), dtype=mx.int64)
        rope_deltas = self._rope_deltas
        if rope_deltas is not None:
            offsets = offsets + rope_deltas[:batch].reshape(-1).astype(mx.int64)
        position_ids = offsets[:, None] + mx.arange(length, dtype=mx.int64)[None]
        if self._position_ids is not None and self._position_ids.ndim == 3:
            position_ids = mx.broadcast_to(position_ids[None], (3, batch, length))
        output = _QWEN4_EXACT_SPECULATIVE_VERIFIER(
            self,
            inputs,
            cache=cache,
            position_ids=position_ids,
            skip_logits=sampler is None,
        )
        rollback_state = (snapshot, inputs)
        hidden = output.hidden_states[-1]
        if sampler is None:
            return hidden, {}, rollback_state
        return hidden, {}, rollback_state, sampler(output.logits)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        return self._speculative_verify(inputs, cache)

    def speculative_verify_logits(self, inputs: mx.array, cache, sampler):
        return self._speculative_verify(inputs, cache, sampler)

    def rollback_speculative_cache(
        self,
        caches,
        rollback_state,
        accepted,
        block_size: int,
    ) -> int:
        del block_size
        if isinstance(accepted, int):
            accepted_list = [accepted]
        elif isinstance(accepted, mx.array):
            accepted_list = [int(x) for x in accepted.reshape(-1).tolist()]
        else:
            accepted_list = [int(x) for x in accepted]
        if len(set(accepted_list)) != 1:
            raise ValueError(
                "Qwen4-Exp MTP batched rollback requires uniform acceptance."
            )

        snapshots, verify_inputs = rollback_state
        self._restore_speculative_cache(caches, snapshots)
        keep = accepted_list[0] + 1
        for index in range(keep):
            self(
                verify_inputs[:, index : index + 1],
                cache=caches,
                skip_logits=True,
            )
        return accepted_list[0]

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if "ple" in layer else 2))
            else:
                caches.append(QSAKVCache())
        return caches

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if ".ple.ple_embedding.ngram_embedding.shards." in path:
                # Affine group-64 cannot represent 160-wide PLE rows. Preserve
                # compatible modes such as NVFP4/group-16 and otherwise fall
                # back to affine group-32.
                return {"fallback_group_size": 32}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate
