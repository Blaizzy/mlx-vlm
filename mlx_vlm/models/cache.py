from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    _BaseCache,
)
from mlx_lm.models.cache import BatchKVCache as _UpstreamBatchKVCache
from mlx_lm.models.cache import BatchRotatingKVCache as _UpstreamBatchRotatingKVCache


# TODO(Prince):  rewrite each VLM's __call__ to use self._idx instead of cache.offset for Mrope Delta calculation
class _ScalarOffsetMixin:
    """Collapse ``offset`` to a Python int when uniform across the batch.

    Every VLM's decode ``__call__`` already has an ``isinstance(offset, int)``
    fast path (used by plain ``KVCache``); keeping the offset as ``int`` when
    it's effectively scalar makes that path win for batch caches too —
    avoiding a ``.item()`` GPU sync per decode step. ``left_padding`` stays
    as ``mx.array`` because it feeds into ``create_causal_mask`` etc.
    """

    def __init__(self, left_padding, *args, **kwargs):
        super().__init__(left_padding, *args, **kwargs)
        if left_padding and all(lp == left_padding[0] for lp in left_padding):
            self.offset = -int(left_padding[0])

    def _inflate(self):
        if isinstance(self.offset, int):
            self.offset = mx.array([self.offset])

    def prepare(self, *args, **kwargs):
        self._inflate()
        return super().prepare(*args, **kwargs)

    def filter(self, *args, **kwargs):
        self._inflate()
        return super().filter(*args, **kwargs)

    def extend(self, other, *args, **kwargs):
        self._inflate()
        if hasattr(other, "_inflate"):
            other._inflate()
        return super().extend(other, *args, **kwargs)


class BatchKVCache(_ScalarOffsetMixin, _UpstreamBatchKVCache):
    pass


class BatchRotatingKVCache(_ScalarOffsetMixin, _UpstreamBatchRotatingKVCache):
    pass


class BatchQuantizedKVCache(_BaseCache):
    """Batch-aware quantized KV cache for continuous batching.

    Mirrors ``BatchKVCache`` but stores keys/values in quantized form
    (packed uint32 + scales + biases), matching the format produced by
    ``mx.quantize``.  Supports ``extend()`` and ``filter()`` so that
    ``Batch.extend`` / ``Batch.filter`` work during continuous-batching.
    """

    step = 256

    def __init__(
        self,
        left_padding: List[int],
        group_size: int = 64,
        bits: int = 8,
    ):
        self.keys = None  # tuple (packed, scales, biases) or None
        self.values = None
        self.left_padding = mx.array(left_padding)
        # Scalar fast-path for offset when uniform across the batch.
        if left_padding and all(lp == left_padding[0] for lp in left_padding):
            self.offset = -int(left_padding[0])
        else:
            self.offset = mx.array([-lp for lp in left_padding])
        self._idx = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Quantize incoming keys/values and append to the cache.

        Args:
            keys:   float [B, H, S, D_k]
            values: float [B, H, S, D_v]

        Returns:
            Quantized (keys_tuple, values_tuple) sliced to ``_idx``.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx
        el_per_int = 8 * mx.uint32.size // self.bits

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def _init(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def _expand(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tuple(k[..., :prev, :] for k in self.keys)
                    self.values = tuple(v[..., :prev, :] for v in self.values)
                self.keys = tuple(_expand(k) for k in self.keys)
                self.values = tuple(_expand(v) for v in self.values)
            else:
                self.keys = _init(k_head_dim)
                self.values = _init(v_head_dim)

        self.offset += num_steps
        self._idx += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = q_keys[i]
            self.values[i][..., prev : self._idx, :] = q_values[i]

        return (
            tuple(k[..., : self._idx, :] for k in self.keys),
            tuple(v[..., : self._idx, :] for v in self.values),
        )

    def filter(self, batch_indices: mx.array):
        """Keep only the sequences at *batch_indices*."""
        if isinstance(self.offset, int):
            self.offset = mx.array([self.offset])
        if self.keys is not None:
            self.keys = tuple(k[batch_indices] for k in self.keys)
            self.values = tuple(v[batch_indices] for v in self.values)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_lp = self.left_padding.min().item()
        if min_lp > 0:
            if self.keys is not None:
                self.keys = tuple(k[..., min_lp:, :] for k in self.keys)
                self.values = tuple(v[..., min_lp:, :] for v in self.values)
            self._idx -= min_lp
            self.left_padding -= min_lp

    def extend(self, other: "BatchQuantizedKVCache"):
        """Concatenate *other* batch into this cache along the batch dim."""
        for c in (self, other):
            if isinstance(c.offset, int):
                c.offset = mx.array([c.offset])
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)

        def _pad_quant(cache_obj):
            if cache_obj.keys is None:
                return None
            left = max_idx - cache_obj._idx
            right_total = max(
                cache_obj.keys[0].shape[-2],
                max_idx,
            )
            right = right_total - cache_obj.keys[0].shape[-2] - left
            if right < 0:
                trimmed_keys = tuple(
                    k[..., : right_total - left, :] for k in cache_obj.keys
                )
                trimmed_values = tuple(
                    v[..., : right_total - left, :] for v in cache_obj.values
                )
                right = 0
            else:
                trimmed_keys = cache_obj.keys
                trimmed_values = cache_obj.values

            if left != 0 or right != 0:
                pad_spec = [(0, 0), (0, 0), (left, right), (0, 0)]
                padded_keys = tuple(mx.pad(k, pad_spec) for k in trimmed_keys)
                padded_values = tuple(mx.pad(v, pad_spec) for v in trimmed_values)
            else:
                padded_keys = trimmed_keys
                padded_values = trimmed_values

            lp = cache_obj.left_padding + left
            return padded_keys, padded_values, cache_obj.offset, lp

        r_self = _pad_quant(self)
        r_other = _pad_quant(other)

        if r_self is None and r_other is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
        if r_self is None:
            ok, ov, oo, olp = r_other
            slp = self.left_padding + max_idx
            self.keys = ok
            self.values = ov
            self.offset = mx.concatenate([self.offset, oo])
            self.left_padding = mx.concatenate([slp, olp])
            self._idx = max_idx
            return
        if r_other is None:
            sk, sv, so, slp = r_self
            olp = other.left_padding + max_idx
            self.keys = sk
            self.values = sv
            self.offset = mx.concatenate([so, other.offset])
            self.left_padding = mx.concatenate([slp, olp])
            self._idx = max_idx
            return

        sk, sv, so, slp = r_self
        ok, ov, oo, olp = r_other

        self.keys = tuple(mx.concatenate([s, o]) for s, o in zip(sk, ok))
        self.values = tuple(mx.concatenate([s, o]) for s, o in zip(sv, ov))
        self.offset = mx.concatenate([so, oo])
        self.left_padding = mx.concatenate([slp, olp])
        self._idx = max_idx

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.left_padding
        k = tuple(x[..., : self._idx, :] for x in self.keys)
        v = tuple(x[..., : self._idx, :] for x in self.values)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, val):
        self.keys, self.values, self.offset, self.left_padding = val
        if self.keys is not None:
            self._idx = self.keys[0].shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self._idx, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0

    def empty(self):
        return self.keys is None

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """
    Construct the model's cache for use in generation.

    This function will defer the cache construction to the model if it has a
    ``make_cache`` method, otherwise it will make a default KV cache.

    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)

    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]


class SimpleKVCache:
    """A simple key-value cache for transformer attention layers.

    Stores and concatenates key/value tensors along sequence dimension.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.cache_length = 0

    def update_and_fetch(self, keys, values):
        """Update cache with new key/value tensors and return full cache.

        Args:
            keys: New key tensor to add [batch, heads, seq_len, head_dim]
            values: New value tensor to add [batch, heads, seq_len, head_dim]

        Returns:
            Tuple of (cached_keys, cached_values) containing full cache history
        """
        if self.cache_length == 0:
            # First update - just store tensors
            self.keys = keys
            self.values = values
        else:
            # Concatenate with existing cache along sequence dimension
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)

        self.cache_length += keys.shape[2]
        return self.keys, self.values

    def fetch(self):
        return self.keys, self.values

    def update(self, keys, values):
        """Update cache with new key/value tensors without returning.

        Args:
            keys: New key tensor to store
            values: New value tensor to store
        """
        self.keys = keys
        self.values = values
        self.cache_length += keys.shape[2]


class SlidingWindowCache(_BaseCache):
    """A sliding window cache for local attention layers."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        if self.keys is None:
            # Initialize cache
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Simple sliding window: keep only the last max_size tokens
        if self.offset + seq_len <= self.max_size:
            # Fits within current window
            start_idx = self.offset
            end_idx = self.offset + seq_len
            self.keys[:, :, start_idx:end_idx, :] = keys
            self.values[:, :, start_idx:end_idx, :] = values
            self.offset += seq_len
        else:
            # Need to slide the window
            if seq_len < self.max_size:
                # Shift existing content left
                shift_amount = min(seq_len, self.max_size - 1)
                self.keys[:, :, :-shift_amount, :] = self.keys[:, :, shift_amount:, :]
                self.values[:, :, :-shift_amount, :] = self.values[
                    :, :, shift_amount:, :
                ]
                # Add new tokens at the end
                self.keys[:, :, -shift_amount:, :] = keys[:, :, -shift_amount:, :]
                self.values[:, :, -shift_amount:, :] = values[:, :, -shift_amount:, :]
            else:
                # New sequence is larger than cache, just keep the last max_size tokens
                self.keys = keys[:, :, -self.max_size :, :]
                self.values = values[:, :, -self.max_size :, :]
            self.offset = self.max_size

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.max_size

    def get_max_cache_shape(self):
        return self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0


class StaticKVCache(_BaseCache):
    """A static cache that grows to accommodate all tokens."""

    def __init__(self, max_size: int, step: int = 256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        B, n_kv_heads, seq_len, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]

        # Initialize cache if needed
        if self.keys is None:
            k_shape = (B, n_kv_heads, self.max_size, k_head_dim)
            v_shape = (B, n_kv_heads, self.max_size, v_head_dim)
            self.keys = mx.zeros(k_shape, dtype=keys.dtype)
            self.values = mx.zeros(v_shape, dtype=values.dtype)

        # Update cache
        end_pos = min(self.offset + seq_len, self.max_size)
        actual_seq_len = end_pos - self.offset

        if actual_seq_len > 0:
            self.keys[:, :, self.offset : end_pos, :] = keys[:, :, :actual_seq_len, :]
            self.values[:, :, self.offset : end_pos, :] = values[
                :, :, :actual_seq_len, :
            ]
            self.offset = end_pos

        return self.keys, self.values

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return self.keys, self.values

    @state.setter
    def state(self, v):
        if v is not None and len(v) == 2:
            self.keys, self.values = v
            if self.keys is not None:
                self.offset = self.max_size

    @property
    def meta_state(self):
        return tuple(map(str, (self.max_size, self.step, self.offset)))

    @meta_state.setter
    def meta_state(self, v):
        self.max_size, self.step, self.offset = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n
