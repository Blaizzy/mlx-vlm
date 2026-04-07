from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    RotatingKVCache,
    _BaseCache,
)


def _load_turboquant_cache_cls():
    from ..turboquant import TurboQuantKVCache

    return TurboQuantKVCache


def _normalize_turbo_bits(bits: float) -> float:
    value = float(bits)
    if value <= 0:
        raise ValueError("TurboQuant bits must be > 0.")
    return value


def _python_quantize(codec, vectors):
    # Force non-single-token path to avoid layout mismatch in fast kernel.
    if vectors.shape[-2] == 1:
        vec2 = mx.concatenate([vectors, vectors], axis=2)
        state2 = codec.quantize(vec2)

        def _slice(x):
            if x.ndim == 3:
                return x[:, :, 0:1]
            elif x.ndim == 4:
                return x[:, :, 0:1, :]
            elif x.ndim == 5:
                return x[:, :, 0:1, :, :]
            return x

        return tree_map(_slice, state2)
    return codec.quantize(vectors)


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


class TurboQuantRotatingKVCache(_BaseCache):
    """
    Rotating/SWA cache with TurboQuant for both keys and values.
    Maintains temporal order to avoid requantizing full history each step.
    """

    def __init__(self, max_size: int, keep: int = 0, bits: float = 4.0):
        self.max_size = max_size
        self.keep = keep
        self.bits = _normalize_turbo_bits(bits)
        self.offset = 0
        self._idx = 0
        self._key_state = None
        self._value_state = None
        self._turbo_cache = None

    @classmethod
    def from_rotating_cache(cls, cache: RotatingKVCache, bits: float):
        new_cache = cls(max_size=cache.max_size, keep=cache.keep, bits=bits)
        # Convert existing dense state if present.
        if cache.keys is not None and cache.values is not None:
            keys = cache.keys[:, :, : cache.offset, :]
            values = cache.values[:, :, : cache.offset, :]
            new_cache._ensure_codecs(keys, values)
            new_cache._key_state = _python_quantize(new_cache._turbo_cache.key_codec, keys)
            new_cache._value_state = _python_quantize(
                new_cache._turbo_cache.value_codec, values
            )
            new_cache._idx = tree_flatten(new_cache._key_state)[0][1].shape[2]
            new_cache.offset = cache.offset
        return new_cache

    def _ensure_codecs(self, keys: mx.array, values: mx.array):
        if self._turbo_cache is None:
            turbo_cache_cls = _load_turboquant_cache_cls()
            self._turbo_cache = turbo_cache_cls(bits=self.bits)
            self._turbo_cache._ensure_codecs(keys, values)

    def _slice_axis2(self, tree, start, end):
        def _slice(x):
            s = start if start is not None else 0
            e = end if end is not None else x.shape[2]
            if x.ndim == 3:
                return x[:, :, s:e]
            elif x.ndim == 4:
                return x[:, :, s:e, :]
            elif x.ndim == 5:
                return x[:, :, s:e, :, :]
            return x

        return tree_map(_slice, tree)

    def _concat_axis2(self, *trees):
        def _concat(*xs):
            return mx.concatenate(xs, axis=2)

        return tree_map(_concat, *trees)

    def _trim(self, trim_size, tree, append_tree=None):
        to_cat = []
        if trim_size > 0:
            to_cat.append(self._slice_axis2(tree, 0, self.keep))
            to_cat.append(self._slice_axis2(tree, trim_size + self.keep, None))
        else:
            to_cat.append(tree)
        if append_tree is not None:
            to_cat.append(append_tree)
        return self._concat_axis2(*to_cat)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._ensure_codecs(keys, values)
        new_key_state = _python_quantize(self._turbo_cache.key_codec, keys)
        new_val_state = _python_quantize(self._turbo_cache.value_codec, values)

        if self._key_state is None:
            self._key_state = new_key_state
            self._value_state = new_val_state
            self._idx = keys.shape[2]
        else:
            trim_size = self._idx - self.max_size + keys.shape[2]
            self._key_state = self._trim(trim_size, self._key_state, new_key_state)
            self._value_state = self._trim(trim_size, self._value_state, new_val_state)
            self._idx = tree_flatten(self._key_state)[0][1].shape[2]

        self.offset += keys.shape[2]
        dense_keys = self._turbo_cache.key_codec.dequantize(self._key_state).astype(keys.dtype)
        dense_values = self._turbo_cache.value_codec.dequantize(self._value_state).astype(
            values.dtype
        )
        return dense_keys, dense_values

    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if N > 1:
            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, self.offset)
            if offset + N > window_size or return_array:
                from mlx_lm.models.cache import create_causal_mask

                return create_causal_mask(N, offset, window_size=window_size)
            return "causal"
        if window_size is None:
            return None
        L = self._idx
        if self.offset >= window_size and self.max_size > window_size:
            return mx.arange(L) >= (L - window_size)
        return None

    def empty(self):
        return self._key_state is None

    @property
    def state(self):
        return self._key_state, self._value_state

    @state.setter
    def state(self, value):
        raise NotImplementedError("TurboQuantRotatingKVCache state restore NYI.")

    @property
    def nbytes(self):
        def _tree_nbytes(t):
            return sum(x.nbytes for _, x in tree_flatten(t)) if t is not None else 0

        return _tree_nbytes(self._key_state) + _tree_nbytes(self._value_state)


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
