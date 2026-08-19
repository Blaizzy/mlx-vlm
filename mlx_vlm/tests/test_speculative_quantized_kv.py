"""Speculative decoding against a quantized KV cache.

Two things are required for a draft block to be verified and rolled back when
the cache is quantized:

1. ``BatchQuantizedKVCache`` must be trimmable, so rejected draft tokens can be
   dropped after verification.
2. The causal verify path in the attention block must slice a quantized cache
   entry, which is a ``(packed, scales, biases)`` tuple rather than one array.
"""

import mlx.core as mx

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache

B, H, D = 1, 2, 64


def _rand(steps, dim=D):
    return mx.random.normal((B, H, steps, dim))


def _quantized_cache(steps):
    cache = BatchQuantizedKVCache(left_padding=[0] * B)
    cache.update_and_fetch(_rand(steps), _rand(steps))
    return cache


def test_quantized_cache_is_trimmable():
    assert BatchQuantizedKVCache(left_padding=[0]).is_trimmable()


def test_trim_matches_unquantized_bookkeeping():
    """Trimming is index arithmetic and must behave the same either way."""
    quantized = _quantized_cache(10)
    dense = BatchKVCache(left_padding=[0] * B)
    dense.update_and_fetch(_rand(10), _rand(10))

    assert quantized.trim(4) == dense.trim(4)
    assert quantized._idx == dense._idx
    assert quantized.offset.tolist() == dense.offset.tolist()


def test_trim_never_exceeds_what_is_cached():
    cache = _quantized_cache(3)
    assert cache.trim(10) == 3
    assert cache._idx == 0


def test_appending_after_trim_yields_the_trimmed_length():
    """A rejected draft block is dropped, then the accepted token is appended."""
    cache = _quantized_cache(8)
    cache.trim(3)
    keys, _ = cache.update_and_fetch(_rand(1), _rand(1))
    assert keys[0].shape[-2] == 6


def test_kv_helpers_accept_arrays_and_quantized_tuples():
    from mlx_vlm.models.qwen3_5.language import _kv_prefix, _kv_seq_len

    dense = _rand(7)
    assert _kv_seq_len(dense) == 7
    assert _kv_prefix(dense, 4).shape[-2] == 4

    # mx.quantize hands back a list; the cache wraps it in a tuple. Both shapes
    # reach this code, so both have to work.
    packed = mx.quantize(dense, group_size=64, bits=8)
    for quantized in (list(packed), tuple(packed)):
        assert _kv_seq_len(quantized) == 7
        sliced = _kv_prefix(quantized, 4)
        assert all(component.shape[-2] == 4 for component in sliced)


def test_causal_verify_prefix_matches_dense_reference():
    """The per-position slices must line up with an unquantized reference.

    Quantization is lossy, so this checks agreement within tolerance rather
    than equality — an off-by-one in the slicing would move the result far
    outside it.
    """
    from mlx_vlm.models.qwen3_5.language import _kv_prefix

    verify_len, prefix_len = 3, 5
    total = prefix_len + verify_len
    keys, values = _rand(total), _rand(total)
    queries = _rand(verify_len)

    # Go through the cache rather than quantizing by hand: this is exactly what
    # the attention block receives during verification.
    quantized_cache = BatchQuantizedKVCache(left_padding=[0] * B)
    quantized_keys, quantized_values = quantized_cache.update_and_fetch(keys, values)

    for i in range(verify_len):
        length = prefix_len + i + 1
        got = scaled_dot_product_attention(
            queries[:, :, i : i + 1, :],
            _kv_prefix(quantized_keys, length),
            _kv_prefix(quantized_values, length),
            cache=quantized_cache,
            scale=D**-0.5,
            mask=None,
        )
        want = mx.fast.scaled_dot_product_attention(
            queries[:, :, i : i + 1, :],
            keys[:, :, :length, :],
            values[:, :, :length, :],
            scale=D**-0.5,
            mask=None,
        )
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=0.05).item()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
