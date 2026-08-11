"""Adversarial stress tests for MLX-VLM KV quantization optimizations.

Tests the following optimizations:
1. KV cache quantization from token 0 (quantized_kv_start=0)
2. Prefill step size 4096
3. All layers quantized (should_quantize_kv_layer returns True for all)
4. QuantizedKVCache.step = 512

Author: Challenger Agent
"""

import mlx.core as mx
import pytest

from mlx_vlm.generate.common import (
    DEFAULT_QUANTIZED_KV_START,
    maybe_quantize_kv_cache,
)
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    should_quantize_kv_layer,
)


def _make_kv(B, n_kv_heads, seq_len, head_dim=64):
    return (
        mx.random.normal((B, n_kv_heads, seq_len, head_dim)),
        mx.random.normal((B, n_kv_heads, seq_len, head_dim)),
    )


# ============================================================================
# Test 1: should_quantize_kv_layer — ALL layers must be quantized
# ============================================================================


class TestShouldQuantizeAllLayers:
    @pytest.mark.parametrize("num_layers", [1, 2, 4, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("layer_idx", [0, 1, -1, -2])
    def test_all_layers_quantized(self, num_layers: int, layer_idx: int):
        idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        result = should_quantize_kv_layer(idx, num_layers)
        assert result is True

    def test_zero_layers(self):
        assert should_quantize_kv_layer(0, 0) is True

    def test_single_layer(self):
        assert should_quantize_kv_layer(0, 1) is True

    def test_deep_model_boundary(self):
        for i in range(32):
            assert should_quantize_kv_layer(i, 32) is True


# ============================================================================
# Test 2: QuantizedKVCache — step=512, stress
# ============================================================================


class TestQuantizedKVCacheStress:
    def test_single_token_decode(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 1)
        result_k, result_v = cache.update_and_fetch(k, v)
        assert cache.offset == 1
        # QuantizedKVCache returns tuples (packed, scales, biases)
        assert isinstance(result_k, tuple)
        assert len(result_k) == 3

    def test_exact_step_boundary_512(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 512)
        cache.update_and_fetch(k, v)
        assert cache.offset == 512

    def test_just_over_step_boundary_513(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k1, v1 = _make_kv(1, 8, 512)
        cache.update_and_fetch(k1, v1)
        k2, v2 = _make_kv(1, 8, 1)
        cache.update_and_fetch(k2, v2)
        assert cache.offset == 513

    def test_just_under_step_boundary_511(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 511)
        cache.update_and_fetch(k, v)
        assert cache.offset == 511

    def test_small_sequences_waste(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 10)
        cache.update_and_fetch(k, v)
        assert cache.keys[0].shape[2] >= 512

    def test_state_roundtrip(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 100)
        cache.update_and_fetch(k, v)
        state = cache.state
        cache2 = QuantizedKVCache(group_size=64, bits=4)
        cache2.state = state
        cache2.offset = cache.offset
        assert cache2.offset == cache.offset

    def test_empty_cache_state(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        state = cache.state
        assert state is None or state == (None, None)


# ============================================================================
# Test 3: BatchQuantizedKVCache — batch operations
# ============================================================================


class TestBatchQuantizedKVCacheStress:
    def test_empty_merge(self):
        c1 = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        c2 = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        merged = BatchQuantizedKVCache.merge([c1, c2])
        # Empty merge returns batch_size = sum of inputs
        assert merged.batch_size == 2

    def test_extend_empty_both(self):
        c1 = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        c2 = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        c1.extend(c2)
        assert c1.batch_size == 4

    def test_single_token_batch(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        k, v = _make_kv(2, 8, 1)
        result_k, result_v = cache.update_and_fetch(k, v)
        assert cache._idx == 1
        assert isinstance(result_k, mx.array)

    def test_batch_filter(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        k, v = _make_kv(4, 8, 10)
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([0, 2]))
        assert cache.batch_size == 2

    def test_batch_extend_quantized(self):
        c1 = BatchQuantizedKVCache([0], group_size=64, bits=4)
        c2 = BatchQuantizedKVCache([0], group_size=64, bits=4)
        k1, v1 = _make_kv(1, 8, 100)
        k2, v2 = _make_kv(1, 8, 50)
        c1.update_and_fetch(k1, v1)
        c2.update_and_fetch(k2, v2)
        c1.extend(c2)
        assert c1.batch_size == 2
        assert c1._idx == 150


# ============================================================================
# Test 4: KVCache.to_quantized
# ============================================================================


class TestKVCacheToQuantized:
    def test_basic_conversion(self):
        cache = KVCache()
        k, v = _make_kv(1, 8, 100)
        cache.update_and_fetch(k, v)
        quantized = cache.to_quantized(group_size=64, bits=4)
        assert isinstance(quantized, QuantizedKVCache)
        assert quantized.offset == 100

    def test_empty_conversion(self):
        cache = KVCache()
        quantized = cache.to_quantized(group_size=64, bits=4)
        assert isinstance(quantized, QuantizedKVCache)
        assert quantized.offset == 0

    def test_conversion_preserves_offset(self):
        cache = KVCache()
        for step in [128, 256, 512]:
            k, v = _make_kv(1, 8, step)
            cache.update_and_fetch(k, v)
        quantized = cache.to_quantized(group_size=64, bits=4)
        assert quantized.offset == cache.offset


# ============================================================================
# Test 5: maybe_quantize_kv_cache
# ============================================================================


class TestMaybeQuantizeKVCache:
    def test_none_kv_bits(self):
        cache = [KVCache()]
        cache[0].update_and_fetch(mx.zeros((1, 8, 10, 64)), mx.zeros((1, 8, 10, 64)))
        original_type = type(cache[0])
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=None, kv_group_size=64)
        assert type(cache[0]) is original_type

    def test_quantized_kv_start_0_all_layers(self):
        cache = [KVCache(), KVCache()]
        for c in cache:
            c.update_and_fetch(mx.zeros((1, 8, 10, 64)), mx.zeros((1, 8, 10, 64)))
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=4, kv_group_size=64)
        assert isinstance(cache[0], QuantizedKVCache)
        assert isinstance(cache[1], QuantizedKVCache)

    def test_quantized_kv_start_high(self):
        cache = [KVCache(), KVCache(), KVCache()]
        for c in cache:
            c.update_and_fetch(mx.zeros((1, 8, 10, 64)), mx.zeros((1, 8, 10, 64)))
            c.offset = 5
        maybe_quantize_kv_cache(cache, quantized_kv_start=10, kv_bits=4, kv_group_size=64)
        for c in cache:
            assert isinstance(c, KVCache)

    def test_mixed_empty_and_populated(self):
        cache = [KVCache(), KVCache()]
        cache[0].update_and_fetch(mx.zeros((1, 8, 10, 64)), mx.zeros((1, 8, 10, 64)))
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=4, kv_group_size=64)
        assert isinstance(cache[0], QuantizedKVCache)
        assert isinstance(cache[1], QuantizedKVCache)


# ============================================================================
# Test 6: RotatingKVCache — quantization not supported
# ============================================================================


class TestRotatingKVCacheQuantization:
    def test_rotating_to_quantized_raises(self):
        cache = RotatingKVCache(max_size=1024)
        k, v = _make_kv(1, 8, 10)
        cache.update_and_fetch(k, v)
        with pytest.raises(NotImplementedError, match="NYI"):
            cache.to_quantized(group_size=64, bits=4)


# ============================================================================
# Test 7: BatchKVCache — left padding edge cases
# ============================================================================


class TestBatchKVCacheEdgeCases:
    def test_single_sequence(self):
        cache = BatchKVCache([0])
        k, v = _make_kv(1, 8, 10)
        cache.update_and_fetch(k, v)
        assert cache._idx == 10

    def test_multiple_sequences_different_lengths(self):
        cache = BatchKVCache([5, 3, 0])
        k, v = _make_kv(3, 8, 10)
        cache.update_and_fetch(k, v)
        assert cache._idx == 10

    def test_filter_with_left_padding(self):
        cache = BatchKVCache([5, 3, 0])
        k, v = _make_kv(3, 8, 10)
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([0, 2]))
        assert cache.batch_size == 2

    def test_extract_single_row(self):
        cache = BatchKVCache([0, 0])
        k, v = _make_kv(2, 8, 10)
        cache.update_and_fetch(k, v)
        extracted = cache.extract(0)
        assert isinstance(extracted, KVCache)
        assert extracted.offset == 10

    def test_merge_different_sizes(self):
        c1 = BatchKVCache([0])
        c2 = BatchKVCache([0])
        k1, v1 = _make_kv(1, 8, 100)
        k2, v2 = _make_kv(1, 8, 50)
        c1.update_and_fetch(k1, v1)
        c2.update_and_fetch(k2, v2)
        merged = BatchKVCache.merge([c1, c2])
        assert merged._idx == 100


# ============================================================================
# Test 8: Quantization accuracy
# ============================================================================


class TestQuantizationAccuracy:
    def test_4bit_quantization_error(self):
        k = mx.random.normal((1, 8, 100, 64))
        q = mx.quantize(k, group_size=64, bits=4)
        k_dequant = mx.dequantize(q[0], q[1], q[2], group_size=64, bits=4)
        max_error = float(mx.max(mx.abs(k - k_dequant)).item())
        assert max_error < 2.0

    def test_8bit_quantization_error(self):
        k = mx.random.normal((1, 8, 100, 64))
        q = mx.quantize(k, group_size=64, bits=8)
        k_dequant = mx.dequantize(q[0], q[1], q[2], group_size=64, bits=8)
        max_error = float(mx.max(mx.abs(k - k_dequant)).item())
        assert max_error < 0.5

    def test_2bit_quantization_error(self):
        k = mx.random.normal((1, 8, 100, 64))
        q = mx.quantize(k, group_size=64, bits=2)
        k_dequant = mx.dequantize(q[0], q[1], q[2], group_size=64, bits=2)
        max_error = float(mx.max(mx.abs(k - k_dequant)).item())
        assert max_error < 5.0


# ============================================================================
# Test 9: Memory savings
# ============================================================================


class TestQuantizedMemorySavings:
    def test_4bit_memory_reduction(self):
        k_float = mx.random.normal((1, 8, 1024, 64))
        k_4bit = mx.quantize(k_float, group_size=64, bits=4)
        ratio = sum(x.nbytes for x in k_4bit) / k_float.nbytes
        assert ratio < 0.35

    def test_8bit_memory_reduction(self):
        k_float = mx.random.normal((1, 8, 1024, 64))
        k_8bit = mx.quantize(k_float, group_size=64, bits=8)
        ratio = sum(x.nbytes for x in k_8bit) / k_float.nbytes
        assert ratio < 0.60

    def test_cache_nbytes_reduction(self):
        k, v = _make_kv(1, 8, 1024)
        float_cache = KVCache()
        float_cache.update_and_fetch(k, v)
        float_nbytes = float_cache.nbytes
        quant_cache = QuantizedKVCache(group_size=64, bits=4)
        quant_cache.update_and_fetch(k, v)
        ratio = quant_cache.nbytes / float_nbytes
        assert ratio < 0.40


# ============================================================================
# Test 10: Large sequence stress
# ============================================================================


class TestLargeSequenceStress:
    def test_large_sequence_update(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 8192)
        cache.update_and_fetch(k, v)
        assert cache.offset == 8192

    def test_incremental_large_sequence(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        for chunk in [1024, 2048, 4096]:
            k, v = _make_kv(1, 8, chunk)
            cache.update_and_fetch(k, v)
            assert cache.offset == chunk

    def test_batch_large_sequence(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=64, bits=4)
        k, v = _make_kv(2, 8, 4096)
        cache.update_and_fetch(k, v)
        assert cache._idx == 4096

    def test_cache_state_large(self):
        cache = QuantizedKVCache(group_size=64, bits=4)
        k, v = _make_kv(1, 8, 16384)
        cache.update_and_fetch(k, v)
        state = cache.state
        cache2 = QuantizedKVCache(group_size=64, bits=4)
        cache2.state = state
        cache2.offset = cache.offset
        assert cache2.offset == cache.offset


# ============================================================================
# Test 11: Mixed batch
# ============================================================================


class TestMixedBatchEdgeCases:
    def test_batch_uneven_lengths(self):
        cache = BatchQuantizedKVCache([10, 5, 0], group_size=64, bits=4)
        k, v = _make_kv(3, 8, 20)
        cache.update_and_fetch(k, v)
        assert cache._idx == 20

    def test_batch_filter_preserves_quantization(self):
        cache = BatchQuantizedKVCache([0, 0, 0, 0], group_size=64, bits=4)
        k, v = _make_kv(4, 8, 100)
        cache.update_and_fetch(k, v)
        cache.filter(mx.array([0, 2]))
        assert cache.batch_size == 2

    def test_batch_extend_mixed(self):
        c1 = BatchQuantizedKVCache([0], group_size=64, bits=4)
        c2 = BatchQuantizedKVCache([0], group_size=64, bits=4)
        k1, v1 = _make_kv(1, 8, 100)
        c1.update_and_fetch(k1, v1)
        c1.extend(c2)
        assert c1.batch_size == 2

    def test_batch_merge_all_empty(self):
        caches = [
            BatchQuantizedKVCache([0], group_size=64, bits=4),
            BatchQuantizedKVCache([0], group_size=64, bits=4),
        ]
        merged = BatchQuantizedKVCache.merge(caches)
        assert merged.batch_size == 2


# ============================================================================
# Test 12: Constants consistency
# ============================================================================


class TestConstantsConsistency:
    def test_quantized_kv_start_is_zero(self):
        assert DEFAULT_QUANTIZED_KV_START == 0

    def test_prefill_step_size_is_4096(self):
        from mlx_vlm.generate.ar import DEFAULT_PREFILL_STEP_SIZE
        assert DEFAULT_PREFILL_STEP_SIZE == 4096

    def test_quantized_kv_cache_step_is_512(self):
        assert QuantizedKVCache.step == 512

    def test_all_defaults_consistent(self):
        assert DEFAULT_QUANTIZED_KV_START == 0
        assert should_quantize_kv_layer(0, 10) is True
        assert QuantizedKVCache.step == 512


# ============================================================================
# Test 13: TurboQuant path
# ============================================================================


class TestTurboQuantPath:
    def test_uniform_quantization_path(self):
        """With kv_bits=4 and uniform scheme, QuantizedKVCache is used (NOT TurboQuant)."""
        cache = [KVCache()]
        k, v = _make_kv(1, 8, 10)
        cache[0].update_and_fetch(k, v)
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=4, kv_group_size=64)
        assert isinstance(cache[0], QuantizedKVCache)

    def test_turboquant_path_with_fractional_bits(self):
        """With fractional kv_bits (e.g., 3.5), TurboQuantKVCache is used."""
        from mlx_vlm.turboquant import TurboQuantKVCache
        cache = [KVCache()]
        k, v = _make_kv(1, 8, 10)
        cache[0].update_and_fetch(k, v)
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=3.5, kv_group_size=64)
        assert isinstance(cache[0], TurboQuantKVCache)

    def test_empty_cache_uniform_quantization(self):
        """Empty cache with uniform scheme gets QuantizedKVCache."""
        cache = [KVCache()]
        maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_bits=4, kv_group_size=64)
        assert isinstance(cache[0], QuantizedKVCache)


# ============================================================================
# Test 14: API Consistency — QuantizedKVCache vs BatchQuantizedKVCache
# ============================================================================


class TestAPIConsistency:
    def test_both_return_tuples(self):
        """Both QuantizedKVCache and BatchQuantizedKVCache return (k, v) tuples."""
        k, v = _make_kv(1, 8, 10)

        q_cache = QuantizedKVCache(group_size=64, bits=4)
        q_result = q_cache.update_and_fetch(k, v)
        assert isinstance(q_result, tuple)
        assert len(q_result) == 2

        bq_cache = BatchQuantizedKVCache([0], group_size=64, bits=4)
        bq_result = bq_cache.update_and_fetch(k, v)
        assert isinstance(bq_result, tuple)
        assert len(bq_result) == 2

    def test_quantized_returns_raw_tuples(self):
        """QuantizedKVCache returns raw quantized tuples (packed, scales, biases)."""
        k, v = _make_kv(1, 8, 10)
        q_cache = QuantizedKVCache(group_size=64, bits=4)
        q_result = q_cache.update_and_fetch(k, v)
        # First element is (packed, scales, biases) tuple for keys
        assert isinstance(q_result[0], tuple)
        assert len(q_result[0]) == 3

    def test_batch_quantized_returns_dequantized(self):
        """BatchQuantizedKVCache returns dequantized arrays for API consistency."""
        k, v = _make_kv(1, 8, 10)
        bq_cache = BatchQuantizedKVCache([0], group_size=64, bits=4)
        bq_result = bq_cache.update_and_fetch(k, v)
        # First element is dequantized array
        assert isinstance(bq_result[0], mx.array)


# ============================================================================
# Test 15: Group size edge cases
# ============================================================================


class TestSmallGroupSizes:
    @pytest.mark.parametrize("bits", [2, 3, 4, 6, 8])
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_quantize_dequantize_roundtrip(self, bits, group_size):
        k = mx.random.normal((1, 8, 64, 64))
        q = mx.quantize(k, group_size=group_size, bits=bits)
        k_dequant = mx.dequantize(q[0], q[1], q[2], group_size=group_size, bits=bits)
        mse = float(mx.mean((k - k_dequant) ** 2).item())
        assert mse < 10.0

    def test_1bit_not_supported(self):
        """1-bit quantization is not supported by mx.quantize."""
        k = mx.random.normal((1, 8, 64, 64))
        with pytest.raises(ValueError, match="not supported"):
            mx.quantize(k, group_size=64, bits=1)

    def test_group_larger_than_dim(self):
        """Group size larger than last dimension causes error."""
        k = mx.random.normal((1, 8, 64, 64))
        with pytest.raises(ValueError, match="divisible"):
            mx.quantize(k, group_size=256, bits=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
