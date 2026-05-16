import mlx.core as mx
import pytest

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _build_codec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    turboquant_enabled,
)


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


def test_turboquant_mse_matches_paper_small_bit_distortions():
    vectors = _sample_unit_vectors(256, 64)
    expected = {1: 0.36, 2: 0.117, 3: 0.03}

    for bits, target in expected.items():
        codec = _TurboQuantMSECodec(64, bits, seed=0)
        state = codec.quantize(vectors)
        reconstructed = codec.dequantize(state)
        mse = mx.mean(mx.sum((vectors - reconstructed) ** 2, axis=-1)).item()
        assert mse == pytest.approx(target, rel=0.25, abs=0.02)


def test_turboquant_prod_is_nearly_unbiased_across_seeds():
    mx.random.seed(42)
    keys = _sample_unit_vectors(128, 64)
    queries = mx.random.normal((128, 64))
    true_inner_products = mx.sum(keys * queries, axis=-1)

    estimates = []
    for seed in range(16):
        codec = _TurboQuantProdCodec(64, 2, seed=seed)
        state = codec.quantize(keys)
        reconstructed = codec.dequantize(state)
        estimates.append(mx.sum(reconstructed * queries, axis=-1))

    mean_estimate = mx.mean(mx.stack(estimates), axis=0)
    bias = mx.mean(mean_estimate - true_inner_products).item()
    assert abs(bias) < 0.03


def test_fractional_turboquant_improves_reconstruction():
    vectors = mx.random.normal((1, 2, 32, 64))

    codec_3bit = _build_codec(vectors, 3.0, mode="mse", seed=0)
    codec_35bit = _build_codec(vectors, 3.5, mode="mse", seed=0)

    state_3bit = codec_3bit.quantize(vectors)
    state_35bit = codec_35bit.quantize(vectors)

    mse_3bit = mx.mean((vectors - codec_3bit.dequantize(state_3bit)) ** 2).item()
    mse_35bit = mx.mean((vectors - codec_35bit.dequantize(state_35bit)) ** 2).item()

    assert turboquant_enabled(3.5)
    assert not turboquant_enabled(3.0)
    assert mse_35bit < mse_3bit


def test_turboquant_cache_replaces_kv_cache_for_fractional_bits():
    layer_cache = KVCache()
    layer_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [layer_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="uniform",
    )

    assert isinstance(prompt_cache[0], TurboQuantKVCache)


def test_explicit_turboquant_scheme_supports_integer_bits():
    layer_cache = KVCache()
    layer_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [layer_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.0,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], TurboQuantKVCache)
    assert prompt_cache[0].bits == pytest.approx(3.0)


def test_turboquant_skips_non_kv_cache_entries():
    linear_cache = ArraysCache(size=2)
    linear_cache[0] = mx.zeros((1, 8))
    linear_cache[1] = mx.ones((1, 8))

    attention_cache = KVCache()
    attention_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)),
        mx.random.normal((1, 2, 8, 32)),
    )
    prompt_cache = [linear_cache, attention_cache]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=4,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
    )

    assert isinstance(prompt_cache[0], ArraysCache)
    assert isinstance(prompt_cache[1], TurboQuantKVCache)


def test_batch_turboquant_extend_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    first = BatchTurboQuantKVCache([0], bits=3.5)
    second = BatchTurboQuantKVCache([0], bits=3.5)

    first.update_and_fetch(keys, values)
    second.update_and_fetch(keys, values)
    first.extend(second)

    assert first.offset.tolist() == [3, 3]
    assert first.left_padding.tolist() == [0, 0]


def test_batch_turboquant_extend_supports_empty_uniform_offsets():
    first = BatchTurboQuantKVCache([0], bits=3.5)
    second = BatchTurboQuantKVCache([0], bits=3.5)

    first.extend(second)

    assert first.offset.tolist() == [0, 0]
    assert first.left_padding.tolist() == [0, 0]


def test_batch_turboquant_filter_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    cache = BatchTurboQuantKVCache([0], bits=3.5)

    cache.update_and_fetch(keys, values)
    cache.filter(mx.array([0]))

    assert cache.offset.tolist() == [3]
    assert cache.left_padding.tolist() == [0]


def test_batch_turboquant_extend_pads_shorter_uniform_batch():
    longer = BatchTurboQuantKVCache([0], bits=3.5)
    shorter = BatchTurboQuantKVCache([0], bits=3.5)

    longer.update_and_fetch(
        mx.ones((1, 2, 5, 8), dtype=mx.float16),
        mx.ones((1, 2, 5, 8), dtype=mx.float16),
    )
    shorter.update_and_fetch(
        mx.ones((1, 2, 3, 8), dtype=mx.float16),
        mx.ones((1, 2, 3, 8), dtype=mx.float16),
    )
    longer.extend(shorter)

    assert longer.offset.tolist() == [5, 3]
    assert longer.left_padding.tolist() == [0, 2]
    assert longer._idx == 5


def test_turboquant_cache_preserves_attention_shape_and_compresses_memory():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))
    queries = mx.random.normal((1, 2, 1, 32))

    fp_cache = KVCache()
    fp_keys, fp_values = fp_cache.update_and_fetch(keys, values)
    reference = scaled_dot_product_attention(
        queries,
        fp_keys,
        fp_values,
        fp_cache,
        scale=32**-0.5,
        mask=None,
    )

    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    diff = mx.mean(mx.abs(reference - quantized)).item()

    assert quantized.shape == reference.shape
    assert turbo_cache.nbytes < fp_cache.nbytes
    assert diff < 0.35


def test_turboquant_decode_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 16, 32))
    values = mx.random.normal((1, 2, 16, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys,
        turbo_values,
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask=None,
    )
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4


def test_turboquant_decode_attention_skips_full_dequantize():
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError("decode_attention should not call full dequantize")

    turbo_cache.dequantize = fail
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_metal_fast_path_skips_unpack(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    import mlx_vlm.turboquant as turboquant

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError("decode metal fast path should not unpack low-bit state")

    monkeypatch.setattr(turboquant, "_unpack_lowbit", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_4bit_uses_paper_prod_key_codec():
    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    # Keys now use MSE-only codec (QJL/Prod dropped for speed+quality)
    assert type(turbo_cache.key_codec).__name__ == "_TurboQuantMSECodec"
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_integer_separate_path_bypasses_fused(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError(
            "separate-kernel path should handle integer bits without fused fallback"
        )

    monkeypatch.setattr(turbo_cache, "_compiled_integer_decode_attention", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_decode_attention_separate_path_bypasses_fused_split(monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    keys = mx.random.normal((1, 2, 8, 32))
    values = mx.random.normal((1, 2, 8, 32))
    queries = mx.random.normal((1, 4, 1, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=4.0)
    turbo_keys, turbo_values = turbo_cache.state

    def fail(*args, **kwargs):
        raise AssertionError(
            "separate-kernel path should handle this without fused split fallback"
        )

    monkeypatch.setattr(turbo_cache, "_compiled_split_decode_attention", fail)
    output = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask=None,
    )

    assert output.shape == queries.shape


def test_turboquant_prod_quantize_skips_mse_dequantize(monkeypatch):
    codec = _TurboQuantProdCodec(32, 4, seed=0)
    vectors = mx.random.normal((1, 2, 8, 32))

    def fail(*args, **kwargs):
        raise AssertionError("Product quantization should not dequantize MSE state")

    monkeypatch.setattr(codec.mse_codec, "_dequantize_unit", fail)
    state = codec.quantize(vectors)

    assert state.mse_indices.shape[:3] == (1, 2, 8)


def test_turboquant_prefill_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 12, 32))
    values = mx.random.normal((1, 2, 12, 32))
    queries = mx.random.normal((1, 4, 4, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys,
        turbo_values,
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask="causal",
    )
    quantized = scaled_dot_product_attention(
        queries,
        turbo_keys,
        turbo_values,
        turbo_cache,
        scale=32**-0.5,
        mask="causal",
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4
