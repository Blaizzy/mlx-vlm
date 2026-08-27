import mlx.core as mx
import pytest

from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _build_codec,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    resolve_kv_bits,
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


def test_batch_turboquant_uses_value_shape_for_value_codec():
    keys = mx.random.normal((1, 1, 3, 32))
    values = mx.random.normal((1, 1, 3, 8))
    cache = BatchTurboQuantKVCache([0], bits=3.5)

    quantized_keys, quantized_values = cache.update_and_fetch(keys, values)
    dequantized_keys, dequantized_values = cache.dequantize(
        quantized_keys, quantized_values
    )
    mx.eval(dequantized_keys, dequantized_values)

    assert cache.key_codec.dim == keys.shape[-1]
    assert cache.value_codec.dim == values.shape[-1]
    assert dequantized_keys.shape == keys.shape
    assert dequantized_values.shape == values.shape


def test_batch_turboquant_cache_supports_uniform_right_trim():
    cache = BatchTurboQuantKVCache([0, 1], bits=3.5)
    keys = mx.ones((2, 2, 5, 8), dtype=mx.float16)
    values = mx.ones((2, 2, 5, 8), dtype=mx.float16)

    cache.update_and_fetch(keys, values)
    trimmed = cache.trim(2)

    assert trimmed == 2
    assert cache.is_trimmable()
    assert cache._idx == 3
    assert cache.offset.tolist() == [3, 2]
    state_keys, state_values, offset, left_padding = cache.state
    assert state_keys.norms.shape[2] == 3
    assert state_values.norms.shape[2] == 3
    assert offset.tolist() == [3, 2]
    assert left_padding.tolist() == [0, 1]


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


def test_batch_turboquant_make_mask_matches_batch_kv_cache_with_left_padding():
    left_padding = [2, 0]
    cache = BatchTurboQuantKVCache(left_padding, bits=3.5)
    reference = BatchKVCache(left_padding)
    keys = mx.random.normal((2, 2, 5, 64))
    values = mx.random.normal((2, 2, 5, 64))

    cache.update_and_fetch(keys, values)
    reference.update_and_fetch(keys, values)

    mask = cache.make_mask(2, return_array=True, window_size=None)
    reference_mask = reference.make_mask(2, return_array=True, window_size=None)

    assert mask.shape == reference_mask.shape
    assert mx.all(mask == reference_mask).item()


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


def _assert_mse_states_equal(left, right):
    assert bool(mx.all(left.norms == right.norms).item())
    assert bool(mx.all(left.indices == right.indices).item())


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_turboquant_mse_prefill_decode_matches_batch_quantized_state(dim):
    mx.random.seed(99)
    keys = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)
    values = mx.random.normal((1, 8, 17, dim)).astype(mx.float16)

    batch_cache = TurboQuantKVCache(bits=4.0)
    batch_cache.update_and_fetch(keys, values)

    split_cache = TurboQuantKVCache(bits=4.0)
    split_cache.update_and_fetch(keys[:, :, :16, :], values[:, :, :16, :])
    split_cache.update_and_fetch(keys[:, :, 16:17, :], values[:, :, 16:17, :])

    batch_keys, batch_values = batch_cache.state
    split_keys, split_values = split_cache.state
    _assert_mse_states_equal(batch_keys, split_keys)
    _assert_mse_states_equal(batch_values, split_values)


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


def test_resolve_kv_bits_defaults_split_fractional_budget():
    assert resolve_kv_bits(3.5) == (3.5, 3.0, 4.0)
    assert resolve_kv_bits(4) == (4.0, 4.0, 4.0)


def test_resolve_kv_bits_overrides_each_side_independently():
    assert resolve_kv_bits(3.5, 8, None) == (3.5, 8.0, 4.0)
    assert resolve_kv_bits(3.5, None, 2) == (3.5, 3.0, 2.0)
    assert resolve_kv_bits(3.5, 8, 3) == (3.5, 8.0, 3.0)


def test_resolve_kv_bits_validates_overrides():
    with pytest.raises(ValueError):
        resolve_kv_bits(4, 0.5, None)
    with pytest.raises(ValueError):
        resolve_kv_bits(4, None, 3.25)


def test_asymmetric_bits_build_matching_codecs():
    cache = TurboQuantKVCache(bits=4, key_bits=8, value_bits=3)
    keys = mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    assert int(cache.key_codec.bits) == 8
    assert int(cache.value_codec.bits) == 3


def test_asymmetric_attention_matches_dequantized_reference():
    cache = TurboQuantKVCache(bits=4, key_bits=8, value_bits=3)
    keys = mx.random.normal((1, 4, 128, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 128, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)

    queries = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
    scale = 256**-0.5
    out = cache.quantized_attention(queries, scale=scale, mask=None)

    deq_keys, deq_values = cache.dequantize()
    reference = scaled_dot_product_attention(
        queries, deq_keys, deq_values, cache=None, scale=scale, mask=None
    )
    error = mx.sqrt(
        mx.sum((out.astype(mx.float32) - reference.astype(mx.float32)) ** 2)
        / mx.sum(reference.astype(mx.float32) ** 2)
    ).item()
    assert error < 0.02


def test_meta_state_round_trips_asymmetric_bits():
    cache = TurboQuantKVCache(bits=3.5, key_bits=8, value_bits=3)
    restored = TurboQuantKVCache(bits=3.5)
    restored.meta_state = cache.meta_state
    assert (restored.key_bits, restored.value_bits) == (8.0, 3.0)


def test_meta_state_accepts_legacy_three_tuple():
    cache = TurboQuantKVCache(bits=3.5)
    cache.meta_state = ("0", "3.5", "1234")
    assert (cache.key_bits, cache.value_bits) == (3.0, 4.0)
    assert cache.seed == 1234


def test_batch_cache_propagates_asymmetric_bits_to_extracted_rows():
    cache = BatchTurboQuantKVCache([0], bits=4, key_bits=8, value_bits=3)
    keys = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    cache.update_and_fetch(keys, values)
    extracted = cache.extract(0)
    assert (extracted.key_bits, extracted.value_bits) == (8.0, 3.0)


def test_maybe_quantize_kv_cache_applies_key_value_overrides():
    prompt_cache = [KVCache(), KVCache(), KVCache()]
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
        kv_key_bits=8,
        kv_value_bits=3,
    )
    converted = [c for c in prompt_cache if isinstance(c, TurboQuantKVCache)]
    assert converted
    for entry in converted:
        assert (entry.key_bits, entry.value_bits) == (8.0, 3.0)


def test_maybe_quantize_kv_cache_defaults_unchanged():
    prompt_cache = [KVCache(), KVCache(), KVCache()]
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
    )
    converted = [c for c in prompt_cache if isinstance(c, TurboQuantKVCache)]
    assert converted
    for entry in converted:
        assert (entry.key_bits, entry.value_bits) == (3.0, 4.0)


def test_apc_namespace_separates_key_value_bit_splits():
    from mlx_vlm.apc import apc_disk_namespace

    base = dict(kv_bits=3.5, kv_group_size=64, kv_quant_scheme="turboquant")
    default = apc_disk_namespace("m", **base)
    k8v3 = apc_disk_namespace("m", **base, kv_key_bits=8, kv_value_bits=3)
    k3v8 = apc_disk_namespace("m", **base, kv_key_bits=3, kv_value_bits=8)

    assert default == apc_disk_namespace("m", **base)
    assert len({default, k8v3, k3v8}) == 3


def _turbo_quant_config(**overrides):
    config = {"bits": 3.5, "group_size": 64, "scheme": "turboquant"}
    config.update(overrides)
    return config


def test_apc_stream_warm_cache_honors_key_value_bits():
    from mlx_vlm.apc import _fill_stream_layer_cache

    keys = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    built = _fill_stream_layer_cache(
        keys,
        values,
        prefix_len=8,
        quantize=True,
        kv_quant_config=_turbo_quant_config(key_bits=8, value_bits=3),
    )
    assert (built.key_bits, built.value_bits) == (8.0, 3.0)


def test_apc_batch_warm_cache_honors_key_value_bits():
    from mlx_vlm.apc import _fill_batch_layer_cache

    keys = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16)
    built = _fill_batch_layer_cache(
        keys,
        values,
        [0],
        [8],
        quantize=True,
        kv_quant_config=_turbo_quant_config(key_bits=8, value_bits=3),
    )
    assert (built.key_bits, built.value_bits) == (8.0, 3.0)


def test_apc_empty_batch_cache_honors_key_value_bits():
    from mlx_vlm.apc import _empty_quant_batch_cache

    built = _empty_quant_batch_cache([0], _turbo_quant_config(key_bits=8, value_bits=3))
    assert (built.key_bits, built.value_bits) == (8.0, 3.0)


def test_apc_warm_cache_defaults_match_live_split():
    from mlx_vlm.apc import _empty_quant_batch_cache

    built = _empty_quant_batch_cache([0], _turbo_quant_config())
    assert (built.key_bits, built.value_bits) == (3.0, 4.0)


def test_kv_quant_policy_default_split_is_not_an_override():
    from mlx_vlm.kv_quant import from_legacy

    policy = from_legacy(3.5, "turboquant", 64)
    assert (policy.key.bits, policy.value.bits) == (3.0, 4.0)
    assert not policy.has_split_override
    assert policy.to_config() == {
        "bits": 3.5,
        "group_size": 64,
        "scheme": "turboquant",
    }


def test_kv_quant_policy_records_explicit_override():
    from mlx_vlm.kv_quant import from_legacy

    policy = from_legacy(3.5, "turboquant", 64, 8, 3)
    assert policy.has_split_override
    assert policy.to_config()["key_bits"] == 8.0
    assert policy.to_config()["value_bits"] == 3.0


def test_kv_quant_policy_round_trips_through_config():
    from mlx_vlm.kv_quant import from_config, from_legacy

    for args in [(3.5, "turboquant", 64, None, None), (3.5, "turboquant", 64, 8, 3)]:
        policy = from_legacy(*args)
        assert from_config(policy.to_config()) == policy


def test_kv_quant_policy_uniform_and_none():
    from mlx_vlm.kv_quant import from_legacy

    assert from_legacy(None) is None
    uniform = from_legacy(8, "uniform", 64)
    assert uniform.scheme == "uniform"
    assert not uniform.is_turboquant
    assert (uniform.key.bits, uniform.value.bits) == (8.0, 8.0)
    assert uniform.is_homogeneous


def test_kv_quant_policy_is_hashable():
    from mlx_vlm.kv_quant import from_legacy

    a = from_legacy(3.5, "turboquant", 64)
    b = from_legacy(3.5, "turboquant", 64)
    c = from_legacy(3.5, "turboquant", 64, 8, 3)
    assert len({a, b, c}) == 2


def test_kv_quant_fingerprint_matches_legacy_descriptor():
    from mlx_vlm.kv_quant import kv_quant_fingerprint

    assert kv_quant_fingerprint(3.5, 64, "turboquant", 0) == "kv3.5-64-turboquant-0"
    assert kv_quant_fingerprint(None, None, None, None) == "kvNone-None-None-None"
    assert (
        kv_quant_fingerprint(3.5, 64, "turboquant", 0, 8, 3)
        == "kv3.5-64-turboquant-0-k8-v3"
    )


def test_kv_quant_policy_supports_per_tensor_schemes():
    from mlx_vlm.kv_quant import from_legacy

    policy = from_legacy(
        8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant"
    )
    assert not policy.is_homogeneous
    assert (policy.key.scheme, policy.key.bits) == ("uniform", 8.0)
    assert (policy.value.scheme, policy.value.bits) == ("turboquant", 3.0)
    assert not policy.is_turboquant


def test_kv_quant_policy_scheme_property_rejects_heterogeneous():
    from mlx_vlm.kv_quant import from_legacy

    policy = from_legacy(
        8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant"
    )
    with pytest.raises(ValueError):
        policy.scheme


def test_kv_quant_policy_rejects_unknown_scheme():
    from mlx_vlm.kv_quant import from_legacy

    with pytest.raises(ValueError):
        from_legacy(8, "uniform", 64, kv_key_scheme="turbo3")


def test_kv_quant_policy_rejects_fractional_uniform_bits():
    from mlx_vlm.kv_quant import from_legacy

    with pytest.raises(ValueError):
        from_legacy(3.5, "turboquant", 64, kv_key_bits=3.5, kv_key_scheme="uniform")


def test_kv_quant_heterogeneous_round_trips_and_fingerprints_distinctly():
    from mlx_vlm.kv_quant import from_config, from_legacy

    hetero = from_legacy(
        8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant"
    )
    assert from_config(hetero.to_config()) == hetero

    homogeneous = from_legacy(8, "uniform", 64)
    assert homogeneous.fingerprint(0) == "kv8.0-64-uniform-0"
    assert hetero.fingerprint(0) != homogeneous.fingerprint(0)
    assert hetero.fingerprint(0).endswith("-ksuniform-vsturboquant")


def test_kv_quant_homogeneous_fingerprint_has_no_scheme_suffix():
    from mlx_vlm.kv_quant import from_legacy

    assert "-ks" not in from_legacy(3.5, "turboquant", 64).fingerprint(0)
    assert "-ks" not in from_legacy(3.5, "turboquant", 64, 8, 3).fingerprint(0)


def _hybrid_policy():
    from mlx_vlm.kv_quant import from_legacy

    return from_legacy(8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant")


def _uniform_roundtrip(tensor, bits, group_size=64):
    weights, scales, biases = mx.quantize(tensor, group_size=group_size, bits=bits)
    return mx.dequantize(weights, scales, biases, group_size=group_size, bits=bits)


def _turbo_roundtrip(tensor, bits, seed):
    codec = _build_codec(tensor, bits, mode="mse", seed=seed)
    return codec.dequantize(codec.quantize(tensor))


def _relative_error(a, b):
    return mx.sqrt(
        mx.sum((a.astype(mx.float32) - b.astype(mx.float32)) ** 2)
        / mx.sum(b.astype(mx.float32) ** 2)
    ).item()


def test_hybrid_cache_quantizes_each_tensor_with_its_own_scheme():
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    keys = mx.random.normal((1, 4, 64, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 64, 256)).astype(mx.bfloat16)
    deq_keys, deq_values = cache.update_and_fetch(keys, values)

    assert cache.offset == 64
    assert _relative_error(deq_keys, _uniform_roundtrip(keys, 8)) < 1e-6
    assert (
        _relative_error(deq_values, _turbo_roundtrip(values, 3, cache.seed + 1)) < 1e-6
    )


def test_hybrid_cache_appends_across_decode_steps():
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    cache.update_and_fetch(
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
    )
    for _ in range(3):
        deq_keys, deq_values = cache.update_and_fetch(
            mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16),
            mx.random.normal((1, 4, 1, 256)).astype(mx.bfloat16),
        )
    assert cache.offset == 11
    assert deq_keys.shape == (1, 4, 11, 256)
    assert deq_values.shape == (1, 4, 11, 256)


def test_hybrid_cache_uses_the_plain_attention_path():
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    deq_keys, deq_values = cache.update_and_fetch(
        mx.random.normal((1, 4, 16, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 16, 256)).astype(mx.bfloat16),
    )
    assert not hasattr(cache, "bits")
    queries = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
    out = scaled_dot_product_attention(
        queries, deq_keys, deq_values, cache=cache, scale=256**-0.5, mask=None
    )
    assert out.shape == (1, 16, 1, 256)


def test_hybrid_cache_trims_both_tensors():
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    cache.update_and_fetch(
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
    )
    assert cache.trim(5) == 5
    assert cache.offset == 15
    deq_keys, deq_values = cache.dequantize()
    assert deq_keys.shape[-2] == 15
    assert deq_values.shape[-2] == 15


def test_hybrid_cache_meta_state_round_trips_policy():
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(_hybrid_policy())
    cache.offset = 12
    restored = HybridQuantKVCache(from_legacy(4, "uniform", 64))
    restored.meta_state = cache.meta_state
    assert restored.policy == cache.policy
    assert restored.offset == 12
    assert restored.seed == cache.seed


def test_hybrid_cache_supports_turboquant_key_uniform_value():
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache

    cache = HybridQuantKVCache(
        from_legacy(
            8,
            "turboquant",
            64,
            kv_key_bits=3,
            kv_value_bits=8,
            kv_value_scheme="uniform",
        )
    )
    keys = mx.random.normal((1, 4, 32, 256)).astype(mx.bfloat16)
    values = mx.random.normal((1, 4, 32, 256)).astype(mx.bfloat16)
    deq_keys, deq_values = cache.update_and_fetch(keys, values)
    assert _relative_error(deq_keys, _turbo_roundtrip(keys, 3, cache.seed)) < 1e-6
    assert _relative_error(deq_values, _uniform_roundtrip(values, 8)) < 1e-6


def test_maybe_quantize_kv_cache_builds_hybrid_for_mixed_schemes():
    from mlx_vlm.turboquant import HybridQuantKVCache

    prompt_cache = [KVCache(), KVCache(), KVCache()]
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=8,
        kv_quant_scheme="uniform",
        kv_value_bits=3,
        kv_value_scheme="turboquant",
    )
    built = [c for c in prompt_cache if isinstance(c, HybridQuantKVCache)]
    assert built
    for entry in built:
        assert entry.policy.key.scheme == "uniform"
        assert entry.policy.key.bits == 8.0
        assert entry.policy.value.scheme == "turboquant"
        assert entry.policy.value.bits == 3.0


def test_maybe_quantize_kv_cache_keeps_turboquant_for_matching_schemes():
    from mlx_vlm.turboquant import HybridQuantKVCache

    prompt_cache = [KVCache(), KVCache(), KVCache()]
    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=64,
        kv_bits=3.5,
        kv_quant_scheme="turboquant",
        kv_key_scheme="turboquant",
        kv_value_scheme="turboquant",
    )
    assert not any(isinstance(c, HybridQuantKVCache) for c in prompt_cache)
    assert any(isinstance(c, TurboQuantKVCache) for c in prompt_cache)


def test_batch_generator_accepts_scheme_overrides():
    import inspect

    from mlx_vlm.generate.ar import BatchGenerator, _make_cache

    for target in (BatchGenerator.__init__, _make_cache):
        params = inspect.signature(target).parameters
        assert "kv_key_scheme" in params
        assert "kv_value_scheme" in params


def test_make_cache_rejects_mixed_schemes():
    from mlx_vlm.generate.ar import _make_cache

    class _FakeModel:
        def make_cache(self):
            return [KVCache()]

    with pytest.raises(NotImplementedError, match="batch path"):
        _make_cache(
            _FakeModel(),
            [0],
            kv_bits=8,
            kv_quant_scheme="uniform",
            kv_value_bits=3,
            kv_value_scheme="turboquant",
        )


def test_make_cache_still_builds_homogeneous_batch_caches():
    from mlx_vlm.generate.ar import _make_cache

    class _FakeModel:
        def make_cache(self):
            return [KVCache()]

    caches = _make_cache(_FakeModel(), [0], kv_bits=3.5, kv_quant_scheme="turboquant")
    assert isinstance(caches[0], BatchTurboQuantKVCache)


def test_apc_batch_builders_reject_mixed_schemes():
    from mlx_vlm.apc import _empty_quant_batch_cache

    with pytest.raises(NotImplementedError, match="batch prefix caches"):
        _empty_quant_batch_cache(
            [0],
            {
                "bits": 8,
                "group_size": 64,
                "scheme": "uniform",
                "value_bits": 3,
                "value_scheme": "turboquant",
            },
        )


def test_apc_stream_builder_supports_mixed_schemes():
    from mlx_vlm.apc import _fill_stream_layer_cache
    from mlx_vlm.turboquant import HybridQuantKVCache

    built = _fill_stream_layer_cache(
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 8, 256)).astype(mx.bfloat16),
        prefix_len=8,
        quantize=True,
        kv_quant_config={
            "bits": 8,
            "group_size": 64,
            "scheme": "uniform",
            "value_bits": 3,
            "value_scheme": "turboquant",
        },
    )
    assert isinstance(built, HybridQuantKVCache)
    assert built.policy.key.scheme == "uniform"
    assert built.policy.value.scheme == "turboquant"
    assert built.offset == 8


def test_hybrid_cache_trims_fractional_turboquant_tensor():
    from mlx_vlm.kv_quant import from_legacy
    from mlx_vlm.turboquant import HybridQuantKVCache, _SplitCodec

    cache = HybridQuantKVCache(
        from_legacy(
            8,
            "uniform",
            64,
            kv_value_bits=3.5,
            kv_value_scheme="turboquant",
        )
    )
    cache.update_and_fetch(
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
        mx.random.normal((1, 4, 20, 256)).astype(mx.bfloat16),
    )
    assert isinstance(cache.value_quantizer.codec, _SplitCodec)
    assert cache.trim(5) == 5
    assert cache.offset == 15
    deq_keys, deq_values = cache.dequantize()
    assert deq_keys.shape[-2] == 15
    assert deq_values.shape[-2] == 15
