import mlx.core as mx
import pytest

from mlx_vlm import affine4
from mlx_vlm.affine4 import Affine4Codec, Affine4KVCache, BatchAffine4KVCache
from mlx_vlm.generate.ar import _make_cache
from mlx_vlm.generate.common import maybe_quantize_kv_cache
from mlx_vlm.kv_quant import from_legacy
from mlx_vlm.models.cache import KVCache
from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache


def _reference(cache, queries, scale, mask=None):
    keys, values = cache.dequantize()
    return mx.fast.scaled_dot_product_attention(
        queries,
        keys.astype(queries.dtype),
        values.astype(queries.dtype),
        scale=scale,
        mask=mask,
    )


def _cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)))


def test_affine4_policy_requires_homogeneous_four_bits():
    policy = from_legacy(4, "affine4")

    assert policy.is_affine4
    assert policy.scheme == "affine4"
    assert policy.key.bits == policy.value.bits == 4
    with pytest.raises(ValueError, match="exactly 4 bits"):
        from_legacy(3.5, "affine4")
    with pytest.raises(ValueError, match="same scheme"):
        from_legacy(4, "affine4", kv_value_scheme="turboquant")


def test_affine4_codec_round_trip_and_layout():
    mx.random.seed(1)
    vectors = mx.random.normal((2, 3, 7, 64), dtype=mx.float16)
    codec = Affine4Codec(64, seed=9)

    state = codec.quantize(vectors)
    restored = codec.dequantize(state)
    mx.eval(state.norms, state.indices, restored)

    assert state.norms.shape == vectors.shape[:-1]
    assert state.indices.shape == (*vectors.shape[:-1], 8)
    assert state.norms.dtype == mx.float16
    assert state.indices.dtype == mx.uint32
    assert restored.shape == vectors.shape
    assert _cosine(vectors, restored).item() > 0.99


def test_fused_affine4_quantizer_matches_portable_codec():
    if not mx.metal.is_available():
        pytest.skip("Metal kernel unavailable")
    mx.random.seed(2)
    keys = mx.random.normal((1, 2, 1, 256), dtype=mx.bfloat16)
    values = mx.random.normal((1, 2, 1, 256), dtype=mx.bfloat16)
    cache = Affine4KVCache()
    cache._ensure_codecs(keys, values)

    fused_keys, fused_values = cache._try_fused_kv_quantize(keys, values)
    ref_keys = cache.key_codec.quantize(keys)
    ref_values = cache.value_codec.quantize(values)
    mx.eval(
        fused_keys.norms,
        fused_keys.indices,
        fused_values.norms,
        fused_values.indices,
        ref_keys.norms,
        ref_keys.indices,
        ref_values.norms,
        ref_values.indices,
    )

    assert mx.array_equal(fused_keys.norms, ref_keys.norms).item()
    assert mx.array_equal(fused_keys.indices, ref_keys.indices).item()
    assert mx.array_equal(fused_values.norms, ref_values.norms).item()
    assert mx.array_equal(fused_values.indices, ref_values.indices).item()


def test_single_cache_conversion_and_prefix_snapshot_round_trip(monkeypatch):
    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    mx.random.seed(3)
    keys = mx.random.normal((1, 2, 17, 64), dtype=mx.float16)
    values = mx.random.normal((1, 2, 17, 64), dtype=mx.float16)
    source = KVCache()
    source.update_and_fetch(keys, values)
    caches = [source, KVCache(), KVCache()]

    maybe_quantize_kv_cache(caches, 0, 64, 4, "affine4")
    converted = caches[0]
    snapshot = converted.prefix_cache_snapshot()
    restored = Affine4KVCache()
    restored.prefix_cache_restore(snapshot)
    dk, dv = restored.dequantize()
    rk, rv = converted.dequantize()
    mx.eval(dk, dv, rk, rv)

    assert isinstance(converted, Affine4KVCache)
    assert restored.offset == converted.offset
    assert mx.array_equal(restored.keys.indices, converted.keys.indices).item()
    assert mx.array_equal(restored.values.indices, converted.values.indices).item()
    assert mx.allclose(dk, rk).item()
    assert mx.allclose(dv, rv).item()


def test_cache_conversion_does_not_confuse_affine4_with_turboquant():
    from mlx_vlm.turboquant import TurboQuantKVCache

    keys = mx.zeros((1, 2, 8, 64), dtype=mx.float16)
    values = mx.zeros((1, 2, 8, 64), dtype=mx.float16)
    affine_cache = Affine4KVCache()
    affine_cache.update_and_fetch(keys, values)
    caches = [affine_cache, KVCache(), KVCache()]

    maybe_quantize_kv_cache(caches, 0, 64, 4, "turboquant")
    assert type(caches[0]) is TurboQuantKVCache

    maybe_quantize_kv_cache(caches, 0, 64, 4, "affine4")
    assert type(caches[0]) is Affine4KVCache


def test_batch_factory_and_extract_preserve_affine4_type():
    class Model:
        def make_cache(self):
            return [KVCache(), KVCache(), KVCache()]

    caches = _make_cache(Model(), [0, 0], kv_bits=4, kv_quant_scheme="affine4")
    cache = caches[0]
    keys = mx.zeros((2, 2, 5, 64), dtype=mx.float16)
    values = mx.zeros((2, 2, 5, 64), dtype=mx.float16)
    cache.update_and_fetch(keys, values)

    extracted = cache.extract(1)

    assert isinstance(cache, BatchAffine4KVCache)
    assert isinstance(extracted, Affine4KVCache)
    assert extracted.offset == 5


@pytest.mark.parametrize(
    "cache_type,batch_type",
    [
        (TurboQuantKVCache, BatchTurboQuantKVCache),
        (Affine4KVCache, BatchAffine4KVCache),
    ],
)
def test_prefix_cache_merge_preserves_packed_cache_type(cache_type, batch_type):
    mx.random.seed(10)
    rows = []
    for length in (5, 3):
        cache = cache_type(bits=4)
        cache.update_and_fetch(
            mx.random.normal((1, 2, length, 64), dtype=mx.float16),
            mx.random.normal((1, 2, length, 64), dtype=mx.float16),
        )
        rows.append(cache)

    merged = rows[0].prefix_cache_merge(rows, [5, 3])

    assert type(merged) is batch_type
    for index, row in enumerate(rows):
        extracted = merged.extract(index)
        assert type(extracted) is cache_type
        actual = extracted.dequantize()
        expected = row.dequantize()
        mx.eval(*actual, *expected)
        assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))


def test_prefix_cache_merge_rejects_different_packed_scheme():
    keys = mx.zeros((1, 2, 3, 64), dtype=mx.float16)
    values = mx.zeros((1, 2, 3, 64), dtype=mx.float16)
    turbo = TurboQuantKVCache(bits=4)
    affine = Affine4KVCache()
    turbo.update_and_fetch(keys, values)
    affine.update_and_fetch(keys, values)

    assert turbo.prefix_cache_merge([turbo, affine], [3, 3]) is None
    assert affine.prefix_cache_merge([affine, turbo], [3, 3]) is None


@pytest.mark.parametrize(
    "query_heads,kv_heads,dim",
    [
        (4, 1, 32),
        (8, 2, 64),
        (12, 2, 96),
        (8, 2, 192),
        (16, 2, 256),
        (24, 4, 256),
        (8, 2, 512),
    ],
)
def test_native_affine4_decode_matches_dequantized_sdpa(query_heads, kv_heads, dim):
    if not affine4._m5_mpp_available():
        pytest.skip("M5 TensorOps unavailable")
    mx.random.seed(query_heads)
    tokens = 257
    keys = mx.random.normal((1, kv_heads, tokens, dim), dtype=mx.float16)
    values = mx.random.normal((1, kv_heads, tokens, dim), dtype=mx.float16)
    queries = mx.random.normal((1, query_heads, 1, dim), dtype=mx.bfloat16)
    cache = Affine4KVCache()
    cache.update_and_fetch(keys, values)

    output = cache.decode_attention(queries, scale=dim**-0.5)
    reference = _reference(cache, queries, dim**-0.5)
    mx.eval(output, reference)

    signature = (
        dim,
        query_heads // kv_heads,
        1,
        8,
        affine4._attention_warps(dim),
        kv_heads,
    )
    assert affine4._NATIVE_LAUNCHABLE[signature]
    assert _cosine(output, reference).item() > 0.999
    assert mx.max(mx.abs(output - reference)).item() < 0.01


@pytest.mark.parametrize("query_length", [2, 3, 4])
def test_native_affine4_causal_multirow(query_length):
    if not affine4._m5_mpp_available():
        pytest.skip("M5 TensorOps unavailable")
    mx.random.seed(query_length)
    tokens, dim = 257, 256
    keys = mx.random.normal((1, 4, tokens, dim), dtype=mx.float16)
    values = mx.random.normal((1, 4, tokens, dim), dtype=mx.float16)
    queries = mx.random.normal((1, 24, query_length, dim), dtype=mx.bfloat16)
    cache = Affine4KVCache()
    key_state, value_state = cache.update_and_fetch(keys, values)

    output = cache.prefill_attention(
        queries,
        key_state,
        value_state,
        scale=dim**-0.5,
        mask="causal",
    )
    q_positions = mx.arange(tokens - query_length, tokens)
    mask = mx.arange(tokens)[None, :] <= q_positions[:, None]
    reference = _reference(cache, queries, dim**-0.5, mask)
    mx.eval(output, reference)

    assert _cosine(output, reference).item() > 0.999
    assert mx.max(mx.abs(output - reference)).item() < 0.01


def test_native_failure_is_cached_and_uses_portable_fallback(monkeypatch):
    mx.random.seed(4)
    keys = mx.random.normal((1, 2, 256, 64), dtype=mx.float16)
    values = mx.random.normal((1, 2, 256, 64), dtype=mx.float16)
    queries = mx.random.normal((1, 8, 1, 64), dtype=mx.float16)
    cache = Affine4KVCache()
    cache.update_and_fetch(keys, values)
    calls = 0

    def rejected(*args, **kwargs):
        nonlocal calls
        calls += 1

        def launch(**kwargs):
            raise RuntimeError("rejected")

        return launch

    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: True)
    monkeypatch.setattr(affine4, "_mpp_attention_kernel", rejected)
    monkeypatch.setattr(affine4, "_NATIVE_LAUNCHABLE", {})

    first = cache.decode_attention(queries, scale=64**-0.5)
    second = cache.decode_attention(queries, scale=64**-0.5)
    reference = _reference(cache, queries, 64**-0.5)
    mx.eval(first, second, reference)

    assert calls == 1
    assert mx.allclose(first, reference, rtol=1e-3, atol=1e-3).item()
    assert mx.allclose(second, reference, rtol=1e-3, atol=1e-3).item()


def test_arbitrary_mask_declines_native_path(monkeypatch):
    mx.random.seed(5)
    keys = mx.random.normal((1, 2, 256, 64), dtype=mx.float16)
    values = mx.random.normal((1, 2, 256, 64), dtype=mx.float16)
    queries = mx.random.normal((1, 8, 1, 64), dtype=mx.float16)
    cache = Affine4KVCache()
    cache.update_and_fetch(keys, values)
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native kernel must not receive an arbitrary mask")

    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: True)
    monkeypatch.setattr(affine4, "_mpp_attention_kernel", unexpected)
    mask = mx.ones((1, 1, 1, 256), dtype=mx.bool_)

    output = cache.decode_attention(queries, scale=64**-0.5, mask=mask)
    reference = _reference(cache, queries, 64**-0.5, mask)
    mx.eval(output, reference)

    assert not called
    assert mx.allclose(output, reference, rtol=1e-3, atol=1e-3).item()


def test_left_padded_batch_marker_has_portable_fallback(monkeypatch):
    from mlx_vlm.models.cache import create_causal_mask

    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    mx.random.seed(6)
    batch, query_heads, kv_heads, prefix, dim = 2, 8, 2, 256, 64
    cache = BatchAffine4KVCache([17, 0])
    cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
    )
    mask = cache.make_mask(1)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, 1, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, 1, dim), dtype=mx.float16),
    )
    queries = mx.random.normal((batch, query_heads, 1, dim), dtype=mx.float16)

    output = cache.decode_attention(queries, keys, values, scale=dim**-0.5, mask=mask)
    dense_mask = create_causal_mask(1, offset=prefix, left_padding=cache.left_padding)
    reference = _reference(cache, queries, dim**-0.5, dense_mask)
    mx.eval(output, reference)

    assert mask == "left_padded_decode"
    assert _cosine(output, reference).item() > 0.999


def test_apc_factories_preserve_affine4_cache_types():
    from mlx_vlm import apc
    from mlx_vlm.apc_adapters import Capability, resolve_capability

    config = {"bits": 4, "group_size": 64, "scheme": "affine4"}
    keys = mx.zeros((1, 2, 8, 64), dtype=mx.float16)
    values = mx.zeros((1, 2, 8, 64), dtype=mx.float16)

    stream = apc._fill_stream_layer_cache(
        keys,
        values,
        8,
        quantize=True,
        kv_quant_config=config,
    )
    batch = apc._empty_quant_batch_cache([0], config)

    assert isinstance(stream, Affine4KVCache)
    assert isinstance(batch, BatchAffine4KVCache)
    assert resolve_capability(stream) is Capability.PAGEABLE
    assert resolve_capability(batch) is Capability.PAGEABLE


def test_native_left_padded_batch_matches_dequantized_sdpa():
    from mlx_vlm.models.cache import create_causal_mask

    if not affine4._m5_mpp_available():
        pytest.skip("M5 TensorOps unavailable")
    mx.random.seed(7)
    batch, query_heads, kv_heads, prefix, dim = 2, 8, 2, 256, 64
    cache = BatchAffine4KVCache([13, 0])
    cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
    )
    mask = cache.make_mask(1)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, 1, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, 1, dim), dtype=mx.float16),
    )
    queries = mx.random.normal((batch, query_heads, 1, dim), dtype=mx.bfloat16)

    output = cache.decode_attention(queries, keys, values, scale=dim**-0.5, mask=mask)
    dense_mask = create_causal_mask(1, offset=prefix, left_padding=cache.left_padding)
    reference = _reference(cache, queries, dim**-0.5, dense_mask)
    mx.eval(output, reference)

    signature = (64, 4, 1, 8, 1, 2)
    assert affine4._NATIVE_LAUNCHABLE[signature]
    assert _cosine(output, reference).item() > 0.999


@pytest.mark.parametrize("query_length", [2, 3, 4])
@pytest.mark.parametrize(
    "left_padding,expected_mask",
    [([13, 0], "left_padded_causal"), ([0, 0], "causal")],
)
def test_native_batch_causal_multirow(query_length, left_padding, expected_mask):
    from mlx_vlm.models.cache import create_causal_mask

    if not affine4._m5_mpp_available():
        pytest.skip("M5 TensorOps unavailable")
    mx.random.seed(20 + query_length)
    batch, query_heads, kv_heads, prefix, dim = 2, 8, 2, 256, 64
    cache = BatchAffine4KVCache(left_padding)
    cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
    )
    mask = cache.make_mask(query_length)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, query_length, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, query_length, dim), dtype=mx.float16),
    )
    queries = mx.random.normal(
        (batch, query_heads, query_length, dim), dtype=mx.bfloat16
    )

    output = cache.prefill_attention(queries, keys, values, scale=dim**-0.5, mask=mask)
    dense_mask = create_causal_mask(
        query_length, offset=prefix, left_padding=cache.left_padding
    )
    reference = _reference(cache, queries, dim**-0.5, dense_mask)
    mx.eval(output, reference)

    signature = (
        dim,
        query_heads // kv_heads,
        query_length,
        ((query_heads // kv_heads) * query_length + 7) // 8 * 8,
        1,
        kv_heads,
    )
    assert mask == expected_mask
    assert affine4._NATIVE_LAUNCHABLE[signature]
    assert _cosine(output, reference).item() > 0.999
    assert mx.max(mx.abs(output - reference)).item() < 0.01


def test_left_padded_multirow_marker_has_portable_fallback(monkeypatch):
    from mlx_vlm.models.base import scaled_dot_product_attention
    from mlx_vlm.models.cache import create_causal_mask

    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: False)
    mx.random.seed(30)
    batch, query_heads, kv_heads, prefix, query_length, dim = 2, 8, 2, 17, 3, 64
    cache = BatchAffine4KVCache([5, 0])
    cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, prefix, dim), dtype=mx.float16),
    )
    mask = cache.make_mask(query_length)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, kv_heads, query_length, dim), dtype=mx.float16),
        mx.random.normal((batch, kv_heads, query_length, dim), dtype=mx.float16),
    )
    queries = mx.random.normal(
        (batch, query_heads, query_length, dim), dtype=mx.float16
    )

    output = scaled_dot_product_attention(queries, keys, values, cache, dim**-0.5, mask)
    dense_mask = create_causal_mask(
        query_length, offset=prefix, left_padding=cache.left_padding
    )
    reference = _reference(cache, queries, dim**-0.5, dense_mask)
    mx.eval(output, reference)

    assert mask == "left_padded_causal"
    assert _cosine(output, reference).item() > 0.999


def test_unsupported_head_dimension_uses_portable_fallback(monkeypatch):
    mx.random.seed(8)
    keys = mx.random.normal((1, 2, 256, 80), dtype=mx.float16)
    values = mx.random.normal((1, 2, 256, 80), dtype=mx.float16)
    queries = mx.random.normal((1, 8, 1, 80), dtype=mx.float16)
    cache = Affine4KVCache()
    cache.update_and_fetch(keys, values)

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported geometry reached native compilation")

    monkeypatch.setattr(affine4, "_m5_mpp_available", lambda: True)
    monkeypatch.setattr(affine4, "_mpp_attention_kernel", unexpected)

    output = cache.decode_attention(queries, scale=80**-0.5)
    reference = _reference(cache, queries, 80**-0.5)
    mx.eval(output, reference)

    assert mx.allclose(output, reference, rtol=1e-3, atol=1e-3).item()


@pytest.mark.parametrize("factory_failure", [False, True])
def test_fused_quantizer_failure_is_cached(monkeypatch, factory_failure):
    mx.random.seed(9)
    keys = mx.random.normal((1, 2, 1, 64), dtype=mx.float16)
    values = mx.random.normal((1, 2, 1, 64), dtype=mx.float16)
    calls = 0

    def rejected(dim):
        nonlocal calls
        calls += 1
        if factory_failure:
            raise RuntimeError("rejected")

        def launch(**kwargs):
            raise RuntimeError("rejected")

        return launch

    monkeypatch.setattr(affine4, "_fused_quantize_kernel", rejected)
    monkeypatch.setattr(affine4, "_FUSED_QUANTIZE_LAUNCHABLE", {})
    cache = Affine4KVCache()

    cache.update_and_fetch(keys, values)
    cache.update_and_fetch(keys, values)
    restored_keys, restored_values = cache.dequantize()
    mx.eval(restored_keys, restored_values)

    assert calls == 1
    assert cache.offset == 2
    assert restored_keys.shape == (1, 2, 2, 64)
    assert restored_values.shape == (1, 2, 2, 64)
