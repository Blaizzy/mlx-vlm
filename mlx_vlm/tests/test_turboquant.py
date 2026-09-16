"""TurboQuant cache integration, batched attention, and value kernels."""

import mlx.core as mx
import pytest

import mlx_vlm.turboquant as tq
from mlx_vlm.generate import maybe_quantize_kv_cache
from mlx_vlm.models.base import (
    _turboquant_attention_applies,
    scaled_dot_product_attention,
)
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _TurboQuantMSECodec,
    _TurboQuantProdCodec,
    resolve_kv_bits,
)

# Cache codecs and integration


def _sample_unit_vectors(count: int, dim: int) -> mx.array:
    vectors = mx.random.normal((count, dim))
    return vectors / mx.linalg.norm(vectors, axis=-1, keepdims=True)


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


def test_turboquant_skips_non_kv_cache_entries():
    linear_cache = ArraysCache(size=2)
    linear_cache[0] = mx.zeros((1, 8))
    linear_cache[1] = mx.ones((1, 8))

    attention_cache = KVCache()
    attention_cache.update_and_fetch(
        mx.random.normal((1, 2, 8, 32)), mx.random.normal((1, 2, 8, 32))
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


def test_batch_turboquant_filter_supports_uniform_single_item_offsets():
    keys = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    values = mx.ones((1, 2, 3, 8), dtype=mx.float16)
    cache = BatchTurboQuantKVCache([0], bits=3.5)

    cache.update_and_fetch(keys, values)
    cache.filter(mx.array([0]))

    assert cache.offset.tolist() == [3]
    assert cache.left_padding.tolist() == [0]


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
        mx.ones((1, 2, 5, 8), dtype=mx.float16), mx.ones((1, 2, 5, 8), dtype=mx.float16)
    )
    shorter.update_and_fetch(
        mx.ones((1, 2, 3, 8), dtype=mx.float16), mx.ones((1, 2, 3, 8), dtype=mx.float16)
    )
    longer.extend(shorter)

    assert longer.offset.tolist() == [5, 3]
    assert longer.left_padding.tolist() == [0, 2]
    assert longer._idx == 5


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
        queries, turbo_keys, turbo_values, turbo_cache, scale=32**-0.5, mask=None
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


def test_turboquant_prefill_attention_matches_dequantized_attention():
    keys = mx.random.normal((1, 2, 12, 32))
    values = mx.random.normal((1, 2, 12, 32))
    queries = mx.random.normal((1, 4, 4, 32))

    fp_cache = KVCache()
    fp_cache.update_and_fetch(keys, values)
    turbo_cache = TurboQuantKVCache.from_cache(fp_cache, bits=3.5)
    turbo_keys, turbo_values = turbo_cache.state
    dequantized_keys, dequantized_values = turbo_cache.dequantize(
        turbo_keys, turbo_values
    )

    reference = mx.fast.scaled_dot_product_attention(
        queries,
        dequantized_keys.astype(queries.dtype),
        dequantized_values.astype(queries.dtype),
        scale=32**-0.5,
        mask="causal",
    )
    quantized = scaled_dot_product_attention(
        queries, turbo_keys, turbo_values, turbo_cache, scale=32**-0.5, mask="causal"
    )

    diff = mx.max(mx.abs(reference - quantized)).item()
    assert quantized.shape == reference.shape
    assert diff < 1e-4


def test_resolve_kv_bits_validates_overrides():
    with pytest.raises(ValueError):
        resolve_kv_bits(4, 0.5, None)
    with pytest.raises(ValueError):
        resolve_kv_bits(4, None, 3.25)


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


def test_apc_warm_cache_defaults_match_live_split():
    from mlx_vlm.apc import _empty_quant_batch_cache

    built = _empty_quant_batch_cache([0], _turbo_quant_config())
    assert (built.key_bits, built.value_bits) == (3.0, 4.0)


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


def _hybrid_policy():
    from mlx_vlm.kv_quant import from_legacy

    return from_legacy(8, "uniform", 64, kv_value_bits=3, kv_value_scheme="turboquant")


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
        from_legacy(8, "uniform", 64, kv_value_bits=3.5, kv_value_scheme="turboquant")
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


# Batched attention

H, D = 4, 64  # kv heads, head_dim
BITS = 4
SCALE = D**-0.5


def _rand_kv(batch, seq_len, heads=H):
    k = mx.random.normal((batch, heads, seq_len, D))
    v = mx.random.normal((batch, heads, seq_len, D))
    return k, v


def _filled(left_padding, seq_len, batch=None, bits=BITS):
    batch = len(left_padding) if batch is None else batch
    cache = BatchTurboQuantKVCache(left_padding, bits=bits)
    keys, values = cache.update_and_fetch(*_rand_kv(batch, seq_len))
    return cache, keys, values


class TestFusedPathGuard:
    def test_cached_eligibility_tracks_batch_lifecycle(self):
        cache = BatchTurboQuantKVCache([0], bits=BITS)
        other = BatchTurboQuantKVCache([0], bits=BITS)
        assert cache.fused_attention_eligible

        cache.extend(other)
        assert not cache.fused_attention_eligible
        assert not _turboquant_attention_applies(cache)

        cache.filter(mx.array([0]))
        assert cache.fused_attention_eligible
        assert _turboquant_attention_applies(cache)

        cache.state = BatchTurboQuantKVCache([2], bits=BITS).state
        assert not cache.fused_attention_eligible
        assert not _turboquant_attention_applies(cache)


class TestNumericalEquivalence:
    """The fused path must agree with the dequantizing fallback."""

    def _reference(self, cache, queries, keys, values, mask=None):
        dq_k, dq_v = cache.dequantize(keys, values)
        return mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=mask,
        )

    def test_multi_row_still_produces_correct_shape(self):
        cache, keys, values = _filled([0, 0], 12)
        queries = mx.random.normal((2, H, 1, D))
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(out)
        assert out.shape == (2, H, 1, D)


class TestDecodeMemoryIsFlat:
    """Regression guard for the bug this change fixes.

    The dequantizing fallback materialised the whole KV cache as float32 on
    every step, so peak memory scaled with the context length. The fused path
    reads the quantized state in place.
    """

    def _peak_delta_for(self, seq_len):
        cache, keys, values = _filled([0], seq_len)
        queries = mx.random.normal((1, H, 1, D))
        mx.eval(cache.keys, cache.values, queries)
        mx.clear_cache()

        before = mx.get_peak_memory()
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(out)
        return mx.get_peak_memory() - before

    def test_peak_does_not_scale_with_context(self):
        short = self._peak_delta_for(256)
        long = self._peak_delta_for(4096)
        # 16x the context. Dequantizing would grow the step's peak roughly in
        # step with it; the fused kernels keep it bounded.
        assert long <= max(short, 1 << 20) * 4


# Rotated value kernels


def _codec_and_state(dim: int, bits: int, n_heads: int, n_tokens: int, seed: int):
    mx.random.seed(seed)
    values = mx.random.normal((1, n_heads, n_tokens, dim))
    codec = _TurboQuantMSECodec(dim, bits, seed=seed)
    return codec, codec.quantize(values)


def _dequant_weighted_sum(codec, state, weights):
    """Math ground truth: weighted sum over the codec's own dequantized values."""
    deq = codec.dequantize(state)  # (1, H, T, D)
    return mx.einsum("bhmlt,bhtd->bhmld", weights, deq)


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("n_repeats", [1, 4])
def test_rht_weighted_sum_matches_einsum_and_dequant(dim, bits, n_repeats, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    n_heads, n_tokens = 2, 24
    codec, state = _codec_and_state(dim, bits, n_heads, n_tokens, seed=0)
    # power-of-2 dim -> the RHT path that #1244 disabled for these kernels
    assert codec.use_rht is True

    weights = mx.softmax(
        mx.random.normal((1, n_heads, n_repeats, 1, n_tokens)), axis=-1
    )

    kernel_out = codec.weighted_sum(weights, state)  # Metal -> RHT kernel fast path
    truth = _dequant_weighted_sum(codec, state, weights)

    # Force the einsum fallback by hiding Metal, then compare paths.
    monkeypatch.setattr(tq, "_metal_available", lambda: False)
    einsum_out = codec.weighted_sum(weights, state)

    assert kernel_out.shape == einsum_out.shape == truth.shape
    assert mx.max(mx.abs(kernel_out - einsum_out)).item() < 1e-4
    assert mx.max(mx.abs(kernel_out - truth)).item() < 1e-3


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bits", [3, 4])
def test_rht_weighted_sum_stats_matches_einsum(dim, bits, monkeypatch):
    if not mx.metal.is_available():
        pytest.skip("Metal kernels are unavailable on this host")

    n_heads, n_repeats, n_tokens = 2, 4, 24
    codec, state = _codec_and_state(dim, bits, n_heads, n_tokens, seed=1)
    assert codec.use_rht is True

    scores = mx.random.normal((1, n_heads, n_repeats, 1, n_tokens))

    out_k, denom_k, max_k = codec.weighted_sum_stats_from_scores(scores, state)

    monkeypatch.setattr(tq, "_metal_available", lambda: False)
    out_e, denom_e, max_e = codec.weighted_sum_stats_from_scores(scores, state)

    assert mx.max(mx.abs(out_k - out_e)).item() < 1e-4
    assert mx.max(mx.abs(denom_k - denom_e)).item() < 1e-4
    assert mx.max(mx.abs(max_k - max_e)).item() < 1e-4
