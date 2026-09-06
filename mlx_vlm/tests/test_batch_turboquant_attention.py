"""Tests for the TurboQuant attention path shared by both KV cache types.

``BatchTurboQuantKVCache`` used to fall back to dequantizing the whole cache
on every step, which made decode memory grow with the context length. It now
reuses the fused kernels through ``_TurboQuantAttentionMixin`` whenever it
holds a single unpadded row.
"""

import mlx.core as mx
import pytest

from mlx_vlm.models.base import (
    _turboquant_attention_applies,
    scaled_dot_product_attention,
)
from mlx_vlm.turboquant import (
    BatchTurboQuantKVCache,
    TurboQuantKVCache,
    _TurboQuantAttentionMixin,
)

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


class TestSharedAttentionSurface:
    """Both caches expose the same attention API through the mixin."""

    @pytest.mark.parametrize("cls", [TurboQuantKVCache, BatchTurboQuantKVCache])
    def test_inherits_mixin(self, cls):
        assert issubclass(cls, _TurboQuantAttentionMixin)

    @pytest.mark.parametrize(
        "name",
        [
            "decode_attention",
            "prefill_attention",
            "quantized_attention",
            "decode_key_chunk_size",
            "prefill_key_chunk_size",
            "prefill_query_block_size",
        ],
    )
    @pytest.mark.parametrize("cls", [TurboQuantKVCache, BatchTurboQuantKVCache])
    def test_attribute_present(self, cls, name):
        # The chunk-size constants live on the mixin: reading them off the
        # batch cache used to raise AttributeError mid-decode.
        assert hasattr(cls, name)

    def test_attention_states_ignores_batch_bookkeeping(self):
        # The batch cache's `state` is a 4-tuple; the mixin must not unpack it.
        cache, _, _ = _filled([0], 4)
        keys_state, values_state = cache._attention_states()
        assert keys_state is not None and values_state is not None


class TestFusedPathGuard:
    def test_single_cache_always_applies(self):
        assert _turboquant_attention_applies(TurboQuantKVCache(bits=BITS))

    def test_single_unpadded_row_applies(self):
        cache, _, _ = _filled([0], 8)
        assert _turboquant_attention_applies(cache)

    def test_multi_row_does_not_apply(self):
        cache, _, _ = _filled([0, 0], 8)
        assert not _turboquant_attention_applies(cache)

    def test_left_padded_row_does_not_apply(self):
        # Padded positions would otherwise be attended to as real tokens.
        cache, _, _ = _filled([3], 8)
        assert not _turboquant_attention_applies(cache)

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

    @pytest.mark.parametrize("seq_len", [16, 300])
    def test_decode_matches_fallback(self, seq_len):
        cache, keys, values = _filled([0], seq_len)
        queries = mx.random.normal((1, H, 1, D))

        fused = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        reference = self._reference(cache, queries, keys, values)
        mx.eval(fused, reference)

        assert fused.shape == reference.shape
        # Both paths read the same quantized state, so they differ only by
        # kernel arithmetic order.
        assert mx.allclose(fused, reference, atol=2e-2).item()

    def test_multi_row_still_produces_correct_shape(self):
        cache, keys, values = _filled([0, 0], 12)
        queries = mx.random.normal((2, H, 1, D))
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(out)
        assert out.shape == (2, H, 1, D)

    def test_left_padded_matches_fallback(self):
        cache, keys, values = _filled([2], 10)
        queries = mx.random.normal((1, H, 1, D))
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        reference = self._reference(cache, queries, keys, values)
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()


class TestShapesBeyondTheDefaultLayout:
    """The fused path has to survive head geometries other than the default.

    Models differ in head_dim and in how many query heads share a KV head, so
    exercise a couple of combinations rather than only 4 heads at 64 dims.
    """

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    @pytest.mark.parametrize("q_per_kv", [1, 4, 6])
    def test_decode_matches_fallback(self, head_dim, q_per_kv):
        kv_heads, seq_len = 4, 96
        scale = head_dim**-0.5
        cache = BatchTurboQuantKVCache([0], bits=BITS)
        keys, values = cache.update_and_fetch(
            mx.random.normal((1, kv_heads, seq_len, head_dim)),
            mx.random.normal((1, kv_heads, seq_len, head_dim)),
        )
        queries = mx.random.normal((1, kv_heads * q_per_kv, 1, head_dim))

        fused = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=None
        )
        dq_k, dq_v = cache.dequantize(keys, values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=scale,
            mask=None,
        )
        mx.eval(fused, reference)
        assert fused.shape == reference.shape
        assert mx.allclose(fused, reference, atol=2e-2).item()


class TestAttentionSinks:
    """Sinks must be applied, not dropped and not rejected.

    The fused kernels carry no sink term, so a request with sinks has to fall
    through to the dequantizing path, which can pass them to MLX.
    """

    def _with_sinks(self, cache, keys, values, queries, sinks):
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None, sinks=sinks
        )

    def test_sinks_are_applied(self):
        cache, keys, values = _filled([0], 64)
        queries = mx.random.normal((1, H, 1, D))
        sinks = mx.random.normal((H,))

        out = self._with_sinks(cache, keys, values, queries, sinks)
        dq_k, dq_v = cache.dequantize(keys, values)
        reference = mx.fast.scaled_dot_product_attention(
            queries,
            dq_k.astype(queries.dtype),
            dq_v.astype(queries.dtype),
            scale=SCALE,
            mask=None,
            sinks=sinks,
        )
        mx.eval(out, reference)
        assert mx.allclose(out, reference, atol=2e-2).item()

    def test_sinks_change_the_result(self):
        # Guards against silently discarding them: the batch cache used to
        # drop sinks on the floor and return the no-sink answer.
        cache, keys, values = _filled([0], 64)
        queries = mx.random.normal((1, H, 1, D))
        sinks = mx.full((H,), 5.0)

        with_sinks = self._with_sinks(cache, keys, values, queries, sinks)
        without = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=SCALE, mask=None
        )
        mx.eval(with_sinks, without)
        assert not mx.allclose(with_sinks, without, atol=1e-3).item()


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
