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
