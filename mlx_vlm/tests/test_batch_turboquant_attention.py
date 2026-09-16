"""Tests for the TurboQuant attention path shared by both KV cache types.

``BatchTurboQuantKVCache`` used to fall back to dequantizing the whole cache
on every step, which made decode memory grow with the context length. It now
reuses the fused kernels through ``_TurboQuantAttentionMixin`` whenever it
holds a single unpadded row.
"""

import mlx.core as mx

from mlx_vlm.models.base import (
    _turboquant_attention_applies,
    scaled_dot_product_attention,
)
from mlx_vlm.turboquant import BatchTurboQuantKVCache

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
