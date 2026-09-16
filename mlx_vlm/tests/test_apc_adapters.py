"""Tests for APC adapters over batch / quantized / SWA cache layouts.

Covers row snapshot, extract, hybrid exact store/lookup, reject stats, and
dequant-aware block harvest. Uses real cache types with small dimensions.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.apc import (
    APCManager,
    _clone_prompt_cache_for_apc,
    extract_prompt_cache_from_batch,
)
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
    RotatingKVCache,
)

# Small dims: fast unit tests, no GPU pressure
B, H, D = 1, 2, 32
GROUP_SIZE = 32
BITS = 8
BLOCK_SIZE = 16
SWA_MAX = 64


def _rand_kv(batch=B, seq_len=32, heads=H, dim=D):
    k = mx.random.normal((batch, heads, seq_len, dim))
    v = mx.random.normal((batch, heads, seq_len, dim))
    mx.eval(k, v)
    return k, v


def _max_abs_error(a: mx.array, b: mx.array) -> float:
    return mx.max(mx.abs(a - b)).item()


def test_pooling_cache_exact_apc_round_trips_warm_and_cold_rows():
    from mlx_vlm.apc import make_warm_batch_exact_cache_multi, snapshot_prompt_cache_row

    def make_row(prompt_length):
        rotating = RotatingKVCache(max_size=16)
        pooling = PoolingCache(ratio=4)
        if prompt_length > 0:
            keys = mx.arange(prompt_length * 3, dtype=mx.float32).reshape(
                1, 1, prompt_length, 3
            )
            rotating.update_and_fetch(keys, keys + 1)
            kv = keys.reshape(1, prompt_length, 3)
            gate = mx.ones((1, prompt_length, 2), dtype=mx.float32)
            ready_kv, _, _ = pooling.accumulate_windows(kv, gate, offset=0)
            pooled_length = ready_kv.shape[1] // pooling.ratio
            pooling.update_and_fetch(mx.ones((1, pooled_length, 3), dtype=mx.float32))
        return [CacheList(rotating, pooling)]

    warm = snapshot_prompt_cache_row(make_row(6), batch_idx=0)
    cold = snapshot_prompt_cache_row(make_row(0), batch_idx=0)

    assert warm is not None
    assert cold is not None
    merged, max_prefix = make_warm_batch_exact_cache_multi(
        [warm, cold], prefix_lens=[6, 0]
    )

    assert merged is not None
    assert max_prefix == 6
    rotating, pooling = merged[0].caches
    assert isinstance(rotating, BatchRotatingKVCache)
    assert rotating.offset.tolist() == [6, 0]
    assert isinstance(pooling, BatchPoolingCache)
    assert pooling.ratio == 4
    assert pooling.remainder == [2, 0]
    assert pooling._pool_lengths == [1, 0]
    assert pooling._processed == [6, 0]


def test_batch_pooling_cache_merge_accepts_prefix_lengths():
    merged = BatchPoolingCache.merge(
        [PoolingCache(ratio=4), PoolingCache(ratio=4)], prefix_lens=[0, 0]
    )

    assert isinstance(merged, BatchPoolingCache)
    assert merged.remainder == [0, 0]
    assert merged._pool_lengths == [0, 0]
    assert merged._processed == [0, 0]


def test_pooling_cache_exact_batch_merge_forwards_prefix_lengths():
    from mlx_vlm.apc_adapters import merge_cache_entries

    warm = PoolingCache(ratio=4)
    kv = mx.arange(18, dtype=mx.float32).reshape(1, 6, 3)
    gate = mx.ones((1, 6, 2), dtype=mx.float32)
    warm.accumulate_windows(kv, gate, offset=0)
    warm.update_and_fetch(mx.ones((1, 1, 3), dtype=mx.float32))
    cold = PoolingCache(ratio=4)

    merged = merge_cache_entries([warm, cold], [6, 0])

    assert isinstance(merged, BatchPoolingCache)
    assert merged.remainder == [2, 0]
    assert merged._pool_lengths == [1, 0]
    assert merged._processed == [6, 0]
    extracted_warm = merged.extract(0)
    extracted_cold = merged.extract(1)
    assert extracted_warm.remainder == 2
    assert extracted_warm.pooled.shape == (1, 1, 3)
    assert extracted_cold.empty()


def _fill_batch_kv(left_padding, seq_len):
    cache = BatchKVCache(list(left_padding))
    k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache, k, v


def _fill_batch_quant(left_padding, seq_len):
    cache = BatchQuantizedKVCache(list(left_padding), group_size=GROUP_SIZE, bits=BITS)
    k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys)
    return cache, k, v


def _fill_batch_rotating(left_padding, seq_len, max_size=SWA_MAX):
    cache = BatchRotatingKVCache(max_size, list(left_padding))
    k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache, k, v


class TestBatchCacheIntrospection:
    """batch_size, is_single_row, and empty on batch cache types."""

    def test_batch_rotating_empty_and_batch_size(self):
        empty = BatchRotatingKVCache(SWA_MAX, [0, 0])
        assert empty.empty() is True
        assert empty.batch_size == 2
        assert empty.is_single_row() is False

        filled, _, _ = _fill_batch_rotating([0], seq_len=8)
        assert filled.empty() is False
        assert filled.batch_size == 1
        assert filled.is_single_row() is True


class TestBatchQuantizedExtract:
    def test_extract_empty_cache(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        row = cache.extract(0)
        assert isinstance(row, QuantizedKVCache)
        assert row.keys is None or row.offset == 0


class TestQuantizedBlockHarvest:
    def test_layer_kv_float_helper_handles_quantized_tuple_keys(self):
        from mlx_vlm.apc import layer_kv_for_apc

        plain = KVCache()
        k, v = _rand_kv(batch=1, seq_len=12)
        plain.update_and_fetch(k, v)
        pk, pv = layer_kv_for_apc(plain)
        assert pk is not None and pv is not None
        assert pk.shape[-2] == 12

        q = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=12)
        q.update_and_fetch(k, v)
        qk, qv = layer_kv_for_apc(q)
        mx.eval(qk, qv)
        assert qk.shape == (1, H, 12, D)
        assert not isinstance(qk, tuple)

        bq, _, _ = _fill_batch_quant([0, 0], seq_len=12)
        bk, bv = layer_kv_for_apc(bq, batch_idx=1)
        mx.eval(bk, bv)
        assert bk.shape[0] == 1
        assert bk.shape[-2] <= 12

    def test_layer_kv_rejects_unknown_without_crashing(self):
        from mlx_vlm.apc import layer_kv_for_apc

        class Bogus:
            keys = (1, 2, 3)
            values = (4, 5, 6)
            offset = 3

        assert layer_kv_for_apc(Bogus()) == (None, None)


class TestAlwaysExtractSemantics:
    def test_extract_b1_batch_rotating_equals_clone_after_extract(self):
        cache, _, _ = _fill_batch_rotating([0], seq_len=16)
        row = extract_prompt_cache_from_batch([cache], 0)
        assert row is not None
        cloned = _clone_prompt_cache_for_apc(row)
        assert cloned is not None


def _fill_batch_turbo(left_padding, seq_len, bits=4.0):
    from mlx_vlm.turboquant import BatchTurboQuantKVCache

    cache = BatchTurboQuantKVCache(list(left_padding), bits=bits)
    k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys)
    return cache, k, v


class TestBatchRightPadPrefill:
    @staticmethod
    def _filter_reordered_row(cache):
        cache.prepare(right_padding=[2, 0], lengths=[4, 6])
        k, v = _rand_kv(batch=2, seq_len=3)
        cache.update_and_fetch(k, v)

        cache.filter(mx.array([1, 0], dtype=mx.int32))
        cache.filter(mx.array([1], dtype=mx.int32))

    def test_batch_kv_filter_keeps_pending_right_padding_aligned(self):
        cache = BatchKVCache([0, 0])

        self._filter_reordered_row(cache)

        assert cache._right_padding.tolist() == [2]
        cache.finalize()
        assert cache.keys.shape[0] == 1
        assert cache.values.shape[0] == 1
        assert cache.offset.tolist() == [1]
        assert cache.left_padding.tolist() == [2]

    def test_batch_quantized_filter_keeps_pending_right_padding_aligned(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)

        self._filter_reordered_row(cache)

        assert cache._right_padding.tolist() == [2]
        cache.finalize()
        assert all(part.shape[0] == 1 for part in cache.keys)
        assert all(part.shape[0] == 1 for part in cache.values)
        assert cache.offset.tolist() == [1]
        assert cache.left_padding.tolist() == [2]

    def test_rotating_filter_keeps_pending_lengths_aligned(self):
        cache = BatchRotatingKVCache(32, [0, 0])

        self._filter_reordered_row(cache)

        assert cache._lengths.tolist() == [4]
        k, v = _rand_kv(batch=1, seq_len=2)
        out_k, out_v = cache.update_and_fetch(k, v)
        mx.eval(out_k, out_v)
        assert out_k.shape[0] == 1
        assert out_v.shape[0] == 1

        cache.finalize()
        assert cache.keys.shape[0] == 1
        assert cache.values.shape[0] == 1
        assert cache.offset.tolist() == [4]
        assert cache.left_padding.tolist() == [1]


class TestBatchTurboQuantParity:
    def test_batch_size_and_is_single_row(self):
        from mlx_vlm.turboquant import BatchTurboQuantKVCache

        empty = BatchTurboQuantKVCache([0, 0], bits=4.0)
        assert empty.empty() is True
        assert empty.batch_size == 2
        assert empty.is_single_row() is False

        filled, _, _ = _fill_batch_turbo([0], seq_len=8)
        assert filled.empty() is False
        assert filled.batch_size == 1
        assert filled.is_single_row() is True

    def test_extract_returns_turboquant_kv_cache(self):
        from mlx_vlm.turboquant import TurboQuantKVCache

        cache, k, _ = _fill_batch_turbo([0, 0], seq_len=24)
        row = cache.extract(1)
        assert isinstance(row, TurboQuantKVCache)
        assert row.offset == 24
        dk, dv = row.dequantize_for_apc()
        mx.eval(dk, dv)
        assert dk.shape == (1, H, 24, D)
        # TurboQuant is lossy; keep a loose bound
        assert _max_abs_error(dk, k[1:2]) < 2.0

    def test_extract_empty(self):
        from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache

        cache = BatchTurboQuantKVCache([0, 0], bits=4.0)
        row = cache.extract(0)
        assert isinstance(row, TurboQuantKVCache)
        assert row.keys is None or row.offset == 0

    def test_snapshot_and_exact_store_multi_row(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 2 * BLOCK_SIZE
        token_ids = list(range(seq_len))
        turbo, _, _ = _fill_batch_turbo([0, 0], seq_len=seq_len)
        batch_kv, _, _ = _fill_batch_kv([0, 0], seq_len=seq_len)
        prompt_cache = [turbo, batch_kv]

        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        for bi in (0, 1):
            snap = snapshot_prompt_cache_row(prompt_cache, batch_idx=bi)
            assert snap is not None
            for c in snap:
                assert not type(c).__name__.startswith("Batch")
            assert manager.store_exact_cache(token_ids, snap, extra_hash=bi + 1)

        warm0, m0 = manager.lookup_exact_cache(token_ids + [9], extra_hash=1)
        warm1, m1 = manager.lookup_exact_cache(token_ids + [9], extra_hash=2)
        assert m0 == seq_len and m1 == seq_len
        assert warm0 is not None and warm1 is not None

    def test_layer_kv_for_apc_batch_turbo(self):
        from mlx_vlm.apc import layer_kv_for_apc

        cache, _, _ = _fill_batch_turbo([0, 0], seq_len=12)
        k, v = layer_kv_for_apc(cache, batch_idx=1)
        mx.eval(k, v)
        assert k is not None and v is not None
        assert k.shape[0] == 1
        assert k.shape[-2] <= 12
        assert not isinstance(k, tuple)
