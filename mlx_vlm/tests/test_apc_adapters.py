"""TDD tests for APC adapter boundary + batch/SWA kv-bits coverage (#1559).

Follow-up to #1534. These tests define the contract before / alongside
implementation:

1. Clear batch cache APIs (``batch_size``, ``is_single_row``, ``empty``)
2. ``BatchQuantizedKVCache.extract`` (multi-row exact path)
3. Always row-normalize via ``snapshot_prompt_cache_row`` (no B=1 special case)
4. Gemma-like hybrid: BatchRotating + BatchQuantized + BatchKV
5. Reject observability (stats + reason codes)
6. Post-decode / block harvest handles quantized ``keys`` tuples

No mocking of cache behavior — same production types as server/BatchGenerator.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.apc import (
    APCManager,
    _cache_entry_supports_exact_apc,
    _clone_cache_entry_for_apc,
    _clone_prompt_cache_for_apc,
    extract_prompt_cache_from_batch,
    harvest_blocks_from_batch_cache,
    model_apc_mode,
)
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    KVCache,
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


# ---------------------------------------------------------------------------
# A — Cache API clarity (Lucas nit)
# ---------------------------------------------------------------------------


class TestBatchCacheIntrospection:
    """batch_size / is_single_row / empty — no poking at _idx or keys.shape[0]."""

    def test_batch_kv_empty_and_single_row(self):
        empty = BatchKVCache([0])
        assert empty.empty() is True
        assert empty.batch_size == 1
        assert empty.is_single_row() is True

        multi = BatchKVCache([0, 0, 0])
        assert multi.batch_size == 3
        assert multi.is_single_row() is False

        filled, _, _ = _fill_batch_kv([0, 0], seq_len=8)
        assert filled.empty() is False
        assert filled.batch_size == 2
        assert filled.is_single_row() is False

    def test_batch_quantized_empty_and_batch_size(self):
        empty = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        assert empty.empty() is True
        assert empty.batch_size == 2
        assert empty.is_single_row() is False

        filled, _, _ = _fill_batch_quant([0], seq_len=8)
        assert filled.empty() is False
        assert filled.batch_size == 1
        assert filled.is_single_row() is True

    def test_batch_rotating_empty_and_batch_size(self):
        empty = BatchRotatingKVCache(SWA_MAX, [0, 0])
        assert empty.empty() is True
        assert empty.batch_size == 2
        assert empty.is_single_row() is False

        filled, _, _ = _fill_batch_rotating([0], seq_len=8)
        assert filled.empty() is False
        assert filled.batch_size == 1
        assert filled.is_single_row() is True

    def test_clone_uses_empty_not_private_fields(self):
        """_clone_cache_entry_for_apc should not require poking _idx for empty."""
        empty_bk = BatchKVCache([0])
        eval_targets: list = []
        cloned = _clone_cache_entry_for_apc(
            empty_bk, min_capacity_tokens=None, eval_targets=eval_targets
        )
        assert isinstance(cloned, KVCache)
        assert cloned.keys is None or int(getattr(cloned, "offset", 0) or 0) == 0


# ---------------------------------------------------------------------------
# A — BatchQuantizedKVCache.extract (multi-row exact)
# ---------------------------------------------------------------------------


class TestBatchQuantizedExtract:
    def test_extract_returns_quantized_kv_cache(self):
        cache, k, v = _fill_batch_quant([0, 0], seq_len=24)
        row = cache.extract(1)
        assert isinstance(row, QuantizedKVCache)
        assert row.group_size == GROUP_SIZE
        assert row.bits == BITS
        assert row.offset == 24
        dk, dv = row.dequantize_for_apc()
        mx.eval(dk, dv)
        assert dk.shape == (1, H, 24, D)
        # Row 1 should match original float within 8-bit tolerance
        assert _max_abs_error(dk, k[1:2]) < 0.15

    def test_extract_respects_left_padding(self):
        # left_padding=[2, 0]: row 0 has 2 pad tokens at the front of the buffer
        cache = BatchQuantizedKVCache([2, 0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=2, seq_len=10)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        row0 = cache.extract(0)
        # After dropping left pad, valid tokens = _idx - pad
        assert row0.offset == cache._idx - 2
        row1 = cache.extract(1)
        assert row1.offset == cache._idx

    def test_extract_empty_cache(self):
        cache = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
        row = cache.extract(0)
        assert isinstance(row, QuantizedKVCache)
        assert row.keys is None or row.offset == 0

    def test_extract_prompt_cache_from_batch_multi_row(self):
        """Multi-row hybrid layout becomes extractable end-to-end."""
        seq_len = 2 * BLOCK_SIZE
        arrays = ArraysCache(2, left_padding=[0, 0])
        arrays.cache = [
            mx.zeros((2, seq_len, D)),
            mx.zeros((2, seq_len, D)),
        ]
        batch_kv, _, _ = _fill_batch_kv([0, 0], seq_len=seq_len)
        batch_q, _, _ = _fill_batch_quant([0, 0], seq_len=seq_len)

        prompt_cache = [arrays, batch_kv, batch_q]
        row0 = extract_prompt_cache_from_batch(prompt_cache, 0)
        row1 = extract_prompt_cache_from_batch(prompt_cache, 1)
        assert row0 is not None and row1 is not None
        assert len(row0) == 3 and len(row1) == 3
        assert isinstance(row0[1], KVCache)
        assert isinstance(row0[2], QuantizedKVCache)
        assert isinstance(row1[2], QuantizedKVCache)


# ---------------------------------------------------------------------------
# A — Always row-normalize (snapshot adapter)
# ---------------------------------------------------------------------------


class TestSnapshotPromptCacheRow:
    """snapshot_prompt_cache_row is the single inbound adapter for APC stores."""

    def test_api_exists(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        assert callable(snapshot_prompt_cache_row)

    def test_b1_batch_kv_collapses_to_single_row(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 2 * BLOCK_SIZE
        batch_kv, _, _ = _fill_batch_kv([0], seq_len=seq_len)
        batch_q, _, _ = _fill_batch_quant([0], seq_len=seq_len)
        snap = snapshot_prompt_cache_row([batch_kv, batch_q], batch_idx=0)
        assert snap is not None
        assert len(snap) == 2
        # APC core never sees Batch* types
        assert not isinstance(snap[0], BatchKVCache)
        assert not isinstance(snap[1], BatchQuantizedKVCache)
        assert isinstance(snap[0], KVCache)
        assert isinstance(snap[1], KVCache)  # dequant at harvest
        assert snap[0].offset == seq_len
        assert snap[1].offset == seq_len

    def test_multi_row_extracts_requested_index(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = BLOCK_SIZE
        batch_kv, k, _ = _fill_batch_kv([0, 0], seq_len=seq_len)
        snap0 = snapshot_prompt_cache_row([batch_kv], batch_idx=0)
        snap1 = snapshot_prompt_cache_row([batch_kv], batch_idx=1)
        assert snap0 is not None and snap1 is not None
        mx.eval(snap0[0].keys, snap1[0].keys)
        # Distinct rows
        assert _max_abs_error(snap0[0].keys, k[0:1, :, :seq_len, :]) < 1e-5
        assert _max_abs_error(snap1[0].keys, k[1:2, :, :seq_len, :]) < 1e-5
        assert _max_abs_error(snap0[0].keys, snap1[0].keys) > 1e-3

    def test_already_single_row_plain_kv_still_works(self):
        """generate() path uses non-batch KVCache — snapshot must not require extract."""
        from mlx_vlm.apc import snapshot_prompt_cache_row

        cache = KVCache()
        k, v = _rand_kv(batch=1, seq_len=20)
        cache.update_and_fetch(k, v)
        snap = snapshot_prompt_cache_row([cache], batch_idx=0)
        assert snap is not None
        assert isinstance(snap[0], KVCache)
        assert snap[0].offset == 20

    def test_store_exact_via_snapshot_multi_row(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 2 * BLOCK_SIZE
        token_ids = list(range(seq_len))
        batch_kv, _, _ = _fill_batch_kv([0, 0], seq_len=seq_len)
        batch_q, _, _ = _fill_batch_quant([0, 0], seq_len=seq_len)
        prompt_cache = [batch_kv, batch_q]

        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        for bi in (0, 1):
            snap = snapshot_prompt_cache_row(prompt_cache, batch_idx=bi)
            assert snap is not None
            # Use distinct extra_hash so rows don't collide in exact dict
            stored = manager.store_exact_cache(token_ids, snap, extra_hash=bi + 1)
            assert stored is True

        warm0, m0 = manager.lookup_exact_cache(token_ids + [9], extra_hash=1)
        warm1, m1 = manager.lookup_exact_cache(token_ids + [9], extra_hash=2)
        assert m0 == seq_len and m1 == seq_len
        assert warm0 is not None and warm1 is not None


# ---------------------------------------------------------------------------
# A — Gemma-like SWA hybrid (BatchRotating + quant + full attn)
# ---------------------------------------------------------------------------


class TestGemmaLikeHybridExact:
    def _gemma_like_layout(self, batch_size: int, seq_len: int):
        """Approximate Gemma 4 hybrid serving layout under --kv-bits.

        Typical pattern: SWA rotating layers + quantized full-attn layers +
        possibly a dense last layer. Arrays not required for pure gemma SWA.
        """
        pads = [0] * batch_size
        rotating, _, _ = _fill_batch_rotating(pads, seq_len=seq_len, max_size=SWA_MAX)
        quant, _, _ = _fill_batch_quant(pads, seq_len=seq_len)
        dense, _, _ = _fill_batch_kv(pads, seq_len=seq_len)
        return [rotating, quant, dense]

    def test_supports_exact_apc_for_batch_rotating(self):
        cache, _, _ = _fill_batch_rotating([0], seq_len=16)
        assert _cache_entry_supports_exact_apc(cache) is True

    def test_extract_batch_rotating_then_clone(self):
        cache, _, _ = _fill_batch_rotating([0], seq_len=20)
        row = cache.extract(0)
        assert isinstance(row, RotatingKVCache)
        eval_targets: list = []
        cloned = _clone_cache_entry_for_apc(
            row, min_capacity_tokens=None, eval_targets=eval_targets
        )
        assert cloned is not None
        assert isinstance(cloned, RotatingKVCache)

    def test_b1_gemma_like_exact_store_and_lookup(self):
        """#1559 success criterion: Gemma batch path B=1 exact store + hit."""
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 2 * BLOCK_SIZE
        token_ids = list(range(seq_len))
        prompt_cache = self._gemma_like_layout(batch_size=1, seq_len=seq_len)

        # Layout must be recognized as exact-capable
        assert all(_cache_entry_supports_exact_apc(c) for c in prompt_cache)

        snap = snapshot_prompt_cache_row(prompt_cache, batch_idx=0)
        assert snap is not None
        # No Batch* types in APC storage
        for c in snap:
            assert not type(c).__name__.startswith("Batch")

        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        assert manager.store_exact_cache(token_ids, snap, extra_hash=0) is True
        assert manager.stats.exact_stores == 1

        warm, matched = manager.lookup_exact_cache(token_ids + [99], extra_hash=0)
        assert matched == seq_len
        assert warm is not None
        assert len(warm) == 3

    def test_multi_row_gemma_like_exact_store(self):
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 2 * BLOCK_SIZE
        token_ids = list(range(seq_len))
        prompt_cache = self._gemma_like_layout(batch_size=2, seq_len=seq_len)

        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        for bi in (0, 1):
            snap = snapshot_prompt_cache_row(prompt_cache, batch_idx=bi)
            assert snap is not None
            assert manager.store_exact_cache(token_ids, snap, extra_hash=10 + bi)

        for bi in (0, 1):
            warm, matched = manager.lookup_exact_cache(
                token_ids + [1], extra_hash=10 + bi
            )
            assert matched == seq_len
            assert warm is not None

    def test_model_apc_mode_exact_for_gemma_like_make_cache(self):
        """model_apc_mode must return exact when make_cache yields SWA hybrid."""

        class FakeGemmaLang:
            def make_cache(self):
                # Empty caches of the hybrid types (serving layout under kv-bits)
                return [
                    BatchRotatingKVCache(SWA_MAX, [0]),
                    BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS),
                    BatchKVCache([0]),
                ]

        assert model_apc_mode(FakeGemmaLang()) == "exact"


# ---------------------------------------------------------------------------
# C — Reject observability
# ---------------------------------------------------------------------------


class TestRejectObservability:
    def test_stats_include_rejects_keys(self):
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        snap = manager.stats.snapshot(manager.num_blocks, manager.block_size)
        assert "rejects" in snap
        assert "rejects_by_reason" in snap
        assert "last_reject" in snap
        assert snap["rejects"] == 0
        assert snap["rejects_by_reason"] == {}
        assert snap["last_reject"] is None

    def test_unclonable_exact_store_increments_rejects(self):
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        token_ids = list(range(BLOCK_SIZE))

        class UnclonableCache:
            """No extract, no dequantize_for_apc, not a known cache type."""

            keys = "not-an-array"
            values = "not-an-array"

        stored = manager.store_exact_cache(token_ids, [UnclonableCache()])
        assert stored is False
        assert manager.stats.rejects >= 1
        assert manager.stats.rejects_by_reason.get("unclonable", 0) >= 1
        last = manager.stats.last_reject
        assert last is not None
        assert last.get("reason") == "unclonable"

        snap = manager.stats.snapshot(manager.num_blocks, manager.block_size)
        assert snap["rejects"] >= 1
        assert "unclonable" in snap["rejects_by_reason"]


# ---------------------------------------------------------------------------
# D — Block harvest / post-decode path with quantized keys
# ---------------------------------------------------------------------------


class TestQuantizedBlockHarvest:
    def test_harvest_single_row_batch_quantized(self):
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE
        cache, _, _ = _fill_batch_quant([0], seq_len=seq_len)
        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, [cache], batch_idx=0, full_token_ids=token_ids
        )
        assert len(blocks) == 2
        manager.release(blocks)

    def test_layer_kv_float_helper_handles_quantized_tuple_keys(self):
        """dispatch post-decode harvest must not slice quantized tuple keys.

        ``layer_kv_for_apc`` (or equivalent) returns float K/V for any supported
        cache dialect so callers never do ``keys[..., :off, :]`` on a tuple.
        """
        from mlx_vlm.apc import layer_kv_for_apc

        # Plain KV
        plain = KVCache()
        k, v = _rand_kv(batch=1, seq_len=12)
        plain.update_and_fetch(k, v)
        pk, pv = layer_kv_for_apc(plain)
        assert pk is not None and pv is not None
        assert pk.shape[-2] == 12

        # Quantized single-row
        q = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=12)
        q.update_and_fetch(k, v)
        qk, qv = layer_kv_for_apc(q)
        mx.eval(qk, qv)
        assert qk.shape == (1, H, 12, D)
        assert not isinstance(qk, tuple)

        # Batch quantized row
        bq, _, _ = _fill_batch_quant([0, 0], seq_len=12)
        bk, bv = layer_kv_for_apc(bq, batch_idx=1)
        mx.eval(bk, bv)
        assert bk.shape[0] == 1
        assert bk.shape[-2] <= 12

    def test_layer_kv_rejects_unknown_without_crashing(self):
        from mlx_vlm.apc import layer_kv_for_apc

        class Bogus:
            keys = (1, 2, 3)  # tuple like quantized but no dequantize_for_apc
            values = (4, 5, 6)
            offset = 3

        assert layer_kv_for_apc(Bogus()) == (None, None)


# ---------------------------------------------------------------------------
# A — No B=1 special case residual in extract path
# ---------------------------------------------------------------------------


class TestAlwaysExtractSemantics:
    """BatchGenerator must not short-circuit B=1; extract works for B=1 Batch*."""

    def test_extract_b1_batch_rotating_equals_clone_after_extract(self):
        cache, _, _ = _fill_batch_rotating([0], seq_len=16)
        row = extract_prompt_cache_from_batch([cache], 0)
        assert row is not None
        cloned = _clone_prompt_cache_for_apc(row)
        assert cloned is not None

    def test_extract_b1_batch_quantized(self):
        cache, _, _ = _fill_batch_quant([0], seq_len=16)
        row = extract_prompt_cache_from_batch([cache], 0)
        assert row is not None
        assert isinstance(row[0], QuantizedKVCache)
        cloned = _clone_prompt_cache_for_apc(row)
        assert cloned is not None
        assert isinstance(cloned[0], KVCache)

    def test_snapshot_equivalent_for_b1_whether_or_not_batch(self):
        """snapshot of BatchKV B=1 matches snapshot of equivalent KVCache content."""
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = 16
        k, v = _rand_kv(batch=1, seq_len=seq_len)

        batch = BatchKVCache([0])
        batch.update_and_fetch(k, v)
        plain = KVCache()
        plain.update_and_fetch(k, v)
        mx.eval(batch.keys, plain.keys)

        snap_b = snapshot_prompt_cache_row([batch], 0)
        snap_p = snapshot_prompt_cache_row([plain], 0)
        assert snap_b is not None and snap_p is not None
        mx.eval(snap_b[0].keys, snap_p[0].keys)
        assert _max_abs_error(snap_b[0].keys, snap_p[0].keys) < 1e-5


# ---------------------------------------------------------------------------
# Integration: store_exact_cache accepts batch caches if snapshot is used
# (document expected call pattern for BatchGenerator refactor)
# ---------------------------------------------------------------------------


class TestStoreExactCallPattern:
    def test_direct_store_of_batch_rotating_without_snapshot_may_reject_or_adapt(
        self,
    ):
        """Preferred path is snapshot_prompt_cache_row first.

        Direct store_exact_cache on Batch* may either:
        - succeed by adapting internally, or
        - reject with stats — both OK as long as snapshot path works.
        The snapshot path is the supported contract.
        """
        from mlx_vlm.apc import snapshot_prompt_cache_row

        seq_len = BLOCK_SIZE
        token_ids = list(range(seq_len))
        layout = [
            _fill_batch_rotating([0], seq_len)[0],
            _fill_batch_quant([0], seq_len)[0],
        ]
        snap = snapshot_prompt_cache_row(layout, 0)
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        assert snap is not None
        assert manager.store_exact_cache(token_ids, snap) is True


# ---------------------------------------------------------------------------
# A — TurboQuant batch parity (same extract / snapshot protocol)
# ---------------------------------------------------------------------------


def _fill_batch_turbo(left_padding, seq_len, bits=4.0):
    from mlx_vlm.turboquant import BatchTurboQuantKVCache

    cache = BatchTurboQuantKVCache(list(left_padding), bits=bits)
    k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    # TurboQuant prefers float16-ish traffic; random float32 is fine.
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys)
    return cache, k, v


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

    def test_extract_respects_left_padding(self):
        cache, _, _ = _fill_batch_turbo([2, 0], seq_len=10)
        row0 = cache.extract(0)
        row1 = cache.extract(1)
        assert row0.offset == cache._idx - 2
        assert row1.offset == cache._idx

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

    def test_extract_prompt_cache_from_batch(self):
        from mlx_vlm.turboquant import TurboQuantKVCache

        turbo, _, _ = _fill_batch_turbo([0, 0], seq_len=16)
        row = extract_prompt_cache_from_batch([turbo], 1)
        assert row is not None
        assert isinstance(row[0], TurboQuantKVCache)
        cloned = _clone_prompt_cache_for_apc(row)
        assert cloned is not None
        assert isinstance(cloned[0], KVCache)

    def test_layer_kv_for_apc_batch_turbo(self):
        from mlx_vlm.apc import layer_kv_for_apc

        cache, _, _ = _fill_batch_turbo([0, 0], seq_len=12)
        k, v = layer_kv_for_apc(cache, batch_idx=1)
        mx.eval(k, v)
        assert k is not None and v is not None
        assert k.shape[0] == 1
        assert k.shape[-2] <= 12
        assert not isinstance(k, tuple)

    def test_supports_exact_apc(self):
        cache, _, _ = _fill_batch_turbo([0], seq_len=8)
        assert _cache_entry_supports_exact_apc(cache) is True
