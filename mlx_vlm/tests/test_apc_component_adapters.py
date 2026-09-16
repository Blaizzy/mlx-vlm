"""Tests for the pluggable APC cache-component adapter layer (issue #1629)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm import apc_adapters as A
from mlx_vlm.models import cache as C


@pytest.fixture(autouse=True)
def _seeded():
    mx.random.seed(0)


def test_chunked_snapshot_preserves_trimmed_offset():
    source = C.ChunkedKVCache(chunk_size=4)
    keys = mx.arange(48, dtype=mx.float32).reshape(1, 1, 6, 8)
    source.update_and_fetch(keys, keys + 1)
    source.maybe_trim_front()
    adapter = A.CheckpointAdapter()
    restored = C.ChunkedKVCache(chunk_size=4)
    adapter.restore(restored, adapter.capture(source, 6))
    assert (restored.offset, restored.start_position) == (6, 2)
    for actual, expected in zip(
        restored.update_and_fetch(keys[..., :1, :], keys[..., :1, :] + 1),
        source.update_and_fetch(keys[..., :1, :], keys[..., :1, :] + 1),
    ):
        assert bool(mx.array_equal(actual, expected))


def test_apc_mode_layouts():
    assert A.apc_mode([C.KVCache(), C.KVCache()]) == "block"
    assert A.apc_mode([C.KVCache(), C.ArraysCache(2), C.KVCache()]) == "exact"
    assert A.apc_mode([C.RotatingKVCache(max_size=64)]) == "exact"
    assert A.apc_mode([]) is None


def test_build_prefix_cache_plan():
    class _Stub:
        def make_cache(self):
            return [C.KVCache(), C.ArraysCache(2)]

    plan = A.build_prefix_cache_plan(_Stub())
    assert len(plan.components) == 2
    assert plan.restorable
    assert plan.capabilities == [A.Capability.PAGEABLE, A.Capability.CHECKPOINT]
    assert len(plan.groups) == 2
    assert plan.is_hybrid
    assert plan.strategy == "checkpoint"
    assert "PrefixCachePlan" in plan.describe()


def test_dense_plan_has_one_pageable_group():
    plan = A.build_prefix_cache_plan_from_caches([C.KVCache(), C.KVCache()])
    assert plan.restorable
    assert not plan.is_hybrid
    assert plan.strategy == "block"
    assert len(plan.groups) == 1
    assert plan.groups[0].layer_indices == (0, 1)


def _clone(c):
    et = []
    out = A.clone_cache_entry(c, min_capacity_tokens=None, eval_targets=et)
    mx.eval(et)
    return out


@pytest.mark.parametrize("length", [0, 4, 48])
def test_buffered_rotating_snapshot_continuation(length):
    source = C.BufferedRotatingKVCache(max_size=8, buffer_size=3)
    for token in range(length):
        keys = mx.full((1, 1, 1, 4), token, dtype=mx.float32)
        source.update_and_fetch(keys, keys + 1)
    restored = _clone(source)
    assert restored.meta_state == source.meta_state

    for count in (3, 12, 1):
        assert mx.array_equal(
            restored.make_mask(2, return_array=True),
            source.make_mask(2, return_array=True),
        ).item()
        keys = mx.random.normal((1, 1, count, 4))
        for actual, expected in zip(
            restored.update_and_fetch(keys, keys + 1),
            source.update_and_fetch(keys, keys + 1),
        ):
            assert mx.array_equal(actual, expected).item()
        restored.trim(1)
        source.trim(1)
        assert restored.meta_state == source.meta_state


@pytest.mark.parametrize("used_slot", [None, 0, 1])
def test_arrays_merge_preserves_optional_state(used_slot):
    rows = [C.ArraysCache(2) for _ in range(3)]
    if used_slot is not None:
        rows[0][used_slot] = mx.ones((1, 4))
        rows[2][used_slot] = mx.full((1, 4), 2.0)
    merged = A.merge_cache_entries(rows, [4, 0, 4])
    assert merged.empty() is (used_slot is None)
    if used_slot is None:
        assert merged.cache == [None, None]
        assert merged.left_padding.tolist() == [0, 0, 0]
    else:
        assert merged[1 - used_slot] is None
        assert merged[used_slot].tolist() == [[1] * 4, [0] * 4, [2] * 4]


def test_ring_sliding_clone_roundtrip():
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    ring = RingSlidingKVCache(window_size=4)
    assert A.apc_exact_eligible(ring) is True and A.apc_block_eligible(ring) is False
    ring.update_and_fetch(
        mx.random.normal((1, 2, 6, 8)), mx.random.normal((1, 2, 6, 8))
    )
    for _ in range(7):
        ring.update_and_fetch(
            mx.random.normal((1, 2, 1, 8)), mx.random.normal((1, 2, 1, 8))
        )
    mx.eval(ring.keys, ring.values)
    cl = _clone(ring)
    assert type(cl).__name__ == "RingSlidingKVCache"
    assert cl.window_size == ring.window_size and cl._ring_pos == ring._ring_pos
    assert cl.prefill_length == ring.prefill_length and cl.offset == ring.offset
    k, v = mx.random.normal((1, 2, 1, 8)), mx.random.normal((1, 2, 1, 8))
    ka, va = ring.update_and_fetch(k, v)
    kb, vb = cl.update_and_fetch(k, v)
    mx.eval(ka, va, kb, vb)
    assert bool(mx.array_equal(ka, kb)) and bool(mx.array_equal(va, vb))


def test_minimax_clone_roundtrip():
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache

    mm = MiniMaxM3KVCache()
    assert A.apc_exact_eligible(mm) is True
    mm.update_and_fetch(mx.random.normal((1, 2, 5, 8)), mx.random.normal((1, 2, 5, 8)))
    mm.update_index_and_fetch(mx.random.normal((1, 2, 5, 8)))
    mx.eval(mm.state[0], mm.index_keys)
    mc = _clone(mm)
    assert (
        type(mc).__name__ == "MiniMaxM3KVCache" and mc.index_offset == mm.index_offset
    )
    k, v = mx.random.normal((1, 2, 1, 8)), mx.random.normal((1, 2, 1, 8))
    oa, _ = mm.update_and_fetch(k, v)
    ob, _ = mc.update_and_fetch(k, v)
    mx.eval(oa, ob)
    assert bool(mx.array_equal(oa, ob))


def test_ring_sliding_batch_merge_rejects_instead_of_crashing():
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    ring = RingSlidingKVCache(window_size=4)
    ring.update_and_fetch(mx.ones((1, 2, 3, 4)), mx.ones((1, 2, 3, 4)))
    assert A.merge_cache_entries([ring], [3]) is None


def test_minimax_batch_merge_still_supported():
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache

    caches = []
    for _ in range(2):
        mm = MiniMaxM3KVCache()
        k = mx.random.normal((1, 2, 5, 8))
        v = mx.random.normal((1, 2, 5, 8))
        idx = mx.random.normal((1, 2, 5, 8))
        mx.eval(k, v, idx)
        mm.update_and_fetch(k, v)
        mm.update_index_and_fetch(idx)
        caches.append(mm)
    assert A.merge_cache_entries(caches, [5, 5]) is not None
