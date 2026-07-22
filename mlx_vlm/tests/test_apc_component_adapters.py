"""Tests for the pluggable APC cache-component adapter layer (issue #1629)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm import apc_adapters as A
from mlx_vlm.models import cache as C


@pytest.fixture(autouse=True)
def _seeded():
    mx.random.seed(0)


def test_capability_classification():
    assert A.resolve_capability(C.KVCache()) == A.Capability.PAGEABLE
    assert A.resolve_capability(C.RotatingKVCache(max_size=64)) == A.Capability.WINDOWED
    assert A.resolve_capability(C.ArraysCache(2)) == A.Capability.CHECKPOINT
    assert (
        A.resolve_capability(C.CacheList(C.KVCache(), C.ArraysCache(2)))
        == A.Capability.COMPOSITE
    )
    assert A.resolve_capability((C.KVCache(),)) == A.Capability.COMPOSITE


def test_kvcache_subclass_is_not_pageable_by_inheritance():

    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    ring = RingSlidingKVCache(window_size=64)
    assert A.resolve_capability(ring) == A.Capability.CHECKPOINT
    assert A.apc_block_eligible(ring) is False
    assert A.apc_exact_eligible(ring) is True


def test_custom_cache_accepted_via_checkpoint():

    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache

    assert A.resolve_capability(MiniMaxM3KVCache()) == A.Capability.CHECKPOINT


def test_checkpoint_adapter_roundtrip_and_detach():
    kv = C.KVCache()
    kv.update_and_fetch(mx.random.normal((1, 2, 5, 4)), mx.random.normal((1, 2, 5, 4)))
    mx.eval(kv.state)
    adapter = A.CheckpointAdapter()
    frag = adapter.capture(kv, prefix_len=5)
    fresh = C.KVCache()
    adapter.restore(fresh, frag)
    for a, b in zip(kv.state, fresh.state):
        assert bool(mx.array_equal(a, b))
    assert kv.offset == fresh.offset

    kv.update_and_fetch(mx.random.normal((1, 2, 1, 4)), mx.random.normal((1, 2, 1, 4)))
    mx.eval(kv.state, fresh.state)
    assert fresh.offset == 5


def test_arrays_cache_roundtrip():
    ac = C.ArraysCache(3)
    ac.cache = [mx.random.normal((2, 5)), None, mx.random.normal((4,))]
    mx.eval([x for x in ac.cache if x is not None])
    adapter = A.CheckpointAdapter()
    fresh = C.ArraysCache(3)
    adapter.restore(fresh, adapter.capture(ac, 5))
    for a, b in zip(ac.cache, fresh.cache):
        assert (a is None and b is None) or bool(mx.array_equal(a, b))


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
    assert "PrefixCachePlan" in plan.describe()


def test_apc_py_delegates_to_registry():
    from mlx_vlm.apc import (
        _cache_entry_supports_block_apc,
        _cache_entry_supports_exact_apc,
    )
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    assert _cache_entry_supports_block_apc(C.KVCache()) is True
    assert _cache_entry_supports_exact_apc(C.ArraysCache(2)) is True

    assert _cache_entry_supports_block_apc(RingSlidingKVCache(window_size=64)) is False


def _clone(c):
    et = []
    out = A.clone_cache_entry(c, min_capacity_tokens=None, eval_targets=et)
    mx.eval(et)
    return out


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
