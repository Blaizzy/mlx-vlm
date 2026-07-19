"""Tests for the pluggable APC cache-component adapter layer (issue #1629)."""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm import apc_adapters as A
from mlx_vlm.models import cache as C


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
    # Key correctness rule (#1629): a KVCache subclass must not inherit the
    # pageable/block capability -- RingSlidingKVCache is ring-windowed.
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    ring = RingSlidingKVCache(window_size=64)
    assert A.resolve_capability(ring) == A.Capability.CHECKPOINT
    assert A.apc_block_eligible(ring) is False  # was wrongly True before
    assert A.apc_exact_eligible(ring) is True  # exact snapshot still works


def test_custom_cache_accepted_via_checkpoint():
    # MiniMaxM3KVCache is not a _BaseCache subclass but exposes state/meta_state.
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
    # snapshot is detached: mutating the source must not change the restore
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
    # the fix, exercised through apc.py's public helper:
    assert _cache_entry_supports_block_apc(RingSlidingKVCache(window_size=64)) is False
