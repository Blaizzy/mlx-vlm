"""APC cache adapters, component layouts, and model compatibility."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_vlm.models as model_packages
from mlx_vlm import apc_adapters as A
from mlx_vlm.apc import (
    APCManager,
    _clone_prompt_cache_for_apc,
    extract_prompt_cache_from_batch,
)
from mlx_vlm.apc_adapters import (
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
    cache_memory_components,
    clone_cache_entry,
)
from mlx_vlm.models import cache as C
from mlx_vlm.models import qwen4_exp
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
    RotatingKVCache,
    SimpleKVCache,
    StaticPrefixKVCache,
)
from mlx_vlm.models.hy_v4.cache import HyV4KVCache
from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpAttention,
)
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache
from mlx_vlm.models.z1t.language import Z1TCache

# Cache adapter behavior

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


# Component layouts and snapshots


@pytest.fixture
def _seed_component_adapters():
    mx.random.seed(0)


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
def test_apc_mode_layouts():
    assert A.apc_mode([C.KVCache(), C.KVCache()]) == "block"
    assert A.apc_mode([C.KVCache(), C.ArraysCache(2), C.KVCache()]) == "exact"
    assert A.apc_mode([C.RotatingKVCache(max_size=64)]) == "exact"
    assert A.apc_mode([]) is None


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
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


@pytest.mark.usefixtures("_seed_component_adapters")
def test_ring_sliding_batch_merge_rejects_instead_of_crashing():
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    ring = RingSlidingKVCache(window_size=4)
    ring.update_and_fetch(mx.ones((1, 2, 3, 4)), mx.ones((1, 2, 3, 4)))
    assert A.merge_cache_entries([ring], [3]) is None


@pytest.mark.usefixtures("_seed_component_adapters")
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


# Weight-free model compatibility audit


def _model_source_root() -> Path:
    return Path(model_packages.__file__).resolve().parent


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _cache_factories_by_package() -> dict[str, set[str]]:
    """Statically discover cache constructors inside every ``make_cache``."""
    found: dict[str, set[str]] = {}
    for path in _model_source_root().rglob("*.py"):
        tree = ast.parse(path.read_text())
        package = path.relative_to(_model_source_root()).parts[0]
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if function.name != "make_cache":
                continue
            names = found.setdefault(package, set())
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    name = _call_name(node)
                    if name is not None and (
                        name.endswith("Cache") or name == "CacheList"
                    ):
                        names.add(name)
    return found


def _cache_samples():
    """One unpopulated instance of every cache family used by model sources."""
    return {
        "ArraysCache": ArraysCache(2),
        "CacheList": CacheList(KVCache(), ArraysCache(1)),
        "ChunkedKVCache": ChunkedKVCache(chunk_size=16),
        "HyV4KVCache": HyV4KVCache(),
        "KVCache": KVCache(),
        "MiniMaxM3KVCache": MiniMaxM3KVCache(),
        "PoolingCache": PoolingCache(ratio=2),
        "QSAKVCache": QSAKVCache(),
        "RingSlidingKVCache": RingSlidingKVCache(window_size=16),
        "RotatingKVCache": RotatingKVCache(max_size=16),
        "SimpleKVCache": SimpleKVCache(),
        "StaticPrefixKVCache": StaticPrefixKVCache(max_size=16),
        "Z1TCache": Z1TCache(),
    }


def _cache_names_in_callable(function) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (OSError, TypeError, IndentationError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None and (name.endswith("Cache") or name == "CacheList"):
                names.add(name)
    return names


def _model_cache_contract(model_cls: type) -> tuple[str, ...]:
    """Resolve a wrapper's cache factory without constructing the wrapper."""
    visited: set[type] = set()

    def visit(cls: type) -> set[str]:
        if cls in visited:
            return set()
        visited.add(cls)

        make_cache = getattr(cls, "make_cache", None)
        if callable(make_cache):
            names = _cache_names_in_callable(make_cache)
            if names:
                return names

        # Wrapper factories and LanguageModel factories often delegate to a
        # class imported as one of these conventional names.
        functions = [getattr(cls, "__init__", None), make_cache]
        for function in functions:
            namespace = getattr(function, "__globals__", {})
            for name in ("LanguageModel", "TextModel", "Model"):
                target = namespace.get(name)
                if isinstance(target, type) and target is not cls:
                    names = visit(target)
                    if names:
                        return names
        return set()

    # No custom factory means generation uses one ordinary KVCache per layer.
    return tuple(sorted(visit(model_cls) or {"KVCache"}))


def _all_generative_model_contracts(
    local_factories: dict[str, set[str]],
) -> list[tuple[str, tuple[str, ...]]]:
    contracts = []
    for info in pkgutil.iter_modules(model_packages.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"mlx_vlm.models.{info.name}")
        except ModuleNotFoundError:
            # Some Omni packages depend on a newer optional companion package
            # than the minimum version in the lockfile. Their cache factories
            # are still fully discoverable from local source, so keep them in
            # the weight-free APC matrix without importing that dependency.
            names = local_factories.get(info.name)
            if names:
                contracts.append((info.name, tuple(sorted(names))))
            continue
        model_cls = getattr(module, "Model", None)
        if model_cls is not None and callable(
            getattr(model_cls, "get_input_embeddings", None)
        ):
            names = local_factories.get(info.name) or set(
                _model_cache_contract(model_cls)
            )
            contracts.append((info.name, tuple(sorted(names))))
    return sorted(contracts)


def _populated_cache(name: str, token_count: int):
    shape = (1, 1, token_count, 4)
    if name == "HyV4KVCache":
        cache = HyV4KVCache()
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "KVCache":
        cache = KVCache()
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "RotatingKVCache":
        cache = RotatingKVCache(max_size=token_count * 2)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        cache._idx = token_count
        return cache
    if name == "ChunkedKVCache":
        cache = ChunkedKVCache(chunk_size=8)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        cache.start_position = 0
        return cache
    if name == "ArraysCache":
        cache = ArraysCache(2)
        cache.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
        return cache
    if name == "PoolingCache":
        cache = PoolingCache(ratio=2)
        cache.pooled = mx.ones((1, token_count // 2, 4))
        cache.buf_kv = mx.ones((1, 2, 4))
        cache.buf_gate = mx.ones((1, 2, 1))
        cache.remainder = 1
        return cache
    if name == "SimpleKVCache":
        cache = SimpleKVCache()
        cache.update_and_fetch(mx.ones(shape), mx.ones(shape) * 2)
        return cache
    if name == "StaticPrefixKVCache":
        cache = StaticPrefixKVCache(max_size=token_count)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "RingSlidingKVCache":
        cache = RingSlidingKVCache(window_size=token_count)
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        return cache
    if name == "MiniMaxM3KVCache":
        cache = MiniMaxM3KVCache()
        cache.update_and_fetch(mx.ones(shape), mx.ones(shape) * 2)
        cache.update_index_and_fetch(mx.ones(shape))
        return cache
    if name == "QSAKVCache":
        cache = QSAKVCache()
        cache.keys = mx.ones(shape)
        cache.values = mx.ones(shape) * 2
        cache.offset = token_count
        cache.index_keys = mx.ones((1, token_count, 4)) * 3
        cache.index_position_ids = mx.arange(token_count, dtype=mx.int64).reshape(
            1, token_count
        )
        return cache
    if name == "CacheList":
        return CacheList(
            _populated_cache("KVCache", token_count),
            _populated_cache("ArraysCache", token_count),
        )
    if name == "Z1TCache":
        cache = Z1TCache()
        cache.offset = token_count
        cache.cum_eKV = mx.ones((1, 4))
        cache.cum_eK = mx.ones((1, 4)) * 2
        cache.win_eKV = mx.ones((1, 3, 4)) * 3
        cache.win_eK = mx.ones((1, 3, 4)) * 4
        return cache
    raise AssertionError(f"No populated APC sample for {name}")


MODEL_CACHE_FACTORIES = _cache_factories_by_package()
MODEL_CACHE_CONTRACTS = _all_generative_model_contracts(MODEL_CACHE_FACTORIES)
# Synthetic cache hits depend only on the cache types, not the model name.
# Keep discovery for every model, but exercise each distinct layout once.
CACHE_CONTRACTS = sorted({names for _, names in MODEL_CACHE_CONTRACTS})


def test_all_generative_model_packages_discovered_without_weights():
    """Every discovered LM/VLM/Omni contract has local architecture source."""
    # This is intentionally a lower bound: new architectures increase it,
    # while accidentally dropping a large family makes the audit fail loudly.
    assert len(MODEL_CACHE_CONTRACTS) >= 120
    for package, _ in MODEL_CACHE_CONTRACTS:
        assert (_model_source_root() / package).is_dir()


def test_every_model_cache_factory_has_a_restorable_apc_adapter():
    """All cache types referenced by all model factories are APC-compatible."""
    by_package = MODEL_CACHE_FACTORIES
    discovered = set().union(*by_package.values())
    samples = _cache_samples()
    unknown = discovered - samples.keys()
    assert not unknown, (
        "New model cache types need an APC adapter/sample: " f"{sorted(unknown)}"
    )

    # Exercise planning and cloning, not just name registration. Empty caches
    # are sufficient because the protocol and constructor metadata are what
    # vary across architectures; populated round-trips live in the APC tests.
    for name in sorted(discovered):
        cache = samples[name]
        plan = build_prefix_cache_plan_from_caches([cache])
        assert plan.restorable, f"{name}: {plan.describe()}"
        eval_targets: list[mx.array] = []
        clone = clone_cache_entry(
            cache, min_capacity_tokens=None, eval_targets=eval_targets
        )
        assert clone is not None, f"{name} cannot be cloned for APC"
        assert all(not c.fallback for c in A.cache_memory_components([clone], 0)), name

    # Also build one heterogeneous plan per model package. This catches a
    # future combination that is individually registered but cannot be
    # coordinated as one architecture.
    for package, names in by_package.items():
        if not names:
            continue
        plan = build_prefix_cache_plan_from_caches(
            [samples[name] for name in sorted(names)]
        )
        assert plan.restorable, f"{package}: {plan.describe()}"

    assert len(by_package) >= 70


@pytest.mark.parametrize(
    "cache_names", CACHE_CONTRACTS, ids=["+".join(names) for names in CACHE_CONTRACTS]
)
def test_cache_hit_for_each_model_cache_contract(cache_names, monkeypatch):
    """A synthetic second request hits APC for every distinct cache layout."""
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    block_size = 8
    token_count = 2 * block_size
    token_ids = list(range(token_count))
    caches = [_populated_cache(name, token_count) for name in cache_names]
    plan = build_prefix_cache_plan_from_caches(caches)
    assert plan.restorable, f"{cache_names}: {plan.describe()}"

    manager = APCManager(num_blocks=8, block_size=block_size)

    class SyntheticModel:
        def make_cache(self):
            return caches

    coordinator = manager.coordinator(SyntheticModel())
    assert coordinator.strategy == plan.strategy, cache_names
    try:
        if plan.strategy == "block":
            stored = manager.store_kv_blocks(
                token_ids,
                [cache.keys for cache in caches],
                [cache.values for cache in caches],
            )
            manager.release(stored)
        else:
            assert manager.store_exact_cache(token_ids, caches), cache_names

        hit = coordinator.lookup(
            token_ids + [999],
            extra_hash=0,
            safe_lookup_min=0,
            suffix_is_text_only=lambda _prefix_len: True,
            prefix_has_media=lambda _prefix_len: False,
        )
        assert hit is not None, cache_names
        assert hit["prefix_len"] == token_count, cache_names
        stats = manager.stats_snapshot()
        if plan.strategy == "block":
            assert stats["lookups_hit"] == 1, cache_names
        else:
            assert hit["warm_cache"] is not None, cache_names
            assert stats["exact_hits"] == 1, cache_names
        coordinator.release_hit(hit)
    finally:
        manager.close()


def test_dense_models_without_make_cache_use_generation_fallback():
    """VLM language backbones without a custom factory remain pageable."""

    class DenseLanguageModel:
        layers = [object(), object(), object()]

    class VisionLanguageModel:
        language_model = DenseLanguageModel()

    plan = build_prefix_cache_plan(VisionLanguageModel())
    assert plan.restorable
    assert plan.strategy == "block"
    assert len(plan.components) == len(DenseLanguageModel.layers)
    assert len(plan.groups) == 1


@pytest.mark.parametrize(
    "base,kwargs",
    [
        (C.ConcatenateKVCache, {}),
        (C.SimpleKVCache, {}),
        (C.KVCache, {}),
        (C.QuantizedKVCache, {}),
        (C.BatchKVCache, {"left_padding": [0]}),
        (C.BatchQuantizedKVCache, {"left_padding": [0]}),
    ],
)
def test_kv_subclasses_inherit_default_memory_profile(base, kwargs):
    class CustomKV(base):
        pass

    cache = CustomKV(**kwargs)
    empty = A.cache_memory_components([cache], 0)[0]
    assert not empty.fallback and empty.footprint(6000) == 0

    keys = mx.ones((1, 1, 16, 64))
    cache.update_and_fetch(keys, keys + 1)
    profile = A.cache_memory_components([cache], 16)[0]
    assert not profile.fallback
    assert profile.source_bytes == C.cache_nbytes((cache.keys, cache.values))
    future = CustomKV(**kwargs)
    keys = mx.ones((1, 1, 6000, 64))
    future.update_and_fetch(keys, keys + 1)
    assert profile.footprint(6000) == C.cache_nbytes((future.keys, future.values))


def test_subclasses_inherit_specialized_memory_profile():
    class CustomState(C.ArraysCache):
        pass

    cache = CustomState(1)
    cache[0] = mx.ones((1, 64))
    profile = A.cache_memory_components([cache], 16)[0]
    assert not profile.fallback
    assert profile.footprint(6000) == cache.nbytes


@pytest.mark.parametrize(
    "make_cache,batch_size,mrope",
    [
        (QSAKVCache, 1, False),
        (lambda: QSAQuantizedKVCache(32, 4), 1, True),
        (lambda: BatchQSAKVCache([0, 0]), 2, True),
        (lambda: BatchQSAKVCache([0, 3]), 2, False),
    ],
)
@pytest.mark.parametrize("seed_length", [1, 16])
def test_qsa_profiles_include_indexer_and_block_growth(
    make_cache, batch_size, mrope, seed_length
):
    cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
        "cases"
    ]
    config = next(case for case in cases if case["id"] == "qwen4_exp")["config"][
        "text_config"
    ]
    # Preserve the upstream indexer's dimensions while reusing the shared config.
    config.update(
        indexer_n_heads=2,
        indexer_head_dim=8,
        indexer_compress_ratio=2,
        head_dim=8,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    indexer = Qwen4ExpAttention(qwen4_exp.TextConfig(**config)).indexer
    cache = make_cache()

    def advance(start, length):
        positions = mx.broadcast_to(
            mx.arange(start, start + length), (batch_size, length)
        )
        if mrope:
            positions = mx.broadcast_to(positions, (3, batch_size, length))
        indexer.select_from_projected(
            mx.ones((batch_size, length, 24)), cache, positions
        )
        cache.update_and_fetch(
            mx.ones((batch_size, 1, length, 32), dtype=mx.float16),
            mx.ones((batch_size, 1, length, 64), dtype=mx.float16),
        )
        mx.eval(cache.state)

    empty = cache_memory_components([cache], 0)[0]
    assert not empty.fallback and empty.footprint(6000) == 0
    advance(0, seed_length)
    cache = clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=[])
    profile = cache_memory_components([cache], seed_length, batch_size=batch_size)[0]
    assert not profile.fallback
    assert profile.source_bytes * batch_size == cache.nbytes
    for start in range(seed_length, 6000, 257):
        advance(start, min(257, 6000 - start))
    estimate = batch_size * profile.footprint(6000, 257)
    assert cache.nbytes <= estimate < 1.3 * cache.nbytes
