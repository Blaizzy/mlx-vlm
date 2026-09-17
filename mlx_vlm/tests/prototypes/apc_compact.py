"""Standalone APC prototype; run explicitly instead of test_apc.py."""

import ast
import copy
import importlib
import logging
import os
import pkgutil
import shutil
import subprocess
import sys
import threading
from functools import partial
from itertools import product
from pathlib import Path
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

import mlx_vlm.models as models
from mlx_vlm import apc as P
from mlx_vlm import apc_adapters as A
from mlx_vlm.apc import harvest_blocks_from_batch_cache as harvest
from mlx_vlm.apc import make_warm_batch_exact_cache_multi as warm_exact
from mlx_vlm.apc import make_warm_batch_kv_cache_multi as warm_blocks
from mlx_vlm.apc import snapshot_prompt_cache_row as snapshot_row
from mlx_vlm.apc_adapters import cache_memory_components as memory_components
from mlx_vlm.generate.ar import PromptProcessingBatch, _extend_cache, _make_cache
from mlx_vlm.models import cache as C
from mlx_vlm.models.hy_v4.cache import HyV4KVCache
from mlx_vlm.models.minimax_m3_vl.language import (
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
)
from mlx_vlm.models.qwen4_exp.language import (
    BatchQSAKVCache,
    QSAKVCache,
    QSAQuantizedKVCache,
    Qwen4ExpAttention,
)
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache
from mlx_vlm.models.z1t.language import AFTConv, Z1TCache
from mlx_vlm.tests.test_models import DATA, build_config
from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache, _SplitCodec

parametrize = pytest.mark.parametrize


def forbid(monkeypatch, target, attr):
    monkeypatch.setattr(
        target, attr, lambda *a, **kw: pytest.fail(f"unexpected {attr}")
    )


def same_arrays(left, right):
    if isinstance(left, mx.array):
        assert (left.shape, left.dtype) == (right.shape, right.dtype)
        assert mx.array_equal(left, right).item()
    elif isinstance(left, (tuple, list, dict)):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            left, right = left.values(), right.values()
        for a, b in zip(left, right, strict=True):
            same_arrays(a, b)
    else:
        assert left == right


def same_cache(left, right):
    assert type(left) is type(right)
    if isinstance(left, (list, tuple)):
        for a, b in zip(left, right, strict=True):
            same_cache(a, b)
    else:
        same_arrays(left.state, right.state)
        same_arrays(left.meta_state, right.meta_state)
        if isinstance(left, C.CacheList):
            same_cache(left.caches, right.caches)


def kv(length=32, batch=1, heads=2, dim=32):
    pair = [mx.random.normal((batch, heads, length, dim)) for _ in range(2)]
    mx.eval(pair)
    return pair


def filled(cache, length=32, batch=1, heads=2, dim=32):
    cache.update_and_fetch(*kv(length, batch, heads, dim))
    mx.eval(cache.state)
    return cache


def allocated(length, value=1):
    cache = C.KVCache()
    cache.step = 1
    cache.keys = mx.full((1, 1, length, 4), value, dtype=mx.float32)
    cache.values = cache.keys + 1
    cache.offset = length
    mx.eval(cache.state)
    return cache


def batch_cache(kind, padding=(0,), **kwargs):
    factory, defaults = {
        "dense": (C.BatchKVCache, {}),
        "rotating": (C.BatchRotatingKVCache, {"max_size": 64}),
        "uniform": (C.BatchQuantizedKVCache, {"group_size": 32, "bits": 8}),
        "turbo": (BatchTurboQuantKVCache, {"bits": 4.0}),
    }[kind]
    return factory(left_padding=list(padding), **(defaults | kwargs))


def clone(cache):
    targets = []
    result = A.clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=targets)
    mx.eval(targets)
    return result


def cache_model(caches):
    return NS(make_cache=lambda: caches)


def coordinate(manager, caches):
    return manager.coordinator(cache_model(caches))


@pytest.fixture
def managers(tmp_path, monkeypatch):
    """Own writers and readers, including reopened namespaces and memory budgets."""
    owned = []

    def make(
        tier="memory", *, blocks=8, block=16, namespace="unit", budget=None, **settings
    ):
        disk = None
        if tier != "memory":
            disk = P.DiskBlockStore(tmp_path, namespace=namespace)
        manager = P.APCManager(num_blocks=blocks, block_size=block, disk=disk)
        if tier == "disk-only":
            manager._exact_cache_max = 0
        if budget is not None:
            manager.memory_max_bytes = budget
            manager.memory_reserve_bytes = 0
            monkeypatch.setattr(manager, "_memory_headroom", lambda: 1 << 40)
        for key, value in settings.items():
            setattr(manager, key, value)
        owned.append(manager)
        return manager

    yield make
    for manager in reversed(owned):
        manager.close()


@pytest.fixture
def memory_manager(managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "16")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    return lambda budget=4096, disk=False: managers(
        "disk" if disk else "memory", budget=budget, blocks=16
    )


@pytest.fixture
def prefix_manager(managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    return managers


def disk_roundtrip(managers, tokens, caches, **kwargs):
    writer = managers("disk", **kwargs)
    assert writer.store_exact_cache(tokens, caches)
    writer.disk.flush()
    writer.close()
    reader = managers("disk", **kwargs)
    restored, count = reader.lookup_exact_cache(tokens + [999])
    assert count == len(tokens) and restored is not None
    assert reader.stats.disk_hits == 1
    return restored, reader


def store_blocks(manager, tokens, layers=2, dim=4, extra_hash=0):
    pairs = [kv(len(tokens), heads=1, dim=dim) for _ in range(layers)]
    return manager.store_kv_blocks(
        tokens, *map(list, zip(*pairs)), extra_hash=extra_hash
    )


def apc_config(name, *, text_only=False):
    profile = DATA["apc"][name]
    cases = {c["id"]: c for c in DATA["cases"]}
    case = cases[profile["case"]] if "case" in profile else profile

    fields = copy.deepcopy(case["config"])

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(fields, profile["config"])
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    if text_only:
        return module, build_config(module, fields["text_config"], "TextConfig")
    return module, build_config(module, fields)


def language_model(name):
    module, config = apc_config(name)
    if name == "qwen3_5":
        return module.LanguageModel(config.text_config, config)
    return module.Model(config).language_model


def test_hashes_and_dependencies():
    tokens = tuple(range(16))
    hash_tokens = P._hash_tokens
    assert hash_tokens(0, tokens, 0) == hash_tokens(0, tokens, 0)
    variants = [(0, 0), (0, 42), (7, 0), (8, 0)]
    assert len({hash_tokens(seed, tokens, extra) for seed, extra in variants}) == 4
    image_hash, tenant_hash = P.hash_image_payload, P.tenant_scoped_hash
    assert image_hash(pixel_values=mx.zeros((1, 3, 8, 8))) != image_hash(
        pixel_values=mx.ones((1, 3, 8, 8))
    )
    refs = ["a.png", "b.png"]
    assert image_hash(image_ref=refs) == image_hash(image_ref=refs)
    assert image_hash(None, None) == 0
    assert P._hash_payload(None) is P._hash_payload([]) is None
    assert P._hash_payload(refs) == P._hash_payload(refs)
    assert P._hash_payload("x") == image_hash(image_ref="x")
    image = image_hash(image_ref="cat.jpg")
    assert tenant_hash(None, image) == image
    assert tenant_hash("a", image) == tenant_hash("a", image)
    variants = [("a", image), ("b", image), ("a", 42)]
    assert len({tenant_hash(t, i) for t, i in variants}) == 3
    code = "from mlx_vlm.apc import tenant_scoped_hash; print(tenant_scoped_hash('a', 123456789))"
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1]
    base = P.semantic_extra_hash(image_hash=5)
    good = NS(apc_key_dependencies=lambda: ["adapter-x"])
    broken = NS(apc_key_dependencies=lambda: (_ for _ in ()).throw(ValueError))
    assert P.semantic_extra_hash(image_hash=5, model=good) != base
    for model in (NS(), NS(apc_key_dependencies=5), broken):
        assert P.semantic_extra_hash(image_hash=5, model=model) == base
    assert P.model_key_dependencies(None, None) == ()


def test_blocks_and_statistics(managers):
    manager = managers(blocks=16)
    tokens = list(range(53))
    assert manager.lookup_prefix(tokens) == ([], 0)
    stored = store_blocks(manager, tokens)
    assert len(stored) == 3
    manager.release(stored)
    for _ in range(2):
        matched, count = manager.lookup_prefix(tokens)
        assert len(matched) == 3 and count == 48
        warm = P.make_warm_kv_cache(matched, min_capacity_tokens=65)
        assert len(warm) == 2
        assert all(
            c.offset == 48 and c.keys.shape[:2] == (1, 1) and c.keys.shape[2] >= 65
            for c in warm
        )
        manager.release(matched)
        assert manager.stats_snapshot()["lookups_hit"] == 1
        manager.reset_stats()
        assert manager.stats_snapshot()["lookups_hit"] == 0
    manager.clear()
    assert manager.stats_snapshot()["pool_used"] == 0
    assert manager.lookup_prefix(tokens) == ([], 0)


def test_layer_major_threshold(managers, monkeypatch):
    monkeypatch.setenv("APC_LAYER_MAJOR_MEMORY_MIN_TOKENS", "1")
    manager = managers(blocks=16)
    tokens = list(range(64))
    sources = [allocated(64, value) for value in (1, 3)]
    keys, values = [c.keys for c in sources], [c.values for c in sources]
    assert manager.store_kv_blocks(tokens, keys, values) == []
    assert manager.lookup_prefix(tokens)[1] == 0
    warm, count = manager.lookup_exact_cache(tokens + [999])
    assert count == 48 and len(warm) == 2
    for restored, source in zip(warm, sources):
        assert restored.offset == 48 and restored.keys.shape[2] >= 65
        same_arrays(restored.keys[..., :48, :], source.keys[..., :48, :])
        same_arrays(restored.values[..., :48, :], source.values[..., :48, :])


def test_disk_block_lifecycle(managers, monkeypatch):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "1")
    manager = managers("disk", blocks=1)
    first, second = list(range(48)), list(range(100, 148))
    manager.release(store_blocks(manager, first))
    manager.disk.flush()
    assert any(manager.disk.dir.glob(f"*{manager.disk.SUFFIX}"))
    shutil.rmtree(manager.disk.dir)
    assert not manager.disk.dir.exists()
    manager.release(store_blocks(manager, second))
    manager.disk.flush()
    size = manager.disk.disk_bytes
    assert size > 0 and manager.disk.dir.exists()
    manager.close()
    reader = managers("disk")
    warm, count = reader.lookup_prefix_disk_cache(second)
    assert count == 48 and all(c.offset == 48 for c in warm)
    assert reader.stats_snapshot()["pool_used"] == 0
    reader.disk.max_bytes = int(size * 0.75)
    assert reader.disk._maybe_evict() > 0
    warm, count = reader.lookup_prefix_disk_cache(second)
    assert warm is not None and 0 < count < 48


def test_disk_policy_and_metadata(managers, monkeypatch):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "3")
    manager = managers("disk")
    tokens = list(range(48))
    manager.release(store_blocks(manager, tokens))
    manager.disk.flush()
    assert manager.lookup_prefix_disk_cache(tokens) == (None, 0)
    warm, count = manager.lookup_prefix_disk_cache(
        tokens, allow_memory_overlap=True, max_prefix_tokens=32, min_prefix_tokens=16
    )
    assert warm is not None and count == 32
    assert manager.lookup_prefix_disk_cache(
        tokens, allow_memory_overlap=True, max_prefix_tokens=32, min_prefix_tokens=32
    ) == (None, 0)
    manager._disk_min_free_ram_bytes = 2
    monkeypatch.setattr(P, "_free_ram_bytes", lambda: 1)
    lookup = manager.lookup_prefix_disk_cache
    assert lookup(tokens, allow_memory_overlap=True) == (None, 0)
    monkeypatch.undo()
    writer = managers("disk", namespace="metadata")
    tokens = list(range(16))
    writer.release(store_blocks(writer, tokens, extra_hash=1))
    writer.disk.flush()
    writer.close()
    reader = managers("disk", namespace="metadata")
    assert reader.lookup_prefix_disk_cache(tokens, extra_hash=2) == (None, 0)
    wrong, real = [P._hash_tokens(0, tuple(tokens), extra) for extra in (2, 1)]
    reader.disk._index[wrong] = reader.disk._index[real]
    assert reader.lookup_prefix_disk_cache(tokens, extra_hash=2) == (None, 0)


def test_exact_promotion_and_priority(managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    tokens = [list(range(20)), list(range(100, 120))]
    writer = managers("disk")
    for value, ids in enumerate(tokens, 1):
        assert writer.store_exact_cache(ids, [allocated(20, value)])
    writer.disk.flush()
    writer.close()
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1")
    reader = managers("disk")
    for index, hits in [(0, 1), (0, 1), (1, 2), (0, 3)]:
        warm, count = reader.lookup_exact_cache(tokens[index] + [999])
        assert warm is not None and count == 20 and reader.stats.disk_hits == hits
    source = allocated(20, 99)
    assert reader.store_exact_cache(tokens[0], [source])
    warm, count = reader.lookup_exact_cache(tokens[0] + [999])
    assert count == 20 and reader.stats.disk_hits == 3
    assert reader.stats.exact_stores == 1
    same_arrays(warm[0].keys[..., :20, :], source.keys)


def test_self_check_and_plans(managers, monkeypatch, caplog):
    monkeypatch.delenv("APC_TRACE", raising=False)
    layouts = [
        ([C.KVCache(), C.KVCache()], "block"),
        ([C.ArraysCache(2), C.KVCache()], "exact"),
        ([C.RotatingKVCache(8), C.KVCache()], "exact"),
        ([object()], None),
    ]
    assert P.model_apc_mode(object()) == "block"
    assert A.apc_mode([]) is None
    for caches, mode in layouts:
        model = cache_model(caches)
        assert P.model_apc_mode(model) == A.apc_mode(caches) == mode
        if mode is None:
            continue
        plan = A.build_prefix_cache_plan(model)
        assert plan.restorable and len(plan.components) == 2
        assert plan.is_hybrid is (mode == "exact")
        assert plan.strategy == ("checkpoint" if mode == "exact" else "block")
        assert len(plan.groups) == (2 if mode == "exact" else 1)
        if isinstance(caches[0], C.ArraysCache):
            assert plan.capabilities == [A.Capability.CHECKPOINT, A.Capability.PAGEABLE]
        if mode == "block":
            assert plan.groups[0].layer_indices == (0, 1)
        assert "PrefixCachePlan" in plan.describe()
    plan = A.build_prefix_cache_plan(NS(language_model=NS(layers=[object()] * 3)))
    assert plan.restorable and plan.strategy == "block"
    assert len(plan.components) == 3 and len(plan.groups) == 1
    assert (
        P.classify_layer_for_apc(filled(C.QuantizedKVCache(), 8, dim=64)).status == "ok"
    )
    rejected = P.classify_layer_for_apc(object())
    assert rejected.status == "unsupported" and rejected.reason
    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        result = P.self_check_model_apc(
            cache_model([batch_cache(k) for k in ("rotating", "uniform", "dense")]),
            kv_bits=8.0,
        )
    assert result.ok and result.apc_mode == "exact"
    assert any("APC self-check ok" in r.message for r in caplog.records)
    assert not P.self_check_model_apc(object()).ok
    broken = P.self_check_model_apc(
        NS(make_cache=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    )
    assert not broken.ok and broken.notes
    monkeypatch.setenv("APC_TRACE", "1")
    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        assert not managers().store_exact_cache(
            list(range(16)), [NS(keys="bad", values="bad")]
        )
    assert any("APC_TRACE reject" in r.message for r in caplog.records)
    assert any("unclonable" in r.message for r in caplog.records)


@parametrize("kind", ["dense", "rotating", "uniform", "turbo"])
def test_batch_protocol(kind):
    cache = batch_cache(kind, [0, 0])
    if kind in ("rotating", "turbo"):
        assert cache.empty() and cache.batch_size == 2 and not cache.is_single_row()
        single = filled(batch_cache(kind), 8)
        assert not single.empty() and single.batch_size == 1 and single.is_single_row()
    if kind in ("uniform", "turbo"):
        expected = C.QuantizedKVCache if kind == "uniform" else TurboQuantKVCache
        row = cache.extract(0)
        assert isinstance(row, expected) and (row.keys is None or row.offset == 0)
        assert row.dequantize_for_apc() == (None, None)
    if kind == "turbo":
        assert cache.dequantize_for_apc() == (None, None)
        keys, values = kv(24, batch=2)
        cache.update_and_fetch(keys, values)
        row = cache.extract(1)
        assert isinstance(row, TurboQuantKVCache) and row.offset == 24
        k, v = row.dequantize_for_apc()
        assert k.shape == (1, 2, 24, 32)
        assert mx.max(mx.abs(k - keys[1:2])).item() < 2
    else:
        cache.prepare(right_padding=[2, 0], lengths=[4, 6])
        cache.update_and_fetch(*kv(3, batch=2))
        cache.filter(mx.array([1, 0], dtype=mx.int32))
        cache.filter(mx.array([1], dtype=mx.int32))
        if kind == "rotating":
            assert cache._lengths.tolist() == [4]
            assert all(x.shape[0] == 1 for x in cache.update_and_fetch(*kv(2)))
        else:
            assert cache._right_padding.tolist() == [2]
        cache.finalize()
        assert cache.offset.tolist() == [4 if kind == "rotating" else 1]
        assert cache.left_padding.tolist() == [1 if kind == "rotating" else 2]
    keys, values = P.layer_kv_for_apc(cache, batch_idx=0)
    assert keys.shape[0] == values.shape[0] == 1
    assert not isinstance(keys, tuple)
    single = filled(batch_cache("rotating"), 16)
    row = P.extract_prompt_cache_from_batch([single], 0)
    assert P._clone_prompt_cache_for_apc(row) is not None


def test_float_extraction_and_harvesting(managers):
    for cache in (C.KVCache(), C.QuantizedKVCache(32, 8)):
        keys, values = P.layer_kv_for_apc(filled(cache, 12))
        assert keys.shape == values.shape == (1, 2, 12, 32)
    bogus = NS(keys=(1, 2, 3), values=(4, 5, 6), offset=3)
    assert P.layer_kv_for_apc(bogus) == (None, None)
    source, target = managers(), managers()
    full, short = list(range(32)), list(range(100, 116))
    blocks = [store_blocks(source, ids) for ids in (full, short)]
    cache, _ = warm_blocks(
        [
            {"matched_blocks": b, "prefix_len": len(ids)}
            for b, ids in zip(blocks, (full, short))
        ],
        num_layers=2,
    )
    harvested = harvest(target, cache, batch_idx=1, full_token_ids=short)
    assert len(harvested) == 1
    same_arrays(harvested[0].keys[0], blocks[1][0].keys[0])
    matched, count = target.lookup_prefix(short)
    assert count == 16
    target.release(matched + harvested)
    source.release(blocks[0] + blocks[1])
    quantized = filled(batch_cache("uniform", [3]), 35)
    harvested = harvest(target, [quantized] * 2, full, batch_idx=0)
    assert len(harvested) == 2
    assert all(k.shape[2] == 16 for block in harvested for k in block.keys)
    target.release(harvested)
    assert harvest(target, [batch_cache("uniform")], list(range(16)), batch_idx=0) == []


def test_pooling_merge():
    rows = []
    for length in (6, 0):
        rotating, pooling = C.RotatingKVCache(16), C.PoolingCache(4)
        if length:
            keys = mx.arange(length * 3, dtype=mx.float32).reshape(1, 1, length, 3)
            rotating.update_and_fetch(keys, keys + 1)
            ready, _, _ = pooling.accumulate_windows(
                keys.reshape(1, length, 3), mx.ones((1, length, 2)), offset=0
            )
            pooling.update_and_fetch(mx.ones((1, ready.shape[1] // 4, 3)))
        rows.append(snapshot_row([C.CacheList(rotating, pooling)], 0))
    merged, count = warm_exact(rows, [6, 0])
    assert count == 6
    rotating, pooling = merged[0].caches
    assert isinstance(rotating, C.BatchRotatingKVCache)
    assert rotating.offset.tolist() == [6, 0]
    direct = A.merge_cache_entries([row[0].caches[1] for row in rows], [6, 0])
    for cache in (pooling, direct):
        assert isinstance(cache, C.BatchPoolingCache) and cache.ratio == 4
        assert (cache.remainder, cache._pool_lengths, cache._processed) == (
            [2, 0],
            [1, 0],
            [6, 0],
        )
        assert cache.extract(0).remainder == 2
        assert cache.extract(0).pooled.shape == (1, 1, 3)
        assert cache.extract(1).empty()
    empty = C.BatchPoolingCache.merge(
        [C.PoolingCache(4), C.PoolingCache(4)], prefix_lens=[0, 0]
    )
    assert empty.remainder == empty._pool_lengths == empty._processed == [0, 0]


@parametrize("slot", [None, 0, 1])
def test_optional_array_slots(slot):
    rows = [C.ArraysCache(2) for _ in range(3)]
    if slot is not None:
        rows[0][slot], rows[2][slot] = mx.ones((1, 4)), mx.full((1, 4), 2.0)
    merged = A.merge_cache_entries(rows, [4, 0, 4])
    assert merged.empty() is (slot is None)
    if slot is None:
        assert merged.cache == [None, None]
        assert merged.left_padding.tolist() == [0, 0, 0]
    else:
        assert merged[1 - slot] is None
        assert merged[slot].tolist() == [[1] * 4, [0] * 4, [2] * 4]


@parametrize(
    "kind,length",
    [("buffered", n) for n in (0, 4, 48)]
    + [("ring", 6), ("indexed", 5), ("chunked", 6)],
)
def test_clone_continuation(kind, length):
    cache = {
        "buffered": lambda: C.BufferedRotatingKVCache(8, buffer_size=3),
        "ring": lambda: RingSlidingKVCache(4),
        "indexed": MiniMaxM3KVCache,
        "chunked": lambda: C.ChunkedKVCache(4),
    }[kind]()
    if kind == "buffered":
        for token in range(length):
            keys = mx.full((1, 1, 1, 4), token, dtype=mx.float32)
            cache.update_and_fetch(keys, keys + 1)
    else:
        filled(cache, length, heads=2, dim=8)
    if kind == "ring":
        assert A.apc_exact_eligible(cache) and not A.apc_block_eligible(cache)
        for _ in range(7):
            filled(cache, 1, heads=2, dim=8)
    if kind == "indexed":
        cache.update_index_and_fetch(kv(5, dim=8)[0])
        assert A.apc_exact_eligible(cache)
        assert A.merge_cache_entries([cache, clone(cache)], [5, 5]) is not None
    if kind == "chunked":
        cache.maybe_trim_front()
        restored = C.ChunkedKVCache(4)
        A.CheckpointAdapter().restore(restored, A.CheckpointAdapter().capture(cache, 6))
        assert (restored.offset, restored.start_position) == (6, 2)
    else:
        restored = clone(cache)
        assert type(restored) is type(cache)
        assert restored.meta_state == cache.meta_state
    for size in ((3, 12, 1) if kind == "buffered" else (1,)):
        if kind == "buffered":
            same_arrays(
                restored.make_mask(2, return_array=True),
                cache.make_mask(2, return_array=True),
            )
        keys, values = kv(
            size,
            heads=1 if kind == "buffered" else 2,
            dim=4 if kind == "buffered" else 8,
        )
        same_arrays(
            restored.update_and_fetch(keys, values),
            cache.update_and_fetch(keys, values),
        )
        if kind == "buffered":
            restored.trim(1)
            cache.trim(1)
            assert restored.meta_state == cache.meta_state
    if kind == "ring":
        assert A.merge_cache_entries([cache], [cache.offset]) is None


def sample(name, length=0):
    constructors = {
        "ArraysCache": lambda: C.ArraysCache(2),
        "CacheList": lambda: C.CacheList(
            sample("KVCache", length), sample("ArraysCache", length)
        ),
        "ChunkedKVCache": lambda: C.ChunkedKVCache(8),
        "PoolingCache": lambda: C.PoolingCache(2),
        "RingSlidingKVCache": lambda: RingSlidingKVCache(max(16, length)),
        "RotatingKVCache": lambda: C.RotatingKVCache(max(16, length * 2)),
        "StaticPrefixKVCache": lambda: C.StaticPrefixKVCache(max(16, length)),
    }
    cache = constructors.get(name, getattr(C, name, globals().get(name)))()
    if not length or name == "CacheList":
        return cache
    keys = mx.ones((1, 1, length, 4))
    if name == "ArraysCache":
        cache.cache = [mx.ones((1, 2, 4)), mx.ones((1, 1, 4)) * 2]
    elif name == "PoolingCache":
        cache.pooled = mx.ones((1, length // 2, 4))
        cache.buf_kv, cache.buf_gate = mx.ones((1, 2, 4)), mx.ones((1, 2, 1))
        cache.remainder = 1
    elif name == "Z1TCache":
        cache.offset = length
        cache.cum_eKV, cache.cum_eK = mx.ones((1, 4)), mx.ones((1, 4)) * 2
        cache.win_eKV, cache.win_eK = mx.ones((1, 3, 4)) * 3, mx.ones((1, 3, 4)) * 4
    elif name in ("SimpleKVCache", "MiniMaxM3KVCache"):
        cache.update_and_fetch(keys, keys * 2)
        if name == "MiniMaxM3KVCache":
            cache.update_index_and_fetch(keys)
    else:
        cache.keys, cache.values, cache.offset = keys, keys * 2, length
        if name == "RotatingKVCache":
            cache._idx = length
        if name == "QSAKVCache":
            cache.index_keys = mx.ones((1, length, 4)) * 3
            cache.index_position_ids = mx.arange(length, dtype=mx.int64)[None]
    return cache


def cache_names(tree):
    names = {
        getattr(n.func, "id", getattr(n.func, "attr", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    return {name for name in names if name.endswith("Cache") or name == "CacheList"}


def discover_contracts():
    """Distinct local factories cover wrappers sharing imported language backbones."""
    root = Path(models.__file__).resolve().parent
    local, generative = {}, 0
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "make_cache"
            ):
                local.setdefault(path.relative_to(root).parts[0], set()).update(
                    cache_names(node)
                )

    for info in pkgutil.iter_modules(models.__path__):
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            cls = getattr(
                importlib.import_module("mlx_vlm.models." + info.name), "Model", None
            )
        except ModuleNotFoundError:
            continue
        generative += callable(getattr(cls, "get_input_embeddings", None))
    assert generative >= 120 and len(local) >= 70
    return sorted({("KVCache",)} | {tuple(sorted(v)) for v in local.values() if v})


@parametrize("names", discover_contracts(), ids=lambda names: "+".join(names))
def test_model_cache_contract(names, managers, monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    for name in names:
        cache = sample(name)
        assert A.build_prefix_cache_plan_from_caches([cache]).restorable
        restored = clone(cache)
        assert restored is not None
        assert all(not p.fallback for p in memory_components([restored], 0))
    caches = [sample(name, 16) for name in names]
    plan = A.build_prefix_cache_plan_from_caches(caches)
    assert plan.restorable, plan.describe()
    manager = managers(block=8)
    runner = coordinate(manager, caches)
    assert runner.strategy == plan.strategy
    tokens = list(range(16))
    if plan.strategy == "block":
        keys, values = [c.keys for c in caches], [c.values for c in caches]
        manager.release(manager.store_kv_blocks(tokens, keys, values))
    else:
        assert manager.store_exact_cache(tokens, caches)
    hit = runner.lookup(
        tokens + [999],
        extra_hash=0,
        safe_lookup_min=0,
        suffix_is_text_only=lambda _: True,
        prefix_has_media=lambda _: False,
    )
    assert hit is not None and hit["prefix_len"] == 16
    counter = "lookups_hit" if plan.strategy == "block" else "exact_hits"
    assert manager.stats_snapshot()[counter] == 1
    if plan.strategy != "block":
        assert hit["warm_cache"] is not None
    runner.release_hit(hit)


@parametrize("kind", ["qsa", "deepseek", "composite", "ring-indexed"])
def test_disk_custom_state(kind, managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1" if kind == "qsa" else "0")
    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")
    if kind == "qsa":
        arrays, qsa = C.ArraysCache(1), sample("QSAKVCache", 40)
        arrays[0] = mx.arange(6, dtype=mx.int64).reshape(1, 2, 3)
        qsa.index_position_ids = mx.arange(120, dtype=mx.int64).reshape(3, 1, 40)
        qsa.index_block_keys, qsa.index_block_ratio = mx.ones((1, 1, 10, 4)), 4
        caches, length = [arrays, qsa], 40
    elif kind == "deepseek":
        rotating = C.RotatingKVCache(8, keep=0)
        rotating.keys, rotating.values = mx.ones((1, 1, 8, 4), mx.bfloat16), mx.zeros(
            (1, 1, 8, 0), mx.bfloat16
        )
        rotating.offset, rotating._idx = 40, 3
        pools = [C.PoolingCache(4), C.PoolingCache(4)]
        pools[0].pooled, pools[1].pooled = mx.ones((1, 6, 4), mx.bfloat16), mx.ones(
            (1, 6, 2)
        )
        caches, length = [C.CacheList(rotating, *pools)], 40
    elif kind == "composite":
        caches, length = [
            (sample("SimpleKVCache", 12), C.SimpleKVCache()),
            C.CacheList(sample("PoolingCache", 10)),
        ], 12
    else:
        ring = RingSlidingKVCache(4)
        ring.keys, ring.values = mx.ones((1, 2, 8, 4)), mx.ones((1, 2, 8, 4)) * 2
        ring.prefill_length, ring.offset, ring._ring_pos = 4, 11, 3
        caches, length = [ring, sample("MiniMaxM3KVCache", 10)], 10
    restored, reader = disk_roundtrip(managers, list(range(length)), caches, block=4)
    same_cache(restored, caches)
    if kind == "qsa":
        assert restored[1].keys.shape[2] >= 41 and restored[1].offset == 40
        memory, count = reader.lookup_exact_cache(list(range(40)) + [998])
        assert count == 40 and memory[1].keys.shape[2] >= 41
        assert reader.stats.disk_hits == 1
        batch, count = warm_exact([restored], [40])
        assert count == 40 and isinstance(batch[1], BatchQSAKVCache)
        same_cache(batch[1].extract(0), qsa)
        path = next(iter(reader.disk._exact_index.values()))
        assert reader.disk._open_shard_header(path)[1]["c1_kind"] == "checkpoint"


@parametrize("family", ["builtin", "model", "qsa", "pooling", "minimax"])
def test_memory_growth_profiles(family):
    factories = {
        "builtin": [
            C.ConcatenateKVCache,
            C.SimpleKVCache,
            partial(C.ChunkedKVCache, 65),
            partial(C.BufferedRotatingKVCache, 65, buffer_size=300),
            partial(C.BufferedRotatingKVCache, 65, keep=1),
            partial(C.StaticPrefixKVCache, 513),
        ],
        "model": [HyV4KVCache, partial(RingSlidingKVCache, 16)],
        "qsa": [
            QSAKVCache,
            partial(QSAQuantizedKVCache, 32, 4),
            partial(BatchQSAKVCache, [0, 0]),
            partial(BatchQSAKVCache, [0, 3]),
        ],
        "pooling": [partial(C.PoolingCache, r) for r in (4, 64)]
        + [partial(C.BatchPoolingCache, r, [0, 3, 7]) for r in (4, 64)],
        "minimax": [MiniMaxM3KVCache, partial(MiniMaxM3BatchKVCache, [0, 3])],
    }[family]
    chunks = {"builtin": [37, 256], "model": [37, 256, 1024], "pooling": [37]}.get(
        family, [257]
    )
    seeds = {"builtin": [1], "qsa": [1, 16], "pooling": [1, 65]}.get(family, [16])
    for (index, factory), chunk, seed in product(enumerate(factories), chunks, seeds):
        mrope = family == "qsa" and index in (1, 2)
        cache = factory()
        batch = len(cache.left_padding) if hasattr(cache, "left_padding") else 1
        ratio = cache.ratio if family == "pooling" else 0

        def advance(start, length):
            if family == "pooling":
                ready, _, _ = cache.accumulate_windows(
                    mx.ones((batch, length, 8), mx.float16),
                    mx.ones((batch, length, 4)),
                    0,
                )
                cache.update_and_fetch(
                    mx.ones((batch, ready.shape[1] // ratio, 4), mx.float16)
                )
                return
            if family == "qsa":
                positions = mx.broadcast_to(
                    mx.arange(start, start + length), (batch, length)
                )
                if mrope:
                    positions = mx.broadcast_to(positions, (3, batch, length))
                indexer.select_from_projected(
                    mx.ones((batch, length, 24)), cache, positions
                )
            kd, vd = (32, 64) if family == "qsa" else (4, 8)
            dtype = mx.float16 if family in ("qsa", "minimax") else mx.float32
            if isinstance(cache, C.ChunkedKVCache):
                cache.maybe_trim_front()
            cache.update_and_fetch(
                mx.ones((batch, 1, length, kd), dtype),
                mx.ones((batch, 1, length, vd), dtype),
            )
            if family == "minimax":
                cache.update_index_and_fetch(mx.ones((batch, 1, length, 8)))

        if family == "qsa":
            indexer = Qwen4ExpAttention(
                apc_config("qwen4_exp", text_only=True)[1]
            ).indexer
        empty = memory_components([cache], 0)[0]
        assert not empty.fallback and empty.footprint(6000, chunk) == 0
        advance(0, seed)
        if family in ("qsa", "model") or (family == "minimax" and batch == 1):
            cache = clone(cache)
        profile = memory_components([cache], seed, batch_size=batch)[0]
        assert not profile.fallback
        if family == "pooling":
            assert profile.fixed_bytes == ratio * 32
        else:
            assert profile.source_bytes * batch == cache.nbytes
        peak = cache.nbytes
        if family == "builtin":
            assert peak <= profile.footprint(1, chunk) < peak + 24576
        for start in range(seed, 6000, chunk):
            advance(start, min(chunk, 6000 - start))
            peak = max(peak, cache.nbytes)
        estimate = batch * profile.footprint(6000, chunk)
        measured = peak if family == "builtin" else cache.nbytes
        upper = (
            measured + 24576
            if family == "builtin"
            else measured * {"qsa": 1.3, "minimax": 1.1}.get(family, 2)
        )
        assert measured <= estimate, (family, chunk, seed, batch)
        assert estimate <= upper if family in ("pooling", "model") else estimate < upper
        if family == "builtin":
            assert profile.footprint(0, chunk) == 0


@parametrize("read_only", [False, True])
def test_static_and_fixed_profiles(read_only):
    prefix = filled(C.StaticPrefixKVCache(513), 16, heads=1, dim=4)
    source = C.StaticPrefixKVCache.from_prefix(prefix) if read_only else prefix
    cache = C.StaticPrefixKVCache.from_state(source.state, source.meta_state)
    profile = memory_components([cache], 16)[0]
    assert not profile.fallback and cache.read_only == read_only
    if read_only:
        assert profile.footprint(1) == profile.footprint(6000) == cache.nbytes
    filled(cache, 1, heads=1, dim=4)
    assert cache.offset == (16 if read_only else 17)
    assert not C.StaticPrefixKVCache.from_state(
        source.state, source.meta_state[:3]
    ).read_only
    layer, cache = AFTConv(apc_config("z1t")[1]), Z1TCache()
    layer(mx.ones((1, 1, 8)), cache)
    cache = clone(cache)
    profile = memory_components([cache], 1)[0]
    assert not profile.fallback
    layer(mx.ones((1, 5999, 8)), cache)
    assert profile.footprint(1) == profile.footprint(6000) == P._cache_nbytes(cache)


@parametrize("factory", [C.KVCache, C.QuantizedKVCache, lambda: C.RotatingKVCache(512)])
def test_unused_capacity_reservation(factory, memory_manager):
    cache = factory()
    cache.step = 1
    filled(cache, 256, heads=1, dim=64)
    cache.trim(240)
    cache.step = 256
    size = P._cache_nbytes(cache)
    manager = memory_manager(budget=1 << 20)
    assert manager.store_exact_cache(list(range(16)), [cache])
    coordinate(manager, [cache]).prepare_prefill([2000] * 3)
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 6 * 2048 * (size // 256)
    assert P._cache_nbytes(cache) == size == 16 * P._cache_nbytes(cache.state)


@parametrize(
    "operation",
    ["disk_expand", "padded", "batch", "exact_load", "block_load", "memory_expand"],
)
def test_admission_before_allocation(operation, memory_manager, monkeypatch):
    disk = operation in ("disk_expand", "batch", "exact_load", "block_load")
    budgets = dict(padded=1024, exact_load=0, block_load=0)
    budget = budgets.get(operation, 4096)
    manager = memory_manager(budget=budget, disk=disk)
    tokens, source = list(range(16)), allocated(16)
    if operation in ("disk_expand", "memory_expand"):
        assert manager.store_exact_cache(tokens, [source])
        if disk:
            manager.disk.flush()
            manager = memory_manager(disk=True)
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 4096)
        attr = "prefix_cache_reserve" if disk else "_clone_prompt_cache_for_apc"
        forbid(monkeypatch, C.KVCache if disk else P, attr)
        request = tokens + [99] * (6000 if disk else 984)
        assert manager.lookup_exact_cache(request) == (None, 0)
    elif operation.endswith("load"):
        source, tokens = allocated(32), list(range(32))
        if operation == "exact_load":
            manager.store_exact_cache(tokens, [source])
            loader, lookup = "load_exact_cache", manager.lookup_exact_cache
        else:
            manager.store_kv_blocks(tokens, [source.keys], [source.values])
            loader, lookup = "load_layer_major_prefix", manager.lookup_prefix_disk_cache
        manager.disk.flush()
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
        forbid(monkeypatch, manager.disk, loader)
        assert lookup(tokens + [99]) == (None, 0)
    elif operation == "padded":
        source = allocated(256)
        source.offset = 16
        forbid(monkeypatch, P, "_clone_prompt_cache_for_apc")
        assert not manager.store_exact_cache(tokens, [source])
        assert manager.resident_bytes() == 0
    else:
        source = C.ArraysCache(1)
        source[0] = mx.ones((2, 4))
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
        forbid(monkeypatch, P, "snapshot_prompt_cache_row")
        assert not coordinate(manager, [source]).store_checkpoint(
            list(range(32)), [source], batch_idx=0
        )
    assert manager.stats.memory_skips >= 1


def test_accounting_and_eviction(memory_manager, monkeypatch):
    cache = NS(
        state={"kv": mx.ones((2, 3)), "nested": [mx.zeros((4,))]},
        meta_state={"offsets": mx.zeros((2,), dtype=mx.int32)},
        prefix_cache_snapshot=lambda: pytest.fail("cloned"),
    )
    assert P._cache_nbytes([cache]) == P._cache_nbytes([cache, cache]) == 48
    manager = memory_manager(budget=1024)
    source = allocated(32)
    runner = coordinate(manager, [source])
    leased = manager.store_kv_blocks(list(range(32)), [source.keys], [source.values])
    assert len(leased) == 2
    manager.release(leased[:1])
    manager.memory_max_bytes = 0
    runner.prepare_prefill(100000)
    assert manager.resident_bytes() == 512
    assert leased[1].ref_cnt == 1
    assert leased[1].keys is not None
    manager.release(leased[1:])
    runner.prepare_prefill(100000)
    assert manager.resident_bytes() == 0
    for base in (object, C._BaseCache):
        cache = type(
            "GrowingCache",
            (base,),
            {"state": mx.ones((1, 256, 1024)), "meta_state": ()},
        )()
        manager = memory_manager(budget=4 << 20)
        monkeypatch.setattr(P, "_clone_prompt_cache_for_apc", lambda cache: cache)
        runner = coordinate(manager, [cache])
        assert manager.store_exact_cache(list(range(18)), [cache])
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 8 << 20)
        runner.prepare_prefill(6001)
        assert manager.resident_bytes() == 0 and not manager._make_room()


def test_single_row_checkpoint(managers):
    tokens = list(range(32))
    arrays = C.ArraysCache(1)
    arrays[0] = mx.ones((1, 3, 5))
    rotating = sample("RotatingKVCache", 8)
    rotating.max_size, rotating.offset, rotating._idx = 8, 32, 4
    caches = [arrays, allocated(32), rotating]
    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.__dict__.update(
        uids=[0],
        prompt_cache=caches,
        _right_pad_per_row=None,
        _left_padding_per_row=[0],
        _suffix_lens=[32],
        _processed_prompt_columns=32,
        _apc_mode="exact",
        _apc_manager=managers(blocks=4, block=4),
        _apc_meta=[
            dict(full_input_ids=tokens, prefix_len=0, checkpoint_len=32, extra_hash=0)
        ],
    )
    assert P.extract_prompt_cache_from_batch(caches, 0) is None
    batch._store_apc_exact_checkpoints()
    assert batch._apc_meta[0]["checkpoint_done"] is True
    assert batch._apc_manager.stats.exact_stores == 1


def test_generation_stream_spill(managers, monkeypatch):
    from mlx_vlm.generate.common import generation_stream

    monkeypatch.setenv("APC_MAX_POOL_TENSORS", "1")
    with mx.stream(generation_stream):
        base = mx.arange(64, dtype=mx.float32).reshape(1, 1, 16, 4)
        keys, values = [base + 1, base + 2], [base + 3, base + 4]
    manager = managers("disk", blocks=1)
    assert manager.store_kv_blocks(list(range(16)), keys, values) == []
    manager.disk.flush()
    assert manager.disk.num_blocks_indexed == 1 and manager.disk.disk_bytes > 0


def test_long_prefix_pressure(memory_manager, monkeypatch):
    manager = memory_manager(budget=2 << 20, disk=True)
    coordinator = coordinate(manager, [C.KVCache()])
    manager.disk.queue_max_bytes = 1 << 20
    live_bytes = [0]
    monkeypatch.setattr(
        manager,
        "_memory_headroom",
        lambda: (7 << 20) - manager.resident_bytes() - live_bytes[0],
    )
    for i, length in enumerate([30_000, 30_000, 50_000, 50_000, 100_000]):
        coordinator.prepare_prefill(length)
        cache = allocated(length, i)
        live_bytes[0] = cache.nbytes
        assert manager.store_exact_cache([i] * length, [cache])
        assert manager.resident_bytes() <= manager.memory_max_bytes
        assert manager.disk.pending_bytes <= manager.disk.queue_max_bytes
        del cache
        live_bytes[0] = 0
        mx.clear_cache()
    coordinator.prepare_prefill(100_001)
    assert manager.resident_bytes() == 0
    restored, count = manager.lookup_exact_cache([4] * 100_000 + [99])
    assert count == 100_000
    assert mx.all(restored[0].state[0] == 4).item()
    assert manager.stats_snapshot()["disk_write_failures"] == 0
    assert manager.stats_snapshot()["disk_hits"] == 1


def test_disk_backpressure(tmp_path, monkeypatch):
    disk = P.DiskBlockStore(tmp_path)
    disk.queue_max_bytes = 512
    started, release, second_started = [threading.Event() for _ in range(3)]
    original = disk._write_exact_cache_snapshot

    def slow_write(path, payload):
        if payload.cache_hash == 1:
            started.set()
            assert release.wait(5)
        else:
            second_started.set()
        return original(path, payload)

    monkeypatch.setattr(disk, "_write_exact_cache_snapshot", slow_write)
    caches = [allocated(16), allocated(16, 2)]
    producer = threading.Thread(
        target=lambda: disk.save_exact_cache(2, [2] * 16, 0, caches[1:])
    )
    try:
        disk.save_exact_cache(1, [1] * 16, 0, caches[:1])
        assert started.wait(5)
        assert disk.pending_bytes == 512
        producer.start()
        assert not second_started.wait(0.05)  # Waits for disk, without queue growth.
        assert disk.pending_bytes <= disk.queue_max_bytes
    finally:
        release.set()
        if producer.ident is not None:
            producer.join(5)
        disk.flush()
        disk.close()
    assert not producer.is_alive()
    assert disk.pending_bytes == 0
    assert disk.num_exact_indexed == 2


def test_failed_spill(memory_manager, monkeypatch):
    manager = memory_manager(budget=0, disk=True)

    def fail_write(*args):
        raise OSError("full")

    monkeypatch.setattr(manager.disk, "_write_exact_cache_snapshot", fail_write)
    assert not manager.store_exact_cache(list(range(32)), [allocated(32)])
    stats = manager.stats_snapshot()
    assert stats["exact_stores"] == stats["resident_bytes"] == 0
    assert stats["disk_write_failures"] == 1
    assert not manager.disk._in_flight


@parametrize("synchronous", [True, False])
def test_oversized_write(managers, synchronous):
    disk = managers("disk").disk
    disk.max_bytes = 512
    disk.save_exact_cache(1, [1] * 32, 0, [allocated(32)], synchronous=synchronous)
    disk.flush()
    assert disk.disk_bytes <= disk.max_bytes
    assert disk.num_exact_indexed == 0 and disk.evictions == 1


@parametrize("opt_out", ["environment", "empty_path"])
def test_default_disk_opt_out(tmp_path, monkeypatch, opt_out):
    monkeypatch.setenv("MLX_VLM_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("APC_ENABLED", "1")
    if opt_out == "environment":
        monkeypatch.setenv("APC_DISK_ENABLED", "0")
        overrides = None
    else:
        overrides = {"disk_path": ""}
    manager = P.from_env(overrides=overrides)
    assert manager.disk is None
    assert not (tmp_path / "apc").exists()


@parametrize("tier", ["memory", "disk", "disk-only"])
def test_divergent_dense_prefix(prefix_manager, tier):
    stored = list(range(80))
    divergent = stored[:37] + [999, 998, 997]
    manager = prefix_manager(tier)
    source = C.KVCache()
    keys = mx.array(stored, dtype=mx.float32).reshape(1, 1, -1, 1)
    source.update_and_fetch(keys, keys + 1)
    assert manager.store_exact_cache(stored, [source], extra_hash=7)
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier)
    assert manager.lookup_exact_cache(divergent, extra_hash=8) == (None, 0)
    restored, count = manager.lookup_exact_cache(divergent, extra_hash=7)
    assert count == 32
    assert restored[0].offset == 32
    assert restored[0].state[0].flatten().tolist() == stored[:32]
    assert manager.stats_snapshot()["matched_tokens"] == 32
    if manager.disk:
        assert manager.stats_snapshot()["disk_hits"] == 1
    restored[0].update_and_fetch(mx.full((1, 1, 1, 1), -1), mx.full((1, 1, 1, 1), -2))
    extended, count = manager.lookup_exact_cache(stored + [1000], extra_hash=7)
    assert count == 80
    assert extended[0].state[0].flatten().tolist() == stored
    limited, count = manager.lookup_exact_cache(
        divergent, extra_hash=7, max_prefix_tokens=31
    )
    assert count == limited[0].offset == 16
    assert manager.lookup_exact_cache(
        divergent, extra_hash=7, min_prefix_tokens=32
    ) == (None, 0)


def test_checkpoint_schedule(prefix_manager):
    manager = prefix_manager()
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(cache_model([C.ArraysCache(1)]))
    tokens = list(range(75))
    assert coordinator.checkpoint_lengths(tokens, set()) == [64, 74]
    manager._exact_cache_max = 4
    assert coordinator.checkpoint_lengths(tokens, set()) == [32, 48, 64, 74]
    tokens[45:67] = [999] * 22
    assert coordinator.checkpoint_lengths(tokens, {999}) == [67, 74]
    manager.checkpoint_interval_tokens = 0
    assert coordinator.checkpoint_lengths(tokens, set()) == [74]


def prompt_batch(lm, manager, tokens, prefixes, caches, step=16, **kwargs):
    runner = manager.coordinator(lm)
    suffixes = [ids[n:] for ids, n in zip(tokens, prefixes)]
    padded = [ids + [0] * (max(map(len, suffixes)) - len(ids)) for ids in suffixes]
    return PromptProcessingBatch(
        model=lm,
        uids=list(range(len(tokens))),
        input_ids=suffixes,
        max_tokens=[1] * len(tokens),
        inputs_embeds=embeddings(lm, mx.array(padded)),
        prompt_kwargs={},
        warm_cache=caches,
        prefill_step_size=step,
        apc_manager=manager,
        apc_coordinator=runner,
        apc_meta=[
            dict(
                full_input_ids=ids,
                prefix_len=n,
                checkpoint_lengths=runner.checkpoint_lengths(ids, set()),
            )
            for ids, n in zip(tokens, prefixes)
        ],
        **kwargs,
    )


def finish_batch(batch, sample=None):
    while batch.needs_processing():
        assert batch.prompt_step() > 0
    batch.generate(
        sample or (lambda lp: mx.argmax(lp, axis=-1)),
        [lambda _: False] * len(batch.uids),
    )


def embeddings(lm, tokens):
    return lm.model.embed_tokens(tokens) * getattr(lm.model, "embed_scale", 1)


@parametrize("prefill_step_size", [None, 4, 16, 32])
def test_lfm_padded_prefill(prefix_manager, prefill_step_size):
    mx.random.seed(19)
    lm = language_model("lfm2")
    manager = prefix_manager()
    manager._exact_cache_max = 8
    manager.checkpoint_interval_tokens = 16
    warm_tokens = [i % 50 + 1 for i in range(71)]
    cold_tokens = [i % 30 + 51 for i in range(25)]
    seed_cache = lm.make_cache()
    lm(mx.array([warm_tokens[:64]]), cache=seed_cache)
    assert manager.store_exact_cache(warm_tokens[:64], seed_cache)
    restored, count = manager.lookup_exact_cache(warm_tokens)
    assert count == 64
    caches, _ = warm_exact([restored, lm.make_cache()], [64, 0])
    batch = prompt_batch(
        lm,
        manager,
        [warm_tokens, cold_tokens],
        [64, 0],
        caches,
        prefill_step_size,
        right_pad_per_row=[18, 0],
        suffix_lens=[7, 25],
    )
    sampled = []

    def sample(logprobs):
        sampled.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    finish_batch(batch, sample)
    for row, tokens in enumerate([warm_tokens, cold_tokens]):
        reference = lm.make_cache()
        for start in range(0, len(tokens) - 1, 4):
            lm(
                mx.array([tokens[start : min(start + 4, len(tokens) - 1)]]),
                cache=reference,
            )
        logits = lm(mx.array([tokens[-1:]]), cache=reference).logits[0, -1]
        logprobs = logits - mx.logsumexp(logits)
        assert mx.allclose(sampled[0][row], logprobs, atol=1e-4, rtol=1e-4).item()
        assert mx.argmax(sampled[0][row]).item() == mx.argmax(logits).item()
        assert mx.allclose(
            caches[0][0][row : row + 1], reference[0][0], atol=1e-4, rtol=1e-4
        ).item()


@parametrize("tier", ["memory", "disk"])
def test_diffusion_suffix(prefix_manager, tier):
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig
    from mlx_vlm.tests.test_diffusion_models import (
        FakeProcessor,
        RecordingEncoder,
        tiny_config_dict,
    )

    mx.random.seed(7)
    model = Model(ModelConfig.from_dict(tiny_config_dict()))
    recorder = RecordingEncoder(model.model.encoder)
    model.model.encoder = recorder
    settings = dict(block=2, checkpoint_interval_tokens=4, exact_cache_min_tokens=1)
    manager = prefix_manager(tier, **settings)
    tokens = list(range(2, 13))

    def generate(ids):
        return list(
            stream_generate(
                model,
                FakeProcessor(),
                "",
                input_ids=mx.array([ids]),
                _apc_manager=manager,
                max_tokens=2,
                max_denoising_steps=1,
                _apc_semantic_hash=11,
            )
        )

    generate(tokens)
    assert generate(tokens)[-1].cached_tokens == 10
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier, **settings)
    recorder.input_lengths.clear()
    assert generate(tokens[:9] + [13, 14])[-1].cached_tokens == 8
    assert recorder.input_lengths == [2, 1]


@parametrize("model_name", ["gemma4", "qwen3_5"])
@parametrize("tier", ["memory", "disk"])
@parametrize("path", ["stream", "batch"])
def test_hybrid_prefix_generation(prefix_manager, model_name, tier, path):
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.models.base import InputEmbeddingsFeatures

    mx.random.seed(13)
    lm = language_model(model_name)
    assert P.model_apc_mode(lm) == "exact"
    manager = prefix_manager(tier)
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    tokens = [i % 50 + 1 for i in range(70)] + [51, 52, 53, 54, 55]
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert boundaries == [64, 74]
    ids = mx.array([tokens])
    if path == "stream":
        wrapper = NS(
            language_model=lm,
            get_input_embeddings=lambda ids, *a, **kw: InputEmbeddingsFeatures(
                inputs_embeds=embeddings(lm, ids)
            ),
        )
        list(
            generate_step(
                ids,
                wrapper,
                None,
                None,
                max_tokens=1,
                temperature=0,
                prefill_step_size=16,
                prompt_cache=lm.make_cache(),
                prompt_cache_checkpoint_lengths=boundaries,
                prompt_cache_checkpoint=lambda n, caches: coordinator.store_checkpoint(
                    tokens[:n], caches
                ),
            )
        )
    else:
        finish_batch(prompt_batch(lm, manager, [tokens], [0], lm.make_cache()))
        lengths = sorted(len(e.token_ids) for e in manager._exact_cache.values())
        assert lengths == boundaries
    if manager.disk:
        manager.close()
        manager = prefix_manager(tier)
    divergent = tokens[:70] + [60, 61, 62, 63]
    restored, count = manager.lookup_exact_cache(divergent)
    assert count == 64
    cold_cache = lm.make_cache()
    for start in range(0, count, 16):
        lm(mx.array([divergent[start : start + 16]]), cache=cold_cache)
    suffix = mx.array([divergent[count:]])
    cold = lm(suffix, cache=cold_cache).logits
    warm = lm(suffix, cache=restored).logits
    mx.eval(cold, warm)
    assert mx.allclose(cold, warm, atol=1e-5, rtol=1e-5).item()
    assert mx.array_equal(mx.argmax(cold, axis=-1), mx.argmax(warm, axis=-1)).item()


@parametrize(
    "kind,bits,split",
    [
        ("uniform", 8, False),
        ("turbo", 4.0, False),
        ("turbo", 3.5, False),
        ("turbo", 3.5, True),
    ],
)
def test_packed_disk_and_batch(kind, bits, split, managers, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    kwargs = {"bits": bits} | ({"key_bits": bits, "value_bits": bits} if split else {})
    source = filled(batch_cache(kind, **kwargs))
    expected = source.extract(0)
    if split:
        assert all(
            isinstance(c, _SplitCodec) for c in (source.key_codec, source.value_codec)
        )
    factory = C.QuantizedKVCache if kind == "uniform" else TurboQuantKVCache
    forbid(monkeypatch, factory, "dequantize_for_apc")
    if kind == "uniform":
        forbid(monkeypatch, mx, "quantize")
    snapshot = snapshot_row([source], 0)
    assert isinstance(snapshot[0], factory)
    same_arrays(snapshot[0].state, expected.state)
    restored, _ = disk_roundtrip(managers, list(range(32)), snapshot)
    assert isinstance(restored[0], factory)
    same_arrays(restored[0].state, snapshot[0].state)
    scheme = "uniform" if kind == "uniform" else "turboquant"
    config = dict(group_size=32, scheme=scheme, **kwargs)
    warm, count = warm_exact([restored, [C.KVCache()]], [32, 0], kv_quant_config=config)
    assert count == 32 and type(warm[0]) is type(source)
    same_arrays(warm[0].extract(0).state, expected.state)
    if kind == "turbo":
        mx.eval(warm[0].update_and_fetch(*kv(1, batch=2)))
        assert warm[0]._idx == 33


def test_quantized_hybrid_snapshots(managers):
    arrays = C.ArraysCache(2)
    arrays.cache = [mx.zeros((1, 32, 32))] * 2
    arrays.left_padding = mx.array([0])
    caches = [arrays, filled(batch_cache("dense")), filled(batch_cache("uniform"))]
    assert all(A.apc_exact_eligible(c) for c in caches)
    cloned = P._clone_cache_entry_for_apc(
        caches[1], min_capacity_tokens=None, eval_targets=[]
    )
    assert isinstance(cloned, C.KVCache) and cloned.offset == 32
    cloned = P._clone_prompt_cache_for_apc(caches)
    assert [type(c) for c in cloned] == [C.ArraysCache, C.KVCache, C.QuantizedKVCache]
    manager = managers()
    assert manager.store_exact_cache(list(range(32)), caches)
    assert manager.stats.exact_stores == 1
    warm, count = manager.lookup_exact_cache(list(range(32)) + [999])
    assert count == 32 and len(warm) == 3


@parametrize("scheme", ["uniform", "turboquant"])
def test_warm_cache_quantization_policy(scheme, managers):
    bits = 8 if scheme == "uniform" else 3.5
    config = dict(bits=bits, group_size=32, scheme=scheme)
    manager = managers(blocks=32)
    tokens = list(range(32 if scheme == "uniform" else 16))
    blocks = store_blocks(manager, tokens, layers=4, dim=32)
    manager.release(blocks)
    blocks, count = manager.lookup_prefix(tokens)
    assert count == len(tokens)
    live = _make_cache(
        NS(layers=[NS()] * 4),
        [0],
        kv_bits=float(bits),
        kv_group_size=32,
        kv_quant_scheme=scheme,
    )
    for cache in live:
        filled(cache, len(tokens))
    if scheme == "uniform":
        warm, count = warm_blocks(
            [{"matched_blocks": blocks, "prefix_len": len(tokens)}, None],
            num_layers=4,
            kv_quant_config=config,
        )
        assert count == len(tokens)
        assert warm[-1].left_padding.tolist() == [0, len(tokens)]
        single = P.make_warm_kv_cache(
            blocks, kv_quant_config={"bits": 8.0, "group_size": 32.0}
        )
        assert isinstance(single[0], C.QuantizedKVCache)
        assert (single[0].bits, single[0].group_size) == (8, 32)

    else:
        warm = P.make_warm_batch_kv_cache(blocks, kv_quant_config=config)
        assert isinstance(warm[0], BatchTurboQuantKVCache)
    assert [type(c) for c in warm] == [type(c) for c in live]
    assert isinstance(warm[-1], C.BatchKVCache)
    manager.release(blocks)
    if scheme == "uniform":
        arrays = C.ArraysCache(2)
        arrays.cache = [mx.zeros((1, 16, 32))] * 2
        row = [arrays] + [filled(C.KVCache(), 16) for _ in range(3)]
        live_arrays = clone(arrays)
        live_arrays.left_padding = mx.array([0])
        live = [live_arrays] + [
            filled(
                batch_cache("uniform" if C.should_quantize_kv_layer(i, 4) else "dense"),
                16,
            )
            for i in range(1, 4)
        ]
        warm, _ = warm_exact([row], [16], kv_quant_config=config)
        assert [type(c) for c in warm] == [type(c) for c in live]
        extended = _extend_cache(live, warm)
        assert extended[1].offset.shape[0] == 2
        assert isinstance(extended[1], C.BatchQuantizedKVCache)
        assert isinstance(extended[-1], C.BatchKVCache)


def test_short_and_multimodal_prefixes(managers):
    config = NS(model_type="deepseek_v4", vision_n_layers=32, vocab_size=129280)
    assert P.multimodal_token_ids_from_config(config) == set(range(129280, 129285))
    assert (
        P.adjust_prefix_to_text_suffix_boundary(
            [1, 42, 42], desired_prefix_len=1, media_token_ids={42}, max_prefix_tokens=2
        )
        == 0
    )
    cache = C.ArraysCache(1)
    cache[0] = mx.zeros((1, 2, 1, 32))
    manager = managers()
    manager.store_exact_cache([1], [cache])
    assert manager.lookup_exact_cache(list(range(1, 400))) == (None, 0)
