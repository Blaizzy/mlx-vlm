"""APC lookup, cache adapters, prefix reuse, memory budgets, and persistence."""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import logging
import os
import pkgutil
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List

import mlx.core as mx
import numpy as np
import pytest

import mlx_vlm.models as model_packages
from mlx_vlm import apc
from mlx_vlm import apc as apc_module
from mlx_vlm import apc_adapters as A
from mlx_vlm.apc import (
    APCManager,
    DiskBlockStore,
    _cache_nbytes,
    _clone_cache_entry_for_apc,
    _clone_prompt_cache_for_apc,
    _hash_payload,
    _hash_tokens,
    classify_layer_for_apc,
    extract_prompt_cache_from_batch,
    from_env,
    harvest_blocks_from_batch_cache,
    hash_image_payload,
    make_warm_batch_exact_cache_multi,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
    make_warm_kv_cache,
    model_apc_mode,
    model_key_dependencies,
    self_check_model_apc,
    semantic_extra_hash,
    snapshot_prompt_cache_row,
    tenant_scoped_hash,
)
from mlx_vlm.apc_adapters import (
    apc_exact_eligible,
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
    cache_memory_components,
    clone_cache_entry,
)
from mlx_vlm.apc_storage import KVBlockHandle
from mlx_vlm.generate.ar import _extend_cache, _make_cache
from mlx_vlm.models import cache as C
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    BufferedRotatingKVCache,
    CacheList,
    ChunkedKVCache,
    ConcatenateKVCache,
    KVCache,
    PoolingCache,
    QuantizedKVCache,
    RotatingKVCache,
    SimpleKVCache,
    StaticPrefixKVCache,
    _BaseCache,
    should_quantize_kv_layer,
)
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
from mlx_vlm.tests.test_models import DATA as MODEL_CASES
from mlx_vlm.tests.test_models import build_config
from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache


def _make_fake_kv(
    num_layers: int = 2, n_kv_heads: int = 1, seq_len: int = 32, head_dim: int = 4
) -> tuple[list[mx.array], list[mx.array]]:
    keys: list[mx.array] = []
    values: list[mx.array] = []
    for layer_idx in range(num_layers):
        base = np.arange(n_kv_heads * seq_len * head_dim, dtype=np.float32)
        base = base.reshape(1, n_kv_heads, seq_len, head_dim)
        keys.append(mx.array(base + layer_idx * 1000))
        values.append(mx.array(base + layer_idx * 1000 + 100))
    mx.eval(keys + values)
    return keys, values


def _assert_allclose(a: mx.array, b: mx.array) -> None:
    assert bool(mx.allclose(a, b).item())


def test_hash_chain_and_image_hash_are_deterministic():
    assert _hash_tokens(0, tuple(range(16)), 0) == _hash_tokens(0, tuple(range(16)), 0)
    assert _hash_tokens(0, tuple(range(16)), 0) != _hash_tokens(0, tuple(range(16)), 42)
    assert _hash_tokens(7, tuple(range(16)), 0) != _hash_tokens(8, tuple(range(16)), 0)

    zeros = mx.zeros((1, 3, 8, 8))
    ones = mx.ones((1, 3, 8, 8))
    assert hash_image_payload(pixel_values=zeros) != hash_image_payload(
        pixel_values=ones
    )
    assert hash_image_payload(None, None) == 0
    assert hash_image_payload(image_ref=["a.png", "b.png"]) == hash_image_payload(
        image_ref=["a.png", "b.png"]
    )


def test_tenant_scoped_hash_is_stable_namespaced_and_process_stable():
    image_hash = hash_image_payload(image_ref="cat.jpg")

    assert tenant_scoped_hash(None, image_hash) == image_hash
    assert tenant_scoped_hash("tenant-a", image_hash) == tenant_scoped_hash(
        "tenant-a", image_hash
    )
    assert tenant_scoped_hash("tenant-a", image_hash) != tenant_scoped_hash(
        "tenant-b", image_hash
    )
    assert tenant_scoped_hash("tenant-a", image_hash) != tenant_scoped_hash(
        "tenant-a", hash_image_payload(image_ref="dog.jpg")
    )

    code = (
        "from mlx_vlm.apc import tenant_scoped_hash; "
        "print(tenant_scoped_hash('tenant-a', 123456789))"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "1"}
    env_b = {**os.environ, "PYTHONHASHSEED": "2"}
    got_a = subprocess.check_output([sys.executable, "-c", code], env=env_a, text=True)
    got_b = subprocess.check_output([sys.executable, "-c", code], env=env_b, text=True)
    assert got_a == got_b


def test_store_lookup_warm_cache_shapes_and_partial_block_ignored():
    block_size = 16
    manager = APCManager(num_blocks=16, block_size=block_size)
    token_ids = list(range(3 * block_size + 5))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    matched, matched_tokens = manager.lookup_prefix(token_ids)
    assert matched == []
    assert matched_tokens == 0

    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    assert len(stored) == 3
    manager.release(stored)

    matched, matched_tokens = manager.lookup_prefix(token_ids)
    assert len(matched) == 3
    assert matched_tokens == 3 * block_size

    warm = make_warm_kv_cache(matched, min_capacity_tokens=3 * block_size + 17)
    assert len(warm) == len(layer_keys)
    assert all(c.offset == 3 * block_size for c in warm)
    assert all(c.keys.shape[:2] == (1, 1) for c in warm)
    assert all(c.keys.shape[2] >= 3 * block_size + 17 for c in warm)
    manager.release(matched)


def test_layer_major_memory_threshold_skips_block_pool(monkeypatch):
    monkeypatch.setenv("APC_LAYER_MAJOR_MEMORY_MIN_TOKENS", "1")
    block_size = 16
    manager = APCManager(num_blocks=16, block_size=block_size)
    token_ids = list(range(4 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)

    assert stored == []
    assert manager.lookup_prefix(token_ids)[1] == 0
    warm, matched_tokens = manager.lookup_exact_cache(token_ids + [999])
    expected_tokens = len(token_ids) - block_size
    assert matched_tokens == expected_tokens
    assert warm is not None
    assert len(warm) == len(layer_keys)
    assert warm[0].offset == expected_tokens
    assert warm[0].keys.shape[2] >= len(token_ids) + 1
    _assert_allclose(
        warm[0].keys[..., :expected_tokens, :], layer_keys[0][..., :expected_tokens, :]
    )
    _assert_allclose(
        warm[1].values[..., :expected_tokens, :],
        layer_values[1][..., :expected_tokens, :],
    )


def test_single_row_prompt_batch_exact_checkpoint_stores_without_extract():
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    token_ids = list(range(32))
    arrays = ArraysCache(size=1)
    arrays[0] = mx.ones((1, 3, 5))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 4))
    kv.values = mx.ones((1, 1, len(token_ids), 4)) * 2
    kv.offset = len(token_ids)
    rotating = RotatingKVCache(max_size=8, keep=0)
    rotating.keys = mx.ones((1, 1, 8, 4)) * 3
    rotating.values = mx.ones((1, 1, 8, 4)) * 4
    rotating.offset = len(token_ids)
    rotating._idx = 4

    batch = PromptProcessingBatch.__new__(PromptProcessingBatch)
    batch.uids = [0]
    batch.prompt_cache = [arrays, kv, rotating]
    batch._right_pad_per_row = None
    batch._left_padding_per_row = [0]
    batch._suffix_lens = [len(token_ids)]
    batch._processed_prompt_columns = len(token_ids)
    batch._apc_mode = "exact"
    batch._apc_manager = APCManager(num_blocks=4, block_size=4)
    batch._apc_meta = [
        {
            "full_input_ids": token_ids,
            "prefix_len": 0,
            "checkpoint_len": len(token_ids),
            "extra_hash": 0,
        }
    ]

    assert extract_prompt_cache_from_batch(batch.prompt_cache, 0) is None

    batch._store_apc_exact_checkpoints()

    assert batch._apc_meta[0]["checkpoint_done"] is True
    assert batch._apc_manager.stats_snapshot()["exact_stores"] == 1


def test_disk_writer_materializes_generation_stream_cache_on_producer(
    tmp_path, monkeypatch
):
    from mlx_vlm.generate.common import generation_stream

    monkeypatch.setenv("APC_MAX_POOL_TENSORS", "1")
    block_size = 16
    token_ids = list(range(block_size))
    with mx.stream(generation_stream):
        base = mx.arange(block_size * 4, dtype=mx.float32).reshape(1, 1, block_size, 4)
        layer_keys = [base + 1, base + 2]
        layer_values = [base + 3, base + 4]

    disk = DiskBlockStore(tmp_path, namespace="generation-stream")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)
    assert manager.store_kv_blocks(token_ids, layer_keys, layer_values) == []
    disk._q.join()

    assert disk.num_blocks_indexed == 1
    assert disk.disk_bytes > 0
    manager.close()


def test_disk_store_recovers_when_cache_dir_is_deleted(tmp_path):
    block_size = 16
    first_tokens = list(range(block_size))
    second_tokens = list(range(100, 100 + block_size))
    first_keys, first_values = _make_fake_kv(num_layers=2, seq_len=len(first_tokens))
    second_keys, second_values = _make_fake_kv(num_layers=2, seq_len=len(second_tokens))

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)

    stored = manager.store_kv_blocks(first_tokens, first_keys, first_values)
    manager.release(stored)
    disk._q.join()
    assert disk.dir.exists()
    assert any(disk.dir.glob(f"*{disk.SUFFIX}"))

    shutil.rmtree(disk.dir)
    assert not disk.dir.exists()

    stored = manager.store_kv_blocks(second_tokens, second_keys, second_values)
    manager.release(stored)
    disk._q.join()

    assert disk.dir.exists()
    assert any(disk.dir.glob(f"*{disk.SUFFIX}"))
    assert disk.disk_bytes > 0
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)
    warm, matched_tokens = manager.lookup_prefix_disk_cache(second_tokens)

    assert warm is not None
    assert matched_tokens == len(second_tokens)
    manager.close()


def test_clear_and_reset_stats_keep_cache_semantics():
    block_size = 16
    manager = APCManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=block_size)

    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    manager.release(stored)

    matched, matched_tokens = manager.lookup_prefix(token_ids)
    assert matched_tokens == block_size
    manager.release(matched)
    assert manager.stats_snapshot()["lookups_hit"] == 1

    manager.reset_stats()
    assert manager.stats_snapshot()["lookups_hit"] == 0
    matched, matched_tokens = manager.lookup_prefix(token_ids)
    assert matched_tokens == block_size
    manager.release(matched)
    assert manager.stats_snapshot()["lookups_hit"] == 1

    manager.clear()
    assert manager.stats_snapshot()["lookups_hit"] == 0
    assert manager.stats_snapshot()["pool_used"] == 0
    matched, matched_tokens = manager.lookup_prefix(token_ids)
    assert matched == []
    assert matched_tokens == 0


def test_lookup_prefix_disk_cache_policy_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "3")
    block_size = 16
    token_ids = list(range(3 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=8, block_size=block_size, disk=disk)
    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    manager.release(stored)
    disk._q.join()

    warm, matched_tokens = manager.lookup_prefix_disk_cache(token_ids)
    assert warm is None
    assert matched_tokens == 0

    warm, matched_tokens = manager.lookup_prefix_disk_cache(
        token_ids,
        allow_memory_overlap=True,
        max_prefix_tokens=2 * block_size,
        min_prefix_tokens=block_size,
    )
    assert warm is not None
    assert matched_tokens == 2 * block_size

    warm, matched_tokens = manager.lookup_prefix_disk_cache(
        token_ids,
        allow_memory_overlap=True,
        max_prefix_tokens=2 * block_size,
        min_prefix_tokens=2 * block_size,
    )
    assert warm is None
    assert matched_tokens == 0

    manager._disk_min_free_ram_bytes = 2
    monkeypatch.setattr(apc_module, "_free_ram_bytes", lambda: 1)
    warm, matched_tokens = manager.lookup_prefix_disk_cache(
        token_ids, allow_memory_overlap=True
    )
    assert warm is None
    assert matched_tokens == 0
    manager.close()


def test_exact_cache_disk_restore_preserves_qsa_state(tmp_path, monkeypatch):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1")

    token_ids = list(range(40))
    arrays = ArraysCache(size=1)
    arrays[0] = mx.arange(6, dtype=mx.int64).reshape(1, 2, 3)
    qsa = QSAKVCache()
    qsa.keys = mx.arange(1 * 2 * len(token_ids) * 4, dtype=mx.float32).reshape(
        1, 2, len(token_ids), 4
    )
    qsa.values = qsa.keys + 1000
    qsa.offset = len(token_ids)
    qsa.index_keys = mx.arange(1 * len(token_ids) * 6, dtype=mx.float32).reshape(
        1, len(token_ids), 6
    )
    qsa.index_position_ids = mx.arange(3 * len(token_ids), dtype=mx.int64).reshape(
        3, 1, len(token_ids)
    )
    qsa.index_block_keys = mx.arange(1 * 1 * 10 * 6, dtype=mx.float32).reshape(
        1, 1, 10, 6
    )
    qsa.index_block_ratio = 4
    mx.eval(
        arrays[0],
        qsa.keys,
        qsa.values,
        qsa.index_keys,
        qsa.index_position_ids,
        qsa.index_block_keys,
    )

    disk = DiskBlockStore(tmp_path, namespace="qsa-exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [arrays, qsa], extra_hash=19)
    disk._q.join()
    snapshot_path = next(iter(disk._exact_index.values()))
    _, metadata, _ = disk._open_shard_header(snapshot_path)
    assert metadata["c1_kind"] == "checkpoint"
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="qsa-exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    warm, matched_tokens = manager.lookup_exact_cache(token_ids + [999], extra_hash=19)

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert manager.stats_snapshot()["disk_hits"] == 1
    assert isinstance(warm[1], QSAKVCache)
    assert warm[1].offset == len(token_ids)
    assert warm[1].keys.shape[2] >= len(token_ids) + 1
    _assert_allclose(warm[1].keys[..., : len(token_ids), :], qsa.keys)
    _assert_allclose(warm[1].values[..., : len(token_ids), :], qsa.values)
    _assert_allclose(warm[1].index_keys, qsa.index_keys)
    assert warm[1].index_position_ids.dtype == mx.int64
    assert bool(
        mx.array_equal(warm[1].index_position_ids, qsa.index_position_ids).item()
    )
    _assert_allclose(warm[1].index_block_keys, qsa.index_block_keys)
    assert warm[1].index_block_ratio == qsa.index_block_ratio

    memory_warm, memory_matched_tokens = manager.lookup_exact_cache(
        token_ids + [998], extra_hash=19
    )
    assert memory_matched_tokens == len(token_ids)
    assert memory_warm is not None
    assert memory_warm[1].keys.shape[2] >= len(token_ids) + 1
    assert manager.stats_snapshot()["disk_hits"] == 1

    batch_cache, max_prefix = make_warm_batch_exact_cache_multi(
        [warm], [len(token_ids)]
    )
    assert max_prefix == len(token_ids)
    assert batch_cache is not None
    assert isinstance(batch_cache[1], BatchQSAKVCache)
    extracted = batch_cache[1].extract(0)
    _assert_allclose(extracted.index_keys, qsa.index_keys)
    _assert_allclose(extracted.index_block_keys, qsa.index_block_keys)
    assert extracted.index_block_ratio == qsa.index_block_ratio
    assert bool(
        mx.array_equal(extracted.index_position_ids, qsa.index_position_ids).item()
    )
    manager.close()


def test_exact_cache_disk_restore_preserves_deepseek_v4_empty_values(
    tmp_path, monkeypatch
):
    """DeepSeek V4's K-only local cache persists its zero-width V tensor."""
    from mlx_vlm.models.cache import CacheList, PoolingCache, RotatingKVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")

    token_ids = list(range(40))
    rotating = RotatingKVCache(max_size=8, keep=0)
    rotating.keys = (
        mx.arange(1 * 1 * 8 * 4, dtype=mx.float32)
        .reshape(1, 1, 8, 4)
        .astype(mx.bfloat16)
    )
    rotating.values = mx.zeros((1, 1, 8, 0), dtype=mx.bfloat16)
    rotating.offset = len(token_ids)
    rotating._idx = 3

    pooled = PoolingCache(ratio=4)
    pooled.pooled = mx.ones((1, 6, 4), dtype=mx.bfloat16)
    index = PoolingCache(ratio=4)
    index.pooled = mx.ones((1, 6, 2), dtype=mx.float32)
    cache = CacheList(rotating, pooled, index)
    mx.eval(rotating.keys, rotating.values, pooled.pooled, index.pooled)

    disk = DiskBlockStore(tmp_path, namespace="deepseek-v4-empty-values")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [cache], extra_hash=19)
    disk._q.join()
    stats = manager.stats_snapshot()
    assert stats["disk_writes"] == 1
    assert stats["disk_write_failures"] == 0
    assert stats["disk_exact_indexed"] == 1
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="deepseek-v4-empty-values")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    warm, matched_tokens = manager.lookup_exact_cache(token_ids + [999], extra_hash=19)

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert manager.stats_snapshot()["disk_hits"] == 1
    restored = warm[0]
    assert isinstance(restored, CacheList)
    restored_rotating = restored.caches[0]
    assert restored_rotating.values.shape == (1, 1, 8, 0)
    assert restored_rotating.values.dtype == mx.bfloat16
    _assert_allclose(restored_rotating.keys, rotating.keys)
    _assert_allclose(restored.caches[1].pooled, pooled.pooled)
    _assert_allclose(restored.caches[2].pooled, index.pooled)
    manager.close()


def test_model_apc_mode_distinguishes_block_and_exact_custom_cache():
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    assert model_apc_mode(object()) == "block"

    class KVOnly:
        def make_cache(self):
            return [KVCache(), KVCache()]

    class Mixed:
        def make_cache(self):
            return [ArraysCache(size=2), KVCache()]

    class SlidingMixed:
        def make_cache(self):
            return [RotatingKVCache(max_size=8), KVCache()]

    class Unsupported:
        def make_cache(self):
            return [object()]

    assert model_apc_mode(KVOnly()) == "block"
    assert model_apc_mode(Mixed()) == "exact"
    assert model_apc_mode(SlidingMixed()) == "exact"
    assert model_apc_mode(Unsupported()) is None


def test_disk_restore_rebuilds_index_and_segment_eviction_preserves_prefix(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "1")
    block_size = 16
    token_ids = list(range(3 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)
    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    manager.release(stored)
    disk._q.join()
    before_bytes = disk.disk_bytes
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=8, block_size=block_size, disk=disk)
    warm, matched_tokens = manager.lookup_prefix_disk_cache(token_ids)
    assert warm is not None
    assert matched_tokens == len(token_ids)
    assert all(c.offset == len(token_ids) for c in warm)
    assert manager.stats_snapshot()["pool_used"] == 0

    disk.max_bytes = int(before_bytes * 0.75)
    assert disk._maybe_evict() > 0
    warm_after_evict, matched_after_evict = manager.lookup_prefix_disk_cache(token_ids)
    assert warm_after_evict is not None
    assert 0 < matched_after_evict < len(token_ids)
    manager.close()


def test_harvest_blocks_from_batch_cache_drops_left_padding():
    block_size = 16
    source_manager = APCManager(num_blocks=8, block_size=block_size)
    harvest_manager = APCManager(num_blocks=8, block_size=block_size)
    full_token_ids = list(range(2 * block_size))
    short_token_ids = list(range(100, 100 + block_size))
    full_keys, full_values = _make_fake_kv(seq_len=len(full_token_ids))
    short_keys, short_values = _make_fake_kv(seq_len=len(short_token_ids))
    full_blocks = source_manager.store_kv_blocks(full_token_ids, full_keys, full_values)
    short_blocks = source_manager.store_kv_blocks(
        short_token_ids, short_keys, short_values
    )
    caches, _ = make_warm_batch_kv_cache_multi(
        [
            {"matched_blocks": full_blocks, "prefix_len": 2 * block_size},
            {"matched_blocks": short_blocks, "prefix_len": block_size},
        ],
        num_layers=2,
    )

    harvested = harvest_blocks_from_batch_cache(
        harvest_manager, caches, batch_idx=1, full_token_ids=short_token_ids
    )

    assert len(harvested) == 1
    _assert_allclose(harvested[0].keys[0], short_blocks[0].keys[0])
    matched, matched_tokens = harvest_manager.lookup_prefix(short_token_ids)
    assert matched_tokens == block_size
    harvest_manager.release(matched + harvested)
    source_manager.release(full_blocks + short_blocks)


def test_disk_metadata_mismatch_is_a_miss(tmp_path):
    block_size = 16
    token_ids = list(range(block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=block_size)

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)
    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values, extra_hash=1)
    manager.release(stored)
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=1, block_size=block_size, disk=disk)
    warm, matched_tokens = manager.lookup_prefix_disk_cache(token_ids, extra_hash=2)

    assert warm is None
    assert matched_tokens == 0

    wrong_hash = _hash_tokens(0, tuple(token_ids), 2)
    real_hash = _hash_tokens(0, tuple(token_ids), 1)
    disk._index[wrong_hash] = disk._index[real_hash]
    warm, matched_tokens = manager.lookup_prefix_disk_cache(token_ids, extra_hash=2)
    assert warm is None
    assert matched_tokens == 0
    manager.close()


def test_deepseek_v4_multimodal_token_ids_cover_all_sentinels():
    config = SimpleNamespace(
        model_type="deepseek_v4", vision_n_layers=32, vocab_size=129280
    )

    assert apc_module.multimodal_token_ids_from_config(config) == set(
        range(129280, 129285)
    )


def test_adjust_prefix_returns_zero_when_no_text_suffix_remains():
    token_ids = [1, 42, 42]

    assert (
        apc_module.adjust_prefix_to_text_suffix_boundary(
            token_ids,
            desired_prefix_len=1,
            media_token_ids={42},
            max_prefix_tokens=len(token_ids) - 1,
        )
        == 0
    )


def test_exact_disk_hit_promotion_lru_eviction(tmp_path, monkeypatch):
    """When _exact_cache_max=1 and a second distinct prefix is promoted, the
    first promoted entry is evicted from memory and subsequent requests for it
    go back to disk."""
    from mlx_vlm.models.cache import KVCache

    def _make_kv(val, n):
        kv = KVCache()
        kv.keys = mx.full((1, 1, n, 2), float(val))
        kv.values = mx.full((1, 1, n, 2), float(val) + 1)
        kv.offset = n
        return kv

    token_ids_a = list(range(20))
    token_ids_b = list(range(100, 120))

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="lru-evict")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids_a, [_make_kv(1, 20)], extra_hash=0)
    assert manager.store_exact_cache(token_ids_b, [_make_kv(2, 20)], extra_hash=0)
    disk._q.join()
    manager.close()

    # Restart with memory capacity = 1.
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "1")
    disk = DiskBlockStore(tmp_path, namespace="lru-evict")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)

    # Disk hit A -> promoted to memory (sole slot).
    warm_a, _ = manager.lookup_exact_cache(token_ids_a + [999], extra_hash=0)
    assert warm_a is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Memory hit A -> disk_hits unchanged.
    warm_a2, _ = manager.lookup_exact_cache(token_ids_a + [999], extra_hash=0)
    assert warm_a2 is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Disk hit B -> promoted, evicts A from the single memory slot.
    warm_b, _ = manager.lookup_exact_cache(token_ids_b + [999], extra_hash=0)
    assert warm_b is not None
    assert manager.stats_snapshot()["disk_hits"] == 2

    # A is now evicted; its next lookup must hit disk again.
    warm_a3, _ = manager.lookup_exact_cache(token_ids_a + [999], extra_hash=0)
    assert warm_a3 is not None
    assert manager.stats_snapshot()["disk_hits"] == 3

    manager.close()


def test_exact_lookup_memory_takes_priority_over_disk(tmp_path, monkeypatch):
    """Memory entries take priority over the disk store.  When the same prefix
    exists in both _exact_cache and on disk, the memory clone is returned and
    disk_hits stays at zero.  This also verifies that the promotion guard
    (skip insert if key already present) is implicitly exercised: because
    store_exact_cache writes to both memory and disk, any subsequent lookup
    hits memory first and never triggers a disk read."""
    from mlx_vlm.models.cache import KVCache

    token_ids = list(range(30))

    def _make_kv(val):
        kv = KVCache()
        kv.keys = mx.ones((1, 1, len(token_ids), 2)) * val
        kv.values = mx.ones((1, 1, len(token_ids), 2)) * (val + 1)
        kv.offset = len(token_ids)
        mx.eval(kv.keys, kv.values)
        return kv

    # Seed disk-only with value 7.
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="priority")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [_make_kv(7)], extra_hash=0)
    disk._q.join()
    manager.close()

    # Restart with memory enabled; store an in-memory entry with value 99.
    # store_exact_cache also writes to disk, but the memory lookup runs first.
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
    disk = DiskBlockStore(tmp_path, namespace="priority")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    kv_mem = _make_kv(99)
    manager.store_exact_cache(token_ids, [kv_mem], extra_hash=0)
    assert manager.stats_snapshot()["exact_stores"] == 1

    # Lookup must come from memory (no disk hit).
    warm, matched = manager.lookup_exact_cache(token_ids + [999], extra_hash=0)
    assert matched == len(token_ids)
    assert warm is not None
    snap = manager.stats_snapshot()
    assert snap["disk_hits"] == 0
    assert snap["exact_hits"] == 1
    # Value should be 99 (memory), not 7 (disk).
    _assert_allclose(warm[0].keys[..., : len(token_ids), :], kv_mem.keys)

    manager.close()


def test_exact_disk_roundtrip_generic_composite_cache(tmp_path, monkeypatch):
    """Checkpoint serialization covers in-tree composite/custom cache leaves."""
    from mlx_vlm.models.cache import CacheList, PoolingCache, SimpleKVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")
    token_ids = list(range(12))

    simple = SimpleKVCache()
    simple.update_and_fetch(mx.ones((1, 2, 12, 4)), mx.ones((1, 2, 12, 4)) * 2)

    pooling = PoolingCache(ratio=2)
    pooling.pooled = mx.ones((1, 5, 4)) * 3
    pooling.buf_kv = mx.ones((1, 2, 4)) * 4
    pooling.buf_gate = mx.ones((1, 2, 1)) * 5
    pooling.remainder = 1
    mx.eval(simple.keys, simple.values, pooling.state)

    disk = DiskBlockStore(tmp_path, namespace="generic-checkpoint")
    manager = APCManager(num_blocks=4, block_size=4, disk=disk)
    assert manager.store_exact_cache(
        token_ids, [(simple, SimpleKVCache()), CacheList(pooling)]
    )
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="generic-checkpoint")
    manager = APCManager(num_blocks=4, block_size=4, disk=disk)
    restored, prefix_len = manager.lookup_exact_cache(token_ids + [99])
    assert prefix_len == len(token_ids)
    assert isinstance(restored[0], tuple)
    assert isinstance(restored[0][0], SimpleKVCache)
    assert restored[0][0].cache_length == len(token_ids)
    assert isinstance(restored[1], CacheList)
    restored_pool = restored[1].caches[0]
    assert isinstance(restored_pool, PoolingCache)
    assert restored_pool.ratio == 2 and restored_pool.remainder == 1
    assert bool(mx.array_equal(restored_pool.pooled, pooling.pooled))
    manager.close()


def test_exact_disk_roundtrip_ring_and_indexed_kv_cache(tmp_path, monkeypatch):
    """Subtype metadata survives checkpoint persistence and process restart."""
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    monkeypatch.setenv("APC_EXACT_MIN_TOKENS", "1")
    token_ids = list(range(10))

    ring = RingSlidingKVCache(window_size=4)
    ring.keys = mx.ones((1, 2, 8, 4))
    ring.values = mx.ones((1, 2, 8, 4)) * 2
    ring.prefill_length = 4
    ring.offset = 11
    ring._ring_pos = 3

    indexed = MiniMaxM3KVCache()
    indexed.kv_cache.update_and_fetch(
        mx.ones((1, 2, 10, 4)) * 3, mx.ones((1, 2, 10, 4)) * 4
    )
    indexed.update_index_and_fetch(mx.ones((1, 1, 10, 4)) * 5)
    mx.eval(ring.keys, ring.values, indexed.state)

    disk = DiskBlockStore(tmp_path, namespace="special-checkpoint")
    manager = APCManager(num_blocks=4, block_size=4, disk=disk)
    assert manager.store_exact_cache(token_ids, [ring, indexed])
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="special-checkpoint")
    manager = APCManager(num_blocks=4, block_size=4, disk=disk)
    restored, prefix_len = manager.lookup_exact_cache(token_ids + [99])
    assert prefix_len == len(token_ids)
    restored_ring, restored_indexed = restored
    assert isinstance(restored_ring, RingSlidingKVCache)
    assert (
        restored_ring.window_size,
        restored_ring.prefill_length,
        restored_ring.offset,
        restored_ring._ring_pos,
    ) == (4, 4, 11, 3)
    assert isinstance(restored_indexed, MiniMaxM3KVCache)
    assert restored_indexed.offset == 10
    assert restored_indexed.index_offset == 10
    assert bool(
        mx.array_equal(
            restored_indexed.index_keys,
            indexed.index_keys[..., : indexed.index_offset, :],
        )
    )
    manager.close()


def _tiny_exact_cache(tokens):
    from mlx_vlm.models.cache import ArraysCache

    c = ArraysCache(size=1)
    c[0] = mx.zeros((1, 2, max(1, len(tokens)), 32))
    return [c]


def test_a_short_prompt_cannot_poison_later_lookups():
    manager = APCManager(num_blocks=8, block_size=16)
    manager.store_exact_cache([1], _tiny_exact_cache([1]))

    later = list(range(1, 400))
    cache, reused = manager.lookup_exact_cache(later)

    assert cache is None
    assert reused == 0


def test_model_processor_hook_contributes_and_is_defensive():
    base = semantic_extra_hash(image_hash=5)

    contributor = SimpleNamespace(apc_key_dependencies=lambda: ["adapter-x"])
    assert semantic_extra_hash(image_hash=5, model=contributor) != base

    plain = SimpleNamespace(foo=1)
    boom = SimpleNamespace(
        apc_key_dependencies=lambda: (_ for _ in ()).throw(ValueError)
    )
    not_callable = SimpleNamespace(apc_key_dependencies=5)
    assert semantic_extra_hash(image_hash=5, model=plain) == base
    assert semantic_extra_hash(image_hash=5, model=boom) == base
    assert semantic_extra_hash(image_hash=5, model=not_callable) == base
    assert model_key_dependencies(None, None) == ()


def test_hash_payload_none_list_and_ref():
    assert _hash_payload(None) is None
    assert _hash_payload([]) is None
    assert _hash_payload(["a.png", "b.png"]) == _hash_payload(["a.png", "b.png"])
    assert _hash_payload("x") == hash_image_payload(image_ref="x")


BLOCK_SIZE = 16
TRACE_GROUP_SIZE = 64
BITS = 8


@pytest.fixture
def _clear_apc_trace_env(monkeypatch):
    monkeypatch.delenv("APC_TRACE", raising=False)
    yield
    monkeypatch.delenv("APC_TRACE", raising=False)


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_reject_records_emit_trace(monkeypatch, caplog):
    monkeypatch.setenv("APC_TRACE", "1")
    manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
    token_ids = list(range(BLOCK_SIZE))

    class UnclonableCache:
        keys = "not-an-array"
        values = "not-an-array"

    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        assert manager.store_exact_cache(token_ids, [UnclonableCache()]) is False
    assert any("APC_TRACE reject" in r.message for r in caplog.records)
    assert any("unclonable" in r.message for r in caplog.records)


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_quantized_with_dequant_ok():
    # Last dim must be divisible by group_size for mx.quantize.
    c = QuantizedKVCache(group_size=TRACE_GROUP_SIZE, bits=BITS)
    c.update_and_fetch(
        mx.random.normal((1, 2, 8, TRACE_GROUP_SIZE)),
        mx.random.normal((1, 2, 8, TRACE_GROUP_SIZE)),
    )
    result = classify_layer_for_apc(c)
    assert result.status == "ok"


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_unsupported_opaque_type():
    class Bogus:
        pass

    result = classify_layer_for_apc(Bogus())
    assert result.status == "unsupported"
    assert result.reason


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_supported_model_ok(caplog):
    class FakeLang:
        def make_cache(self):
            return [
                BatchRotatingKVCache(32, [0]),
                BatchQuantizedKVCache([0], group_size=TRACE_GROUP_SIZE, bits=BITS),
                BatchKVCache([0]),
            ]

    with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
        result = self_check_model_apc(FakeLang(), kv_bits=8.0)
    assert result.ok is True
    assert result.apc_mode == "exact"
    assert any("APC self-check ok" in r.message for r in caplog.records)


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_no_make_cache_not_ok(caplog):
    class NoCache:
        pass

    result = self_check_model_apc(NoCache())
    assert result.ok is False


@pytest.mark.usefixtures("_clear_apc_trace_env")
def test_does_not_raise_on_failure():
    class FakeLang:
        def make_cache(self):
            raise RuntimeError("boom")

    result = self_check_model_apc(FakeLang())
    assert result.ok is False
    assert result.notes


B, H, D = 1, 2, 32
GROUP_SIZE = 32
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


def _batch_cache(kind, left_padding, **kwargs):
    if kind == "rotating":
        return BatchRotatingKVCache(
            kwargs.pop("max_size", SWA_MAX), list(left_padding), **kwargs
        )
    factory, defaults = {
        "dense": (BatchKVCache, {}),
        "uniform": (BatchQuantizedKVCache, dict(group_size=GROUP_SIZE, bits=BITS)),
        "turbo": (BatchTurboQuantKVCache, dict(bits=4.0)),
    }[kind]
    return factory(list(left_padding), **(defaults | kwargs))


def _fill_batch_cache(kind, left_padding, seq_len, **kwargs):
    cache = _batch_cache(kind, left_padding, **kwargs)
    keys, values = _rand_kv(batch=len(left_padding), seq_len=seq_len)
    cache.update_and_fetch(keys, values)
    mx.eval(cache.state)
    return cache, keys, values


@pytest.mark.parametrize("kind", ["rotating", "turbo"])
def test_batch_cache_introspection(kind):
    empty = _batch_cache(kind, [0, 0])
    assert empty.empty() is True
    assert empty.batch_size == 2
    assert empty.is_single_row() is False
    filled, _, _ = _fill_batch_cache(kind, [0], seq_len=8)
    assert filled.empty() is False
    assert filled.batch_size == 1
    assert filled.is_single_row() is True


@pytest.mark.parametrize(
    "kind, expected", [("uniform", QuantizedKVCache), ("turbo", TurboQuantKVCache)]
)
def test_extract_empty_cache(kind, expected):
    row = _batch_cache(kind, [0, 0]).extract(0)
    assert isinstance(row, expected)
    assert row.keys is None or row.offset == 0


def test_layer_kv_float_helper_handles_quantized_tuple_keys():
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

    bq, _, _ = _fill_batch_cache("uniform", [0, 0], seq_len=12)
    bk, bv = layer_kv_for_apc(bq, batch_idx=1)
    mx.eval(bk, bv)
    assert bk.shape[0] == 1
    assert bk.shape[-2] <= 12


def test_layer_kv_rejects_unknown_without_crashing():
    from mlx_vlm.apc import layer_kv_for_apc

    class Bogus:
        keys = (1, 2, 3)
        values = (4, 5, 6)
        offset = 3

    assert layer_kv_for_apc(Bogus()) == (None, None)


def test_extract_b1_batch_rotating_equals_clone_after_extract():
    cache, _, _ = _fill_batch_cache("rotating", [0], seq_len=16)
    row = extract_prompt_cache_from_batch([cache], 0)
    assert row is not None
    cloned = _clone_prompt_cache_for_apc(row)
    assert cloned is not None


def _filter_reordered_row(cache):
    cache.prepare(right_padding=[2, 0], lengths=[4, 6])
    k, v = _rand_kv(batch=2, seq_len=3)
    cache.update_and_fetch(k, v)

    cache.filter(mx.array([1, 0], dtype=mx.int32))
    cache.filter(mx.array([1], dtype=mx.int32))


@pytest.mark.parametrize("kind", ["dense", "uniform"])
def test_filter_keeps_pending_right_padding_aligned(kind):
    cache = _batch_cache(kind, [0, 0])
    _filter_reordered_row(cache)
    assert cache._right_padding.tolist() == [2]
    cache.finalize()
    assert all(part.shape[0] == 1 for part in _array_leaves((cache.keys, cache.values)))
    assert cache.offset.tolist() == [1]
    assert cache.left_padding.tolist() == [2]


def test_rotating_filter_keeps_pending_lengths_aligned():
    cache = BatchRotatingKVCache(32, [0, 0])

    _filter_reordered_row(cache)

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


def test_extract_returns_turboquant_kv_cache():
    from mlx_vlm.turboquant import TurboQuantKVCache

    cache, k, _ = _fill_batch_cache("turbo", [0, 0], seq_len=24)
    row = cache.extract(1)
    assert isinstance(row, TurboQuantKVCache)
    assert row.offset == 24
    dk, dv = row.dequantize_for_apc()
    mx.eval(dk, dv)
    assert dk.shape == (1, H, 24, D)
    # TurboQuant is lossy; keep a loose bound
    assert _max_abs_error(dk, k[1:2]) < 2.0


def test_snapshot_and_exact_store_multi_row():
    from mlx_vlm.apc import snapshot_prompt_cache_row

    seq_len = 2 * BLOCK_SIZE
    token_ids = list(range(seq_len))
    turbo, _, _ = _fill_batch_cache("turbo", [0, 0], seq_len=seq_len)
    batch_kv, _, _ = _fill_batch_cache("dense", [0, 0], seq_len=seq_len)
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


def test_layer_kv_for_apc_batch_turbo():
    from mlx_vlm.apc import layer_kv_for_apc

    cache, _, _ = _fill_batch_cache("turbo", [0, 0], seq_len=12)
    k, v = layer_kv_for_apc(cache, batch_idx=1)
    mx.eval(k, v)
    assert k is not None and v is not None
    assert k.shape[0] == 1
    assert k.shape[-2] <= 12
    assert not isinstance(k, tuple)


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
    _, config = _apc_config("qwen4_exp", text_only=True)
    indexer = Qwen4ExpAttention(config).indexer
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


def _allocated_kv(length, value=1):
    cache = KVCache()
    cache.step = 1  # Allocate exactly the requested length.
    cache.keys = mx.full((1, 1, length, 4), value, dtype=mx.float32)
    cache.values = mx.full((1, 1, length, 4), value + 1, dtype=mx.float32)
    cache.offset = length
    mx.eval(cache.state)
    return cache


def _coordinator(manager, caches):
    return manager.coordinator(SimpleNamespace(make_cache=lambda: caches))


@pytest.fixture
def memory_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "16")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    managers = []

    def make(*, budget=4096, disk=False):
        store = DiskBlockStore(tmp_path, namespace="memory") if disk else None
        manager = APCManager(num_blocks=16, block_size=16, disk=store)
        manager.memory_max_bytes = budget
        manager.memory_reserve_bytes = 0
        monkeypatch.setattr(manager, "_memory_headroom", lambda: 1 << 40)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close()


@pytest.fixture
def disk_reader(memory_manager):
    """Seed a disk checkpoint and return a manager with no resident entries."""

    def make(tokens, caches, *, budget=1 << 20):
        writer = memory_manager(budget=budget, disk=True)
        assert writer.store_exact_cache(tokens, caches)
        writer.disk.flush()
        return memory_manager(budget=budget, disk=True)

    return make


def test_custom_state_accounting_without_snapshot_or_evaluation():
    class CustomCache:
        def __init__(self):
            self.state = {"kv": mx.ones((2, 3)), "nested": [mx.zeros((4,))]}
            self.meta_state = {"offsets": mx.zeros((2,), dtype=mx.int32)}

        def prefix_cache_snapshot(self):
            raise AssertionError("Memory accounting must not clone the state")

    cache = CustomCache()
    assert _cache_nbytes([cache]) == 48
    assert _cache_nbytes([cache, cache]) == 48


@pytest.mark.parametrize(
    "make_cache",
    [KVCache, QuantizedKVCache, lambda: RotatingKVCache(max_size=512)],
    ids=["dense", "quantized", "windowed"],
)
def test_kv_growth_ignores_unused_capacity(memory_manager, make_cache):
    capacity = 256
    cache = make_cache()
    cache.step = 1
    tensor = mx.ones((1, 1, capacity, 64))
    cache.update_and_fetch(tensor, tensor + 1)
    cache.trim(capacity - 16)
    cache.step = 256
    allocated_bytes = _cache_nbytes(cache)
    per_token = allocated_bytes // capacity
    manager = memory_manager(budget=1 << 20)
    coordinator = _coordinator(manager, [cache])
    assert manager.store_exact_cache(list(range(16)), [cache])

    coordinator.prepare_prefill([2000, 2000, 2000])
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == (
        2 * 3 * 2048 * per_token
    )
    assert _cache_nbytes(cache) == allocated_bytes
    assert allocated_bytes == capacity * _cache_nbytes(cache.state) // 16


def test_disk_expansion_is_admitted_before_reserving_capacity(disk_reader, monkeypatch):
    tokens = list(range(16))
    reader = disk_reader(tokens, [_allocated_kv(16)], budget=4096)
    monkeypatch.setattr(reader, "_memory_headroom", lambda: 4096)
    monkeypatch.setattr(
        KVCache, "prefix_cache_reserve", lambda *a: pytest.fail("expanded")
    )
    assert reader.lookup_exact_cache(tokens + [99] * 6000) == (None, 0)
    assert reader.stats.memory_skips == 1


def test_padded_kv_admission_counts_allocated_buffers(memory_manager, monkeypatch):
    cache = _allocated_kv(256)
    cache.offset = 16
    manager = memory_manager(budget=1024)
    monkeypatch.setattr(
        apc, "_clone_prompt_cache_for_apc", lambda *a, **kw: pytest.fail("cloned")
    )
    # Admission includes the unused portion of the 8 KiB buffer.
    assert not manager.store_exact_cache(list(range(16)), [cache])
    assert manager.resident_bytes() == 0
    assert manager.stats.memory_skips == 1


def test_batch_checkpoint_skips_extraction_without_headroom(
    memory_manager, monkeypatch
):
    manager = memory_manager(disk=True)
    coordinator = manager.coordinator(
        SimpleNamespace(make_cache=lambda: [ArraysCache(1)])
    )
    cache = ArraysCache(1)
    cache[0] = mx.ones((2, 4))
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
    monkeypatch.setattr(
        apc, "snapshot_prompt_cache_row", lambda *a, **kw: pytest.fail("extracted")
    )
    assert not coordinator.store_checkpoint(list(range(32)), [cache], batch_idx=0)
    assert manager.stats_snapshot()["memory_skips"] == 1


@pytest.mark.parametrize("exact", [True, False])
def test_disk_restore_checks_headroom_before_loading(
    memory_manager, monkeypatch, exact
):
    manager = memory_manager(budget=0, disk=True)
    source = _allocated_kv(32)
    tokens = list(range(32))
    if exact:
        manager.store_exact_cache(tokens, [source])
        load_method, lookup_method = "load_exact_cache", "lookup_exact_cache"
    else:
        manager.store_kv_blocks(tokens, [source.keys], [source.values])
        load_method, lookup_method = (
            "load_layer_major_prefix",
            "lookup_prefix_disk_cache",
        )
    manager.disk.flush()
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 0)
    monkeypatch.setattr(
        manager.disk, load_method, lambda *a, **kw: pytest.fail("loaded")
    )
    assert getattr(manager, lookup_method)(tokens + [99]) == (None, 0)
    assert manager.stats_snapshot()["memory_skips"] >= 1


def test_memory_restore_accounts_for_extended_prompt_capacity(
    memory_manager, monkeypatch
):
    manager = memory_manager()
    tokens = list(range(16))
    assert manager.store_exact_cache(tokens, [_allocated_kv(16)])
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 4096)
    monkeypatch.setattr(
        apc, "_clone_prompt_cache_for_apc", lambda *a, **kw: pytest.fail("cloned")
    )
    # The stored 512 bytes fit, but allocating capacity for 1,000 tokens does not.
    assert manager.lookup_exact_cache(tokens + [99] * 984) == (None, 0)
    assert manager.stats_snapshot()["memory_skips"] == 1


def test_byte_eviction_preserves_leased_blocks(memory_manager):
    manager = memory_manager(budget=1024)
    source = _allocated_kv(32)
    coordinator = _coordinator(manager, [source])
    leased = manager.store_kv_blocks(list(range(32)), [source.keys], [source.values])
    assert len(leased) == 2
    manager.release(leased[:1])
    manager.memory_max_bytes = 0
    coordinator.prepare_prefill(100_000)
    assert manager.resident_bytes() == 512
    assert leased[1].ref_cnt == 1 and leased[1].keys is not None
    manager.release(leased[1:])
    coordinator.prepare_prefill(100_000)
    assert manager.resident_bytes() == 0


def test_sequential_long_prefixes_remain_bounded_and_replay_from_disk(
    memory_manager, monkeypatch
):
    manager = memory_manager(budget=2 << 20, disk=True)
    coordinator = _coordinator(manager, [KVCache()])
    manager.disk.queue_max_bytes = 1 << 20
    live_bytes = [0]
    # Model weights and other allocations leave 7 MiB. These synthetic caches
    # use actual Metal tensors, but exercise pressure at a small, safe scale.
    monkeypatch.setattr(
        manager,
        "_memory_headroom",
        lambda: (7 << 20) - manager.resident_bytes() - live_bytes[0],
    )
    for i, length in enumerate([30_000, 30_000, 50_000, 50_000, 100_000]):
        coordinator.prepare_prefill(length)
        cache = _allocated_kv(length, i)
        live_bytes[0] = cache.nbytes
        assert manager.store_exact_cache([i] * length, [cache])
        assert manager.resident_bytes() <= manager.memory_max_bytes
        assert manager.disk.pending_bytes <= manager.disk.queue_max_bytes
        del cache
        live_bytes[0] = 0
        mx.clear_cache()

    coordinator.prepare_prefill(100_001)
    # The oldest cached states are gone before the next 100k allocation.
    assert manager.resident_bytes() == 0
    restored, count = manager.lookup_exact_cache([4] * 100_000 + [99])
    assert count == 100_000
    assert mx.all(restored[0].state[0] == 4).item()
    assert manager.stats_snapshot()["disk_write_failures"] == 0
    assert manager.stats_snapshot()["disk_hits"] == 1


def test_disk_queue_applies_byte_backpressure(tmp_path, monkeypatch):
    disk = DiskBlockStore(tmp_path)
    disk.queue_max_bytes = 512
    started, release, second_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original = disk._write_exact_cache_snapshot

    def slow_write(path, payload):
        if payload.cache_hash == 1:
            started.set()
            assert release.wait(5)
        else:
            second_started.set()
        return original(path, payload)

    monkeypatch.setattr(disk, "_write_exact_cache_snapshot", slow_write)
    caches = [_allocated_kv(16), _allocated_kv(16, 2)]
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


def test_failed_direct_spill_does_not_report_a_store(memory_manager, monkeypatch):
    manager = memory_manager(budget=0, disk=True)

    def fail_write(*args):
        raise OSError("full")

    monkeypatch.setattr(manager.disk, "_write_exact_cache_snapshot", fail_write)
    assert not manager.store_exact_cache(list(range(32)), [_allocated_kv(32)])
    stats = manager.stats_snapshot()
    assert stats["exact_stores"] == stats["resident_bytes"] == 0
    assert stats["disk_write_failures"] == 1
    assert not manager.disk._in_flight


@pytest.mark.parametrize("synchronous", [True, False])
def test_completed_oversized_disk_write_obeys_cap(tmp_path, synchronous):
    disk = DiskBlockStore(tmp_path, max_bytes=512)
    try:
        disk.save_exact_cache(
            1, [1] * 32, 0, [_allocated_kv(32)], synchronous=synchronous
        )
        disk.flush()
        assert disk.disk_bytes <= disk.max_bytes
        assert disk.num_exact_indexed == 0
        assert disk.evictions == 1
    finally:
        disk.close()


@pytest.mark.parametrize("opt_out", ["environment", "empty_path"])
def test_default_disk_opt_out(tmp_path, monkeypatch, opt_out):
    monkeypatch.setenv("MLX_VLM_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("APC_ENABLED", "1")
    if opt_out == "environment":
        monkeypatch.setenv("APC_DISK_ENABLED", "0")
        overrides = None
    else:
        overrides = {"disk_path": ""}
    manager = from_env(overrides=overrides)
    assert manager.disk is None
    assert not (tmp_path / "apc").exists()


@pytest.fixture
def _seed_block_storage():
    mx.random.seed(0)


@pytest.mark.usefixtures("_seed_block_storage")
def test_kv_block_handle_empty():
    handle = KVBlockHandle()
    assert handle.resident_bytes() == 0


@pytest.mark.parametrize("base", [object, _BaseCache])
def test_unknown_checkpoint_state_keeps_conservative_growth_estimate(
    memory_manager, monkeypatch, base
):
    class GrowingCache(base):
        state = mx.ones((1, 256, 1024))
        meta_state = ()

    manager = memory_manager(budget=4 << 20)
    # Opaque checkpoints may grow with tokens.
    monkeypatch.setattr(apc, "_clone_prompt_cache_for_apc", lambda cache: cache)
    cache = GrowingCache()
    coordinator = _coordinator(manager, [cache])
    assert manager.store_exact_cache(list(range(18)), [cache])
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 8 << 20)
    coordinator.prepare_prefill(6001)
    assert manager.resident_bytes() == 0
    assert not manager._make_room()


@pytest.mark.parametrize(
    "make_cache",
    [
        ConcatenateKVCache,
        SimpleKVCache,
        lambda: ChunkedKVCache(65),
        lambda: BufferedRotatingKVCache(65, buffer_size=300),
        lambda: BufferedRotatingKVCache(65, keep=1),
        lambda: StaticPrefixKVCache(513),
    ],
)
@pytest.mark.parametrize("chunk_size", [37, 256])
def test_builtin_kv_profiles_bound_prefill_allocations(make_cache, chunk_size):
    cache = make_cache()
    empty = cache_memory_components([cache], 0)[0]
    assert not empty.fallback
    assert empty.footprint(6000, chunk_size) == 0
    cache.update_and_fetch(mx.ones((1, 1, 1, 4)), mx.ones((1, 1, 1, 8)))
    profile = cache_memory_components([cache], 1)[0]
    assert not profile.fallback
    assert profile.source_bytes == cache.nbytes
    estimate = profile.footprint(1, chunk_size)
    assert cache.nbytes <= estimate < cache.nbytes + 2 * 256 * 48

    peak = cache.nbytes
    for start in range(1, 6000, chunk_size):
        if isinstance(cache, ChunkedKVCache):
            cache.maybe_trim_front()
        size = min(chunk_size, 6000 - start)
        cache.update_and_fetch(mx.ones((1, 1, size, 4)), mx.ones((1, 1, size, 8)))
        peak = max(peak, cache.nbytes)
    estimate = profile.footprint(6000, chunk_size)
    assert peak <= estimate < peak + 2 * 256 * 48
    assert profile.footprint(0, chunk_size) == 0


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("ratio", [4, 64])
@pytest.mark.parametrize("seed_length", [1, 65])
def test_pooling_profiles_separate_buffers_from_compressed_growth(
    batch_size, ratio, seed_length
):
    cache = (
        PoolingCache(ratio)
        if batch_size == 1
        else BatchPoolingCache(ratio, left_padding=[0, 3, 7])
    )

    def advance(length):
        kv = mx.ones((batch_size, length, 8), dtype=mx.float16)
        gate = mx.ones((batch_size, length, 4), dtype=mx.float32)
        ready, _, _ = cache.accumulate_windows(kv, gate, 0)
        cache.update_and_fetch(
            mx.ones((batch_size, ready.shape[1] // ratio, 4), dtype=mx.float16)
        )

    empty = cache_memory_components([cache], 0)[0]
    assert not empty.fallback and empty.footprint(6000) == 0
    advance(seed_length)
    profile = cache_memory_components([cache], seed_length, batch_size=batch_size)[0]
    assert not profile.fallback
    assert profile.fixed_bytes == ratio * 32

    for start in range(seed_length, 6000, 37):
        advance(min(37, 6000 - start))
    estimate = batch_size * profile.footprint(6000)
    assert cache.nbytes <= estimate <= 2 * cache.nbytes


@pytest.mark.parametrize("read_only", [False, True])
def test_static_prefix_profile_survives_restore(read_only):
    prefix = StaticPrefixKVCache(513)
    prefix.update_and_fetch(mx.ones((1, 1, 16, 4)), mx.ones((1, 1, 16, 8)))
    source = StaticPrefixKVCache.from_prefix(prefix) if read_only else prefix
    cache = StaticPrefixKVCache.from_state(source.state, source.meta_state)
    assert cache.read_only == read_only
    profile = cache_memory_components([cache], 16)[0]
    assert not profile.fallback
    if read_only:
        assert profile.footprint(1) == profile.footprint(6000) == cache.nbytes
    cache.update_and_fetch(mx.ones((1, 1, 1, 4)), mx.ones((1, 1, 1, 8)))
    assert cache.offset == (16 if read_only else 17)
    legacy = StaticPrefixKVCache.from_state(source.state, source.meta_state[:3])
    assert not legacy.read_only


@pytest.mark.parametrize("make_cache", [HyV4KVCache, lambda: RingSlidingKVCache(16)])
@pytest.mark.parametrize("chunk_size", [37, 256, 1024])
def test_model_kv_profiles_bound_restored_prefill(make_cache, chunk_size):
    cache = make_cache()
    cache.update_and_fetch(mx.ones((1, 1, 16, 4)), mx.ones((1, 1, 16, 8)))
    cache = clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=[])
    profile = cache_memory_components([cache], 16)[0]
    assert not profile.fallback
    assert profile.source_bytes == cache.nbytes
    for start in range(16, 6000, chunk_size):
        size = min(chunk_size, 6000 - start)
        cache.update_and_fetch(mx.ones((1, 1, size, 4)), mx.ones((1, 1, size, 8)))
    assert cache.nbytes <= profile.footprint(6000, chunk_size) <= 2 * cache.nbytes


def test_z1t_prefill_memory_is_fixed():
    _, config = _apc_config("z1t")
    layer = AFTConv(config)
    cache = Z1TCache()
    layer(mx.ones((1, 1, 8)), cache)
    cache = clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=[])
    profile = cache_memory_components([cache], 1)[0]
    assert not profile.fallback
    layer(mx.ones((1, 5999, 8)), cache)
    assert profile.footprint(1) == profile.footprint(6000) == _cache_nbytes(cache)


@pytest.mark.parametrize(
    "make_cache,batch_size",
    [(MiniMaxM3KVCache, 1), (lambda: MiniMaxM3BatchKVCache([0, 3]), 2)],
)
def test_minimax_profiles_include_indexer_allocations(make_cache, batch_size):
    cache = make_cache()

    def advance(length):
        cache.update_and_fetch(
            mx.ones((batch_size, 1, length, 4), dtype=mx.float16),
            mx.ones((batch_size, 1, length, 8), dtype=mx.float16),
        )
        cache.update_index_and_fetch(mx.ones((batch_size, 1, length, 8)))

    empty = cache_memory_components([cache], 0)[0]
    assert not empty.fallback and empty.footprint(6000) == 0
    advance(16)
    if batch_size == 1:
        cache = clone_cache_entry(cache, min_capacity_tokens=None, eval_targets=[])
    profile = cache_memory_components([cache], 16, batch_size=batch_size)[0]
    assert not profile.fallback
    assert profile.source_bytes * batch_size == cache.nbytes
    for start in range(16, 6000, 257):
        advance(min(257, 6000 - start))
    estimate = batch_size * profile.footprint(6000, 257)
    assert cache.nbytes <= estimate < 1.1 * cache.nbytes


def _apc_config(name, *, text_only=False):
    """Apply plain APC settings before config constructors derive layer layouts."""
    profile = MODEL_CASES["apc"][name]
    case = (
        next(case for case in MODEL_CASES["cases"] if case["id"] == profile["case"])
        if "case" in profile
        else {"module": profile["module"], "config": {}}
    )
    fields = copy.deepcopy(case["config"])

    def merge(target, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(fields, profile["config"])
    module = importlib.import_module("mlx_vlm.models." + case["module"])
    if text_only:
        return module, build_config(module, fields["text_config"], "TextConfig")
    return module, build_config(module, fields)


def _apc_language_model(name):
    module, config = _apc_config(name)
    if name == "qwen3_5":
        return module.LanguageModel(config.text_config, config)
    return module.Model(config).language_model


@pytest.mark.parametrize("model_name", ["gemma4", "qwen3_5"])
def test_apc_exact_mode_detected_for_hybrid_models(model_name):
    """Hybrid models must route to exact mode, not block mode."""
    lm = _apc_language_model(model_name)
    assert model_apc_mode(lm) == "exact"


def _token_kv(tokens):
    cache = KVCache()
    values = mx.array(tokens, dtype=mx.float32).reshape(1, 1, -1, 1)
    cache.update_and_fetch(values, values + 1)
    return cache


@pytest.fixture
def prefix_manager(monkeypatch, tmp_path):
    monkeypatch.setenv("APC_CHECKPOINT_ENTRIES", "2")
    monkeypatch.setenv("APC_DISK_MIN_FREE_RAM_GB", "0")
    managers = []

    def make(tier="memory"):
        disk = (
            None if tier == "memory" else DiskBlockStore(tmp_path, namespace="partial")
        )
        manager = APCManager(num_blocks=8, block_size=16, disk=disk)
        if tier == "disk-only":
            manager._exact_cache_max = 0
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        if manager.disk is not None:
            manager.close()


@pytest.mark.parametrize("tier", ["memory", "disk", "disk-only"])
def test_dense_checkpoint_reuses_only_common_blocks(prefix_manager, tier):
    stored = list(range(80))
    divergent = stored[:37] + [999, 998, 997]
    manager = prefix_manager(tier)
    assert manager.store_exact_cache(stored, [_token_kv(stored)], extra_hash=7)
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = prefix_manager(tier)

    assert manager.lookup_exact_cache(divergent, extra_hash=8) == (None, 0)
    restored, count = manager.lookup_exact_cache(divergent, extra_hash=7)
    assert count == 32
    assert restored[0].offset == 32
    assert restored[0].state[0].flatten().tolist() == stored[:32]
    assert manager.stats_snapshot()["matched_tokens"] == 32
    if manager.disk:
        assert manager.stats_snapshot()["disk_hits"] == 1

    # Mutating the returned row must not alter either stored checkpoint.
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


def test_checkpoint_schedule_respects_media_budget_and_opt_out(prefix_manager):
    manager = prefix_manager()
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(
        SimpleNamespace(make_cache=lambda: [ArraysCache(1)])
    )
    tokens = list(range(75))
    assert coordinator.checkpoint_lengths(tokens, set()) == [64, 74]
    manager._exact_cache_max = 4
    assert coordinator.checkpoint_lengths(tokens, set()) == [32, 48, 64, 74]
    # A media span crossing a nominal boundary must be completely prefetched.
    tokens[45:67] = [999] * 22
    assert coordinator.checkpoint_lengths(tokens, {999}) == [67, 74]
    manager.checkpoint_interval_tokens = 0
    assert coordinator.checkpoint_lengths(tokens, set()) == [74]


def _embeddings(lm, tokens):
    return lm.model.embed_tokens(tokens) * getattr(lm.model, "embed_scale", 1)


@pytest.mark.parametrize("prefill_step_size", [None, 4, 16, 32])
def test_lfm_mixed_prefill_keeps_logits_before_right_padding(
    prefix_manager, prefill_step_size
):
    from mlx_vlm.apc import make_warm_batch_exact_cache_multi
    from mlx_vlm.generate.ar import PromptProcessingBatch

    mx.random.seed(19)
    lm = _apc_language_model("lfm2")
    manager = prefix_manager()
    manager._exact_cache_max = 8
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    warm_tokens = [i % 50 + 1 for i in range(71)]
    cold_tokens = [i % 30 + 51 for i in range(25)]
    seed_cache = lm.make_cache()
    lm(mx.array([warm_tokens[:64]]), cache=seed_cache)
    assert manager.store_exact_cache(warm_tokens[:64], seed_cache)
    restored, count = manager.lookup_exact_cache(warm_tokens)
    assert count == 64
    caches, _ = make_warm_batch_exact_cache_multi([restored, lm.make_cache()], [64, 0])
    suffixes = [warm_tokens[64:], cold_tokens]
    padded = mx.array([suffixes[0] + [0] * 18, suffixes[1]])
    batch = PromptProcessingBatch(
        model=lm,
        uids=[0, 1],
        input_ids=suffixes,
        max_tokens=[1, 1],
        inputs_embeds=_embeddings(lm, padded),
        prompt_kwargs={},
        warm_cache=caches,
        prefill_step_size=prefill_step_size,
        right_pad_per_row=[18, 0],
        suffix_lens=[7, 25],
        apc_manager=manager,
        apc_coordinator=coordinator,
        apc_meta=[
            {
                "full_input_ids": tokens,
                "prefix_len": prefix,
                "checkpoint_lengths": coordinator.checkpoint_lengths(tokens, set()),
            }
            for tokens, prefix in [(warm_tokens, 64), (cold_tokens, 0)]
        ],
    )
    while batch.needs_processing():
        assert batch.prompt_step() > 0
    sampled = []

    def sample(logprobs):
        sampled.append(logprobs)
        return mx.argmax(logprobs, axis=-1)

    batch.generate(sample, [lambda _: False, lambda _: False])
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
        # Right-padding must preserve LFM's final real convolution state too.
        assert mx.allclose(
            caches[0][0][row : row + 1], reference[0][0], atol=1e-4, rtol=1e-4
        ).item()


@pytest.mark.parametrize("tier", ["memory", "disk"])
def test_diffusion_prefills_only_divergent_suffix(prefix_manager, tier):
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
    manager = prefix_manager(tier)
    manager.block_size = 2
    manager.checkpoint_interval_tokens = 4
    manager.exact_cache_min_tokens = 1
    tokens = list(range(2, 13))
    kwargs = dict(max_tokens=2, max_denoising_steps=1, _apc_semantic_hash=11)
    list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    # An identical replay has no new checkpoint to capture. It must not store
    # a third full-prompt snapshot that evicts the divergent-prefix checkpoint.
    repeated = list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    assert repeated[-1].cached_tokens == 10
    if manager.disk:
        manager.close()
        manager.disk = None
        manager = prefix_manager(tier)
        manager.block_size = 2
        manager.checkpoint_interval_tokens = 4
        manager.exact_cache_min_tokens = 1
    recorder.input_lengths.clear()
    warm = list(
        stream_generate(
            model,
            FakeProcessor(),
            "",
            input_ids=mx.array([tokens[:9] + [13, 14]]),
            _apc_manager=manager,
            **kwargs,
        )
    )
    assert warm[-1].cached_tokens == 8
    assert recorder.input_lengths == [2, 1]


@pytest.mark.parametrize("model_name", ["gemma4", "qwen3_5"])
@pytest.mark.parametrize("tier", ["memory", "disk"])
@pytest.mark.parametrize("path", ["stream", "batch"])
def test_hybrid_generation_restores_before_divergence(
    prefix_manager, model_name, tier, path
):
    from mlx_vlm.generate.ar import PromptProcessingBatch, generate_step
    from mlx_vlm.models.base import InputEmbeddingsFeatures

    mx.random.seed(13)
    lm = _apc_language_model(model_name)
    manager = prefix_manager(tier)
    manager.checkpoint_interval_tokens = 16
    coordinator = manager.coordinator(lm)
    tokens = [i % 50 + 1 for i in range(70)] + [51, 52, 53, 54, 55]
    boundaries = coordinator.checkpoint_lengths(tokens, set())
    assert boundaries == [64, 74]
    ids = mx.array([tokens])

    if path == "stream":
        wrapper = SimpleNamespace(
            language_model=lm,
            get_input_embeddings=lambda ids, *a, **kw: InputEmbeddingsFeatures(
                inputs_embeds=_embeddings(lm, ids)
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
        batch = PromptProcessingBatch(
            model=lm,
            uids=[0],
            input_ids=[tokens],
            max_tokens=[1],
            inputs_embeds=_embeddings(lm, ids),
            prompt_kwargs={},
            warm_cache=lm.make_cache(),
            prefill_step_size=16,
            apc_manager=manager,
            apc_coordinator=coordinator,
            apc_meta=[
                {
                    "full_input_ids": tokens,
                    "prefix_len": 0,
                    "checkpoint_lengths": boundaries,
                }
            ],
        )
        while batch.needs_processing():
            assert batch.prompt_step() > 0
        batch.generate(lambda lp: mx.argmax(lp, axis=-1), [lambda _: False])
        # Final prompt harvest must retain the intermediate state in the LRU.
        assert (
            sorted(len(entry.token_ids) for entry in manager._exact_cache.values())
            == boundaries
        )

    if manager.disk:
        manager.close()
        manager.disk = None
        manager = prefix_manager(tier)
    divergent = tokens[:70] + [60, 61, 62, 63]
    restored, count = manager.lookup_exact_cache(divergent)
    assert count == 64
    # Reference computes the shared document in the same chunks, but never
    # processes A's instructions. This detects stale recurrent/window state.
    cold_cache = lm.make_cache()
    for start in range(0, count, 16):
        lm(mx.array([divergent[start : start + 16]]), cache=cold_cache)
    suffix = mx.array([divergent[count:]])
    cold = lm(suffix, cache=cold_cache).logits
    warm = lm(suffix, cache=restored).logits
    mx.eval(cold, warm)
    assert mx.allclose(cold, warm, atol=1e-5, rtol=1e-5).item()
    assert mx.array_equal(mx.argmax(cold, axis=-1), mx.argmax(warm, axis=-1)).item()


def _array_leaves(value):
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _array_leaves(item)]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _array_leaves(item)]
    return []


def _assert_packed_state_equal(lhs, rhs):
    lhs_leaves = _array_leaves(lhs)
    rhs_leaves = _array_leaves(rhs)
    assert len(lhs_leaves) == len(rhs_leaves)
    for left, right in zip(lhs_leaves, rhs_leaves):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert bool(mx.array_equal(left, right).item())


def _disk_roundtrip(tmp_path, namespace, token_ids, snapshot):
    disk = DiskBlockStore(tmp_path, namespace=namespace)
    manager = APCManager(num_blocks=1, block_size=BLOCK_SIZE, disk=disk)
    assert manager.store_exact_cache(token_ids, snapshot)
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace=namespace)
    manager = APCManager(num_blocks=1, block_size=BLOCK_SIZE, disk=disk)
    restored = manager.lookup_exact_cache(token_ids + [999])
    manager.close()
    return restored


def test_with_left_padding():
    """Left-padding is correctly handled when harvesting from quantized cache."""
    manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
    left_pad = 3
    content_len = 2 * BLOCK_SIZE  # enough for 2 blocks after removing padding

    cache = BatchQuantizedKVCache([left_pad], group_size=GROUP_SIZE, bits=BITS)
    k, v = _rand_kv(batch=1, seq_len=content_len + left_pad)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys)

    batch_caches = [cache, cache]  # 2 layers, same data for simplicity
    token_ids = list(range(content_len))
    blocks = harvest_blocks_from_batch_cache(
        manager, batch_caches, batch_idx=0, full_token_ids=token_ids
    )

    # 2 * BLOCK_SIZE content tokens = 2 full blocks
    assert len(blocks) == 2
    for block in blocks:
        for k_layer in block.keys:
            assert k_layer.shape[2] == BLOCK_SIZE
    manager.release(blocks)


def test_apc_not_disabled_when_kv_bits_set():
    """The ar.py guard that kills APC when kv_bits is set must be removed.

    Checks the source of BatchGenerator.__init__ to ensure the old pattern
    'if apc_manager is not None and kv_bits is not None: apc_manager = None'
    is no longer present.
    """
    import inspect

    from mlx_vlm.generate.ar import BatchGenerator

    source = inspect.getsource(BatchGenerator.__init__)
    # The old guard unconditionally disabled APC when kv_bits was set
    assert not (
        "kv_bits is not None" in source and "apc_manager = None" in source
    ), "Guard still disables APC when kv_bits is set"


def test_hybrid_batch_kv_and_quantized_exact_store():
    """Exact store works for the --kv-bits hybrid layout (pinglin / #1534).

    With kv-bits, single-row continuous-batching uses batch cache classes
    even for B=1: ArraysCache (SSM) + BatchKVCache (unquantized last
    attention layer) + BatchQuantizedKVCache. store_exact_cache must
    clone this mix rather than silently returning False.
    """
    seq_len = 2 * BLOCK_SIZE
    token_ids = list(range(seq_len))

    arrays = ArraysCache(2)
    arrays.cache = [mx.zeros((1, seq_len, D)), mx.zeros((1, seq_len, D))]
    arrays.left_padding = mx.array([0])

    batch_kv = BatchKVCache([0])
    k, v = _rand_kv(batch=1, seq_len=seq_len)
    batch_kv.update_and_fetch(k, v)

    batch_q = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
    kq, vq = _rand_kv(batch=1, seq_len=seq_len)
    batch_q.update_and_fetch(kq, vq)
    mx.eval(batch_kv.keys, batch_q.keys)

    prompt_cache = [arrays, batch_kv, batch_q]
    assert all(apc_exact_eligible(c) for c in prompt_cache)

    # Batch KV collapses to KVCache; quantized state stays packed.
    eval_targets: list = []
    cloned_bk = _clone_cache_entry_for_apc(
        batch_kv, min_capacity_tokens=None, eval_targets=eval_targets
    )
    assert isinstance(cloned_bk, KVCache)
    assert cloned_bk.offset == seq_len

    cloned = _clone_prompt_cache_for_apc(prompt_cache)
    assert cloned is not None
    assert len(cloned) == 3
    assert isinstance(cloned[0], ArraysCache)
    assert isinstance(cloned[1], KVCache)
    assert isinstance(cloned[2], QuantizedKVCache)

    manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
    stored = manager.store_exact_cache(token_ids, prompt_cache, extra_hash=0)
    assert stored is True
    assert manager.stats.exact_stores == 1

    warm, matched_tokens = manager.lookup_exact_cache(token_ids + [999], extra_hash=0)
    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert len(warm) == 3


def test_uniform_roundtrip_and_merge_stay_packed(tmp_path, monkeypatch):
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    seq_len = 32
    token_ids = list(range(seq_len))
    source = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
    keys, values = _rand_kv(seq_len=seq_len)
    source.update_and_fetch(keys, values)
    mx.eval(source.state)
    expected = source.extract(0)

    def fail(*args, **kwargs):
        raise AssertionError("native exact APC must not quantize or dequantize")

    monkeypatch.setattr(QuantizedKVCache, "dequantize_for_apc", fail)
    monkeypatch.setattr(mx, "quantize", fail)

    snapshot = snapshot_prompt_cache_row([source], batch_idx=0)
    assert snapshot is not None
    assert isinstance(snapshot[0], QuantizedKVCache)
    _assert_packed_state_equal(snapshot[0].state, expected.state)

    warm, matched = make_warm_batch_exact_cache_multi(
        [snapshot, [KVCache()]],
        [seq_len, 0],
        kv_quant_config={
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "scheme": "uniform",
        },
    )
    assert matched == seq_len
    assert warm is not None
    assert isinstance(warm[0], BatchQuantizedKVCache)
    _assert_packed_state_equal(warm[0].extract(0).state, expected.state)

    restored, matched = _disk_roundtrip(tmp_path, "native-uniform", token_ids, snapshot)
    assert matched == seq_len
    assert restored is not None
    assert isinstance(restored[0], QuantizedKVCache)
    _assert_packed_state_equal(restored[0].state, snapshot[0].state)


@pytest.mark.parametrize(
    "bits,key_bits,value_bits",
    [
        pytest.param(4.0, None, None, id="integer"),
        pytest.param(3.5, None, None, id="fractional-budget"),
        pytest.param(3.5, 3.5, 3.5, id="fractional-split-codecs"),
    ],
)
def test_turboquant_disk_roundtrip_preserves_packed_state(
    tmp_path, monkeypatch, bits, key_bits, value_bits
):
    from mlx_vlm.turboquant import TurboQuantKVCache, _SplitCodec

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    seq_len = 32
    token_ids = list(range(seq_len))
    source = BatchTurboQuantKVCache(
        [0], bits=bits, key_bits=key_bits, value_bits=value_bits
    )
    keys, values = _rand_kv(seq_len=seq_len)
    source.update_and_fetch(keys, values)
    mx.eval(source.state)
    if key_bits == 3.5:
        assert isinstance(source.key_codec, _SplitCodec)
        assert isinstance(source.value_codec, _SplitCodec)

    def fail(*args, **kwargs):
        raise AssertionError("TurboQuant checkpoint must stay packed")

    monkeypatch.setattr(TurboQuantKVCache, "dequantize_for_apc", fail)
    snapshot = snapshot_prompt_cache_row([source], batch_idx=0)
    assert snapshot is not None
    assert isinstance(snapshot[0], TurboQuantKVCache)

    restored, matched = _disk_roundtrip(
        tmp_path,
        f"native-turbo-{bits}-{key_bits}-{value_bits}",
        token_ids,
        snapshot,
    )
    assert matched == seq_len
    assert restored is not None
    assert isinstance(restored[0], TurboQuantKVCache)
    _assert_packed_state_equal(restored[0].state, snapshot[0].state)

    kv_quant_config = {
        "bits": bits,
        "group_size": GROUP_SIZE,
        "scheme": "turboquant",
    }
    if key_bits is not None:
        kv_quant_config["key_bits"] = key_bits
    if value_bits is not None:
        kv_quant_config["value_bits"] = value_bits
    warm, _ = make_warm_batch_exact_cache_multi(
        [restored, [KVCache()]], [seq_len, 0], kv_quant_config=kv_quant_config
    )
    assert warm is not None
    assert isinstance(warm[0], BatchTurboQuantKVCache)
    _assert_packed_state_equal(warm[0].extract(0).state, snapshot[0].state)

    next_keys, next_values = _rand_kv(batch=2, seq_len=1)
    updated = warm[0].update_and_fetch(next_keys, next_values)
    mx.eval(updated)
    assert warm[0]._idx == seq_len + 1


def test_dequantize_for_apc_returns_none_when_empty():
    """dequantize_for_apc() returns (None, None) on an empty cache."""
    cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
    dk, dv = cache.dequantize_for_apc()
    assert dk is None
    assert dv is None


def test_turboquant_dequantize_for_apc_returns_none_when_empty():
    """TurboQuantKVCache.dequantize_for_apc() returns (None, None) when empty."""
    from mlx_vlm.turboquant import BatchTurboQuantKVCache, TurboQuantKVCache

    cache = TurboQuantKVCache(bits=4)
    dk, dv = cache.dequantize_for_apc()
    assert dk is None
    assert dv is None

    batch_cache = BatchTurboQuantKVCache([0], bits=4)
    dk, dv = batch_cache.dequantize_for_apc()
    assert dk is None
    assert dv is None


def test_harvest_handles_empty_quantized_cache():
    """harvest_blocks_from_batch_cache returns [] for empty quantized caches."""
    manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
    empty_cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
    token_ids = list(range(BLOCK_SIZE))
    blocks = harvest_blocks_from_batch_cache(
        manager, [empty_cache], batch_idx=0, full_token_ids=token_ids
    )
    assert blocks == []


def test_int_coercion_on_float_bits():
    """Float bits value (e.g. 8.0 from JSON) doesn't crash."""
    manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
    seq_len = BLOCK_SIZE

    lk = [_rand_kv(seq_len=seq_len)[0]]
    lv = [_rand_kv(seq_len=seq_len)[1]]
    mx.eval(lk + lv)

    token_ids = list(range(seq_len))
    blocks = manager.store_kv_blocks(token_ids, lk, lv)
    manager.release(blocks)

    matched, _ = manager.lookup_prefix(token_ids)
    # Simulate JSON-parsed config with float values
    quant_config = {"bits": 8.0, "group_size": 32.0}
    warm = make_warm_kv_cache(matched, kv_quant_config=quant_config)

    assert len(warm) == 1
    assert isinstance(warm[0], QuantizedKVCache)
    assert warm[0].bits == 8
    assert warm[0].group_size == 32
    manager.release(matched)


KV_CFG = {"bits": BITS, "group_size": GROUP_SIZE}
TQ_CFG = {"bits": 3.5, "group_size": GROUP_SIZE, "scheme": "turboquant"}


def _store_prefix_blocks(
    manager: APCManager, num_layers: int, seq_len: int, token_ids: List[int]
):
    lk, lv = [], []
    for _ in range(num_layers):
        k, v = _rand_kv(seq_len=seq_len)
        lk.append(k)
        lv.append(v)
    mx.eval(lk + lv)
    blocks = manager.store_kv_blocks(token_ids, lk, lv)
    manager.release(blocks)
    matched, _ = manager.lookup_prefix(token_ids)
    assert matched, "expected APC hit after store"
    return matched


def _layer_type_names(caches) -> List[str]:
    return [type(c).__name__ for c in caches]


def _expected_make_cache_types(num_layers: int) -> List[str]:
    class FakeLayer:
        pass

    class FakeModel:
        layers = [FakeLayer() for _ in range(num_layers)]

    caches = _make_cache(
        FakeModel(),
        [0],
        kv_bits=float(BITS),
        kv_group_size=GROUP_SIZE,
        kv_quant_scheme="uniform",
    )
    return _layer_type_names(caches)


def test_make_warm_batch_kv_cache_multi_matches_make_cache_types():
    num_layers = 4
    seq_len = 2 * BLOCK_SIZE
    manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
    try:
        token_ids = list(range(seq_len))
        matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
        pick = {"matched_blocks": matched, "prefix_len": seq_len}
        warm, max_prefix = make_warm_batch_kv_cache_multi(
            [pick, None], num_layers=num_layers, kv_quant_config=KV_CFG
        )
        assert max_prefix == seq_len
        assert _layer_type_names(warm) == _expected_make_cache_types(num_layers)
        assert isinstance(warm[-1], BatchKVCache)
        assert warm[-1].left_padding.tolist() == [0, seq_len]
    finally:
        manager.close()


def _hybrid_row_caches(seq_len: int, *, n_full_attn: int = 3):
    """Synthetic hybrid: ArraysCache + n_full_attn KVCache layers."""
    arrays = ArraysCache(2)
    arrays.cache = [mx.zeros((1, seq_len, D)), mx.zeros((1, seq_len, D))]
    rows = [arrays]
    for _ in range(n_full_attn):
        c = KVCache()
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        c.keys = k
        c.values = v
        c.offset = seq_len
        rows.append(c)
    return rows


def _live_hybrid_batch(seq_len: int, n_full_attn: int = 3):
    """Live continuous-batching row: ArraysCache + quant full-attn + last float."""
    arrays = ArraysCache(2)
    arrays.cache = [mx.zeros((1, seq_len, D)), mx.zeros((1, seq_len, D))]
    arrays.left_padding = mx.array([0])
    caches = [arrays]
    n = 1 + n_full_attn
    for i in range(n_full_attn):
        layer_idx = 1 + i
        quantize = should_quantize_kv_layer(layer_idx, n)
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        if quantize:
            c = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        else:
            c = BatchKVCache([0])
        c.update_and_fetch(k, v)
        caches.append(c)
    return caches


def test_exact_warm_with_kv_config_extends_live_quant():
    seq_len = 16
    live = _live_hybrid_batch(seq_len)
    row = _hybrid_row_caches(seq_len)
    warm, _ = make_warm_batch_exact_cache_multi(
        [row], [seq_len], kv_quant_config=KV_CFG
    )
    assert _layer_type_names(live) == _layer_type_names(warm)
    extended = _extend_cache(live, warm)
    assert int(extended[1].offset.shape[0]) == 2
    assert isinstance(extended[1], BatchQuantizedKVCache)
    assert isinstance(extended[-1], BatchKVCache)


def _live_tq_batch(seq_len: int, num_layers: int = 4, head_dim: int = D):
    class FakeLayer:
        pass

    class FakeModel:
        layers = [FakeLayer() for _ in range(num_layers)]

    caches = _make_cache(
        FakeModel(),
        [0],
        kv_bits=3.5,
        kv_group_size=GROUP_SIZE,
        kv_quant_scheme="turboquant",
    )
    for c in caches:
        k = mx.random.normal((1, H, seq_len, head_dim))
        v = mx.random.normal((1, H, seq_len, head_dim))
        mx.eval(k, v)
        c.update_and_fetch(k, v)
    return caches


def test_block_warm_multi_turboquant_matches_make_cache_types():
    num_layers = 4
    seq_len = 16
    manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
    try:
        # Store float blocks (APC always float); restore with TQ config.
        token_ids = list(range(seq_len))
        matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
        warm = make_warm_batch_kv_cache(matched, kv_quant_config=TQ_CFG)
        live = _live_tq_batch(seq_len, num_layers=num_layers)
        assert _layer_type_names(warm) == _layer_type_names(live)
        assert isinstance(warm[0], BatchTurboQuantKVCache)
        assert isinstance(warm[-1], BatchKVCache)
    finally:
        manager.close()


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_APC_KV_JOIN", "0") != "1",
    reason="Set RUN_LIVE_APC_KV_JOIN=1 to run live model staggered join smoke",
)
def test_live_batch_generator_staggered_apc_kv_join():
    """Live repro of server concurrent APC+kv join (optional smoke)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from mlx_vlm import load
    from mlx_vlm.generate import BatchGenerator

    model_id = os.environ.get("REPRO_MODEL", "mlx-community/Qwen3-0.6B-4bit")
    model, processor = load(model_id)
    lm = model.language_model if hasattr(model, "language_model") else model
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    def ids(text: str):
        x = tok.encode(text)
        return list(x.ids if hasattr(x, "ids") else x)

    def embeds(id_list):
        e = model.get_input_embeddings(mx.array([id_list]))
        return {k: v for k, v in e.to_dict().items() if v is not None}

    def close(gen):
        if hasattr(gen, "close") and callable(gen.close):
            gen.close()
        elif hasattr(gen, "_wire_stack"):
            gen._wire_stack.close()

    prefix = "Shared prefix for agent tools: " + ("schema " * 50)
    a = ids(prefix + " task A")
    b = ids(prefix + " task B")
    apc = APCManager(num_blocks=4096, block_size=16)
    gen = BatchGenerator(
        lm,
        processor,
        max_tokens=24,
        kv_bits=8.0,
        kv_quant_scheme="uniform",
        apc_manager=apc,
        prefill_step_size=64,
        compute_logprobs=False,
    )
    try:
        gen.insert([a], max_tokens=24, prompt_kwargs=[embeds(a)])
        steps = 0
        while gen.has_work and steps < 200:
            _pr, resp = gen.next()
            steps += 1
            if resp:
                gen.insert([b], max_tokens=24, prompt_kwargs=[embeds(b)])
                break
        while gen.has_work:
            gen.next()
            steps += 1
            if steps > 800:
                raise TimeoutError("drain too long")
    finally:
        close(gen)
        apc.close()
