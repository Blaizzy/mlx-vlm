"""APC lookup, semantic keys, lifecycle, and diagnostics."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm import apc as apc_module
from mlx_vlm.apc import (
    APCManager,
    DiskBlockStore,
    _hash_payload,
    _hash_tokens,
    classify_layer_for_apc,
    extract_prompt_cache_from_batch,
    harvest_blocks_from_batch_cache,
    hash_image_payload,
    make_warm_batch_exact_cache_multi,
    make_warm_batch_kv_cache_multi,
    make_warm_kv_cache,
    model_apc_mode,
    model_key_dependencies,
    self_check_model_apc,
    semantic_extra_hash,
    tenant_scoped_hash,
)
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    QuantizedKVCache,
)

# Lookup and cache lifecycle


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


# Semantic cache keys


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


# Trace logging and layout diagnostics

BLOCK_SIZE = 16
GROUP_SIZE = 64
BITS = 8


@pytest.fixture
def _clear_apc_trace_env(monkeypatch):
    monkeypatch.delenv("APC_TRACE", raising=False)
    yield
    monkeypatch.delenv("APC_TRACE", raising=False)


@pytest.mark.usefixtures("_clear_apc_trace_env")
class TestApcTrace:
    def test_reject_records_emit_trace(self, monkeypatch, caplog):
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
class TestClassifyLayer:
    def test_quantized_with_dequant_ok(self):
        # Last dim must be divisible by group_size for mx.quantize.
        c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        c.update_and_fetch(
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
            mx.random.normal((1, 2, 8, GROUP_SIZE)),
        )
        result = classify_layer_for_apc(c)
        assert result.status == "ok"

    def test_unsupported_opaque_type(self):
        class Bogus:
            pass

        result = classify_layer_for_apc(Bogus())
        assert result.status == "unsupported"
        assert result.reason


@pytest.mark.usefixtures("_clear_apc_trace_env")
class TestSelfCheckModel:
    def test_supported_model_ok(self, caplog):
        class FakeLang:
            def make_cache(self):
                return [
                    BatchRotatingKVCache(32, [0]),
                    BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS),
                    BatchKVCache([0]),
                ]

        with caplog.at_level(logging.INFO, logger="mlx_vlm.apc"):
            result = self_check_model_apc(FakeLang(), kv_bits=8.0)
        assert result.ok is True
        assert result.apc_mode == "exact"
        assert any("APC self-check ok" in r.message for r in caplog.records)

    def test_no_make_cache_not_ok(self, caplog):
        class NoCache:
            pass

        result = self_check_model_apc(NoCache())
        assert result.ok is False

    def test_does_not_raise_on_failure(self):
        class FakeLang:
            def make_cache(self):
                raise RuntimeError("boom")

        result = self_check_model_apc(FakeLang())
        assert result.ok is False
        assert result.notes
