from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm import apc as apc_module
from mlx_vlm.apc import (
    APCManager,
    DiskBlockStore,
    _copy_mlx_array,
    _hash_tokens,
    extract_prompt_cache_from_batch,
    from_env,
    harvest_blocks_from_batch_cache,
    hash_image_payload,
    make_warm_batch_exact_cache_multi,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
    make_warm_kv_cache,
    model_apc_mode,
    tenant_scoped_hash,
)


def _make_fake_kv(
    num_layers: int = 2,
    n_kv_heads: int = 1,
    seq_len: int = 32,
    head_dim: int = 4,
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


def _make_exact_row_cache(prefix_len: int):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    arrays = ArraysCache(size=2)
    arrays.cache = [
        mx.full((1, 3, 5), prefix_len, dtype=mx.float32),
        mx.full((1, 2, 4, 6), prefix_len + 10, dtype=mx.float32),
    ]
    kv = KVCache()
    kv.keys = mx.full((1, 1, prefix_len, 4), prefix_len + 20, dtype=mx.float32)
    kv.values = mx.full((1, 1, prefix_len, 4), prefix_len + 30, dtype=mx.float32)
    kv.offset = prefix_len
    mx.eval(arrays.cache + [kv.keys, kv.values])
    return [arrays, kv]


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


def test_image_hash_preserves_tensor_shape_and_dtype():
    flat_values = mx.arange(12, dtype=mx.float32)
    first_shape = flat_values.reshape(1, 3, 2, 2)
    second_shape = flat_values.reshape(1, 3, 1, 4)

    assert hash_image_payload(pixel_values=first_shape) != hash_image_payload(
        pixel_values=second_shape
    )
    assert hash_image_payload(pixel_values=first_shape) != hash_image_payload(
        pixel_values=first_shape.astype(mx.float16)
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


def test_lookup_stops_at_first_missing_or_mismatched_block():
    block_size = 16
    manager = APCManager(num_blocks=16, block_size=block_size)
    token_ids = list(range(3 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))
    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    manager.release(stored)

    changed = list(token_ids)
    changed[2 * block_size] = 999
    matched, matched_tokens = manager.lookup_prefix(changed)

    assert len(matched) == 2
    assert matched_tokens == 2 * block_size
    manager.release(matched)


def test_refcount_protects_blocks_from_lru_eviction():
    block_size = 16
    manager = APCManager(num_blocks=2, block_size=block_size)
    layer_keys, layer_values = _make_fake_kv(seq_len=block_size)

    first_tokens = list(range(block_size))
    second_tokens = list(range(100, 100 + block_size))
    third_tokens = list(range(200, 200 + block_size))

    first = manager.store_kv_blocks(first_tokens, layer_keys, layer_values)
    manager.release(first)
    held, held_tokens = manager.lookup_prefix(first_tokens)
    assert held_tokens == block_size

    second = manager.store_kv_blocks(second_tokens, layer_keys, layer_values)
    manager.release(second)
    third = manager.store_kv_blocks(third_tokens, layer_keys, layer_values)
    manager.release(third)

    still_held, matched_tokens = manager.lookup_prefix(first_tokens)
    assert matched_tokens == block_size
    manager.release(still_held)
    manager.release(held)


def test_extra_hash_isolates_image_and_tenant_prefixes():
    block_size = 16
    manager = APCManager(num_blocks=16, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    tenant_a = tenant_scoped_hash("tenant-a", hash_image_payload(image_ref="cat.jpg"))
    tenant_b = tenant_scoped_hash("tenant-b", hash_image_payload(image_ref="cat.jpg"))
    other_image = tenant_scoped_hash(
        "tenant-a", hash_image_payload(image_ref="dog.jpg")
    )

    stored = manager.store_kv_blocks(
        token_ids, layer_keys, layer_values, extra_hash=tenant_a
    )
    manager.release(stored)

    assert manager.lookup_prefix(token_ids, extra_hash=tenant_b)[1] == 0
    assert manager.lookup_prefix(token_ids, extra_hash=other_image)[1] == 0
    matched, matched_tokens = manager.lookup_prefix(token_ids, extra_hash=tenant_a)
    assert matched_tokens == len(token_ids)
    manager.release(matched)


def test_stored_block_tensors_are_decoupled_from_source_cache():
    block_size = 16
    manager = APCManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=2 * block_size)
    expected_key = mx.array(
        layer_keys[0][..., :block_size, :], dtype=layer_keys[0].dtype
    )

    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    layer_keys[0][..., :block_size, :] = mx.zeros_like(
        layer_keys[0][..., :block_size, :]
    )
    mx.eval(layer_keys[0], stored[0].keys[0])

    assert stored[0].keys[0].shape == expected_key.shape
    _assert_allclose(stored[0].keys[0], expected_key)
    manager.release(stored)


def test_copy_mlx_array_returns_a_distinct_materialized_array():
    source = mx.arange(8).reshape(1, 1, 8, 1)
    copied = _copy_mlx_array(source)
    mx.eval(source, copied)

    assert copied is not source
    _assert_allclose(copied, source)


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
        warm[0].keys[..., :expected_tokens, :],
        layer_keys[0][..., :expected_tokens, :],
    )
    _assert_allclose(
        warm[1].values[..., :expected_tokens, :],
        layer_values[1][..., :expected_tokens, :],
    )


def test_exact_batch_cache_merge_and_extract_supports_arrays_and_kv():
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    warm = _make_exact_row_cache(12)
    cold = _make_exact_row_cache(0)
    cold[0].cache = [None, None]
    cold[1].keys = None
    cold[1].values = None
    cold[1].offset = 0

    batch_cache, max_prefix = make_warm_batch_exact_cache_multi(
        [warm, cold],
        [12, 0],
    )

    assert max_prefix == 12
    assert batch_cache is not None
    assert isinstance(batch_cache[0], ArraysCache)
    assert isinstance(batch_cache[1], BatchKVCache)
    assert batch_cache[0].left_padding is None
    assert batch_cache[0].cache[0].shape == (2, 3, 5)
    assert batch_cache[1].keys.shape == (2, 1, 12, 4)
    _assert_allclose(batch_cache[0].cache[0][0:1], warm[0].cache[0])
    _assert_allclose(batch_cache[0].cache[0][1:2], mx.zeros_like(warm[0].cache[0]))
    _assert_allclose(batch_cache[1].keys[0:1], warm[1].keys)
    _assert_allclose(batch_cache[1].keys[1:2], mx.zeros_like(warm[1].keys))

    extracted = extract_prompt_cache_from_batch(batch_cache, 0)
    assert extracted is not None
    assert isinstance(extracted[0], ArraysCache)
    assert isinstance(extracted[1], KVCache)
    _assert_allclose(extracted[0].cache[1], warm[0].cache[1])
    _assert_allclose(extracted[1].keys, warm[1].keys)
    _assert_allclose(extracted[1].values, warm[1].values)
    assert extracted[1].offset == 12


def test_single_row_prompt_batch_exact_checkpoint_stores_without_extract():
    from mlx_vlm.generate.ar import PromptProcessingBatch
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    token_ids = list(range(12))
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


def test_apc_max_pool_tensors_keeps_disk_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("APC_MAX_POOL_TENSORS", "2")
    monkeypatch.setenv("APC_DISK_SHARD_MAX_BLOCKS", "2")

    block_size = 16
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(num_layers=2, seq_len=len(token_ids))

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=8, block_size=block_size, disk=disk)
    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)
    assert stored == []
    assert manager.stats_snapshot()["pool_used"] == 0
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="unit")
    manager = APCManager(num_blocks=8, block_size=block_size, disk=disk)
    warm, matched_tokens = manager.lookup_prefix_disk_cache(token_ids)

    assert warm is not None
    assert matched_tokens == len(token_ids)
    assert manager.stats_snapshot()["pool_used"] == 0
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


def test_from_env_respects_opt_in_and_disk_config(tmp_path, monkeypatch):
    monkeypatch.delenv("APC_ENABLED", raising=False)
    monkeypatch.delenv("APC_DISK_PATH", raising=False)
    assert from_env() is None

    monkeypatch.setenv("APC_ENABLED", "1")
    monkeypatch.setenv("APC_BLOCK_SIZE", "8")
    monkeypatch.setenv("APC_NUM_BLOCKS", "3")

    manager = from_env()
    assert manager is not None
    assert manager.block_size == 8
    assert manager.num_blocks == 3
    assert manager.disk is None
    manager.close()

    monkeypatch.setenv("APC_DISK_PATH", str(tmp_path))
    monkeypatch.setenv("APC_DISK_MAX_GB", "0.001")
    monkeypatch.setenv("APC_DISK_WORKERS", "1")

    manager = from_env("unit_model")
    assert manager is not None
    assert manager.disk is not None
    assert manager.disk.dir == tmp_path / "unit_model"
    assert manager.disk.max_bytes == int(0.001 * (1 << 30))
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
        token_ids,
        allow_memory_overlap=True,
    )
    assert warm is None
    assert matched_tokens == 0
    manager.close()


def test_make_warm_batch_kv_cache_single_row_shapes():
    block_size = 16
    manager = APCManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))
    blocks = manager.store_kv_blocks(token_ids, layer_keys, layer_values)

    caches = make_warm_batch_kv_cache(blocks)

    assert len(caches) == 2
    assert caches[0].keys.shape == (1, 1, 2 * block_size, 4)
    assert caches[0]._idx == 2 * block_size
    assert caches[0].offset.tolist() == [2 * block_size]
    assert caches[0].left_padding.tolist() == [0]
    manager.release(blocks)


def test_exact_cache_supports_mixed_kv_and_arrays_cache():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    block_size = 16
    manager = APCManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(2 * block_size))

    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)
    arrays = ArraysCache(size=2)
    arrays[0] = mx.ones((1, 3, 4))
    arrays[1] = mx.ones((1, 2, 3)) * 3

    assert manager.store_exact_cache(token_ids, [arrays, kv], extra_hash=7)
    warm, matched_tokens = manager.lookup_exact_cache(
        token_ids + [999],
        extra_hash=7,
    )

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert warm[0] is not arrays
    assert warm[1] is not kv
    _assert_allclose(warm[0][0], arrays[0])
    _assert_allclose(warm[0][1], arrays[1])
    _assert_allclose(warm[1].keys[..., : len(token_ids), :], kv.keys)
    assert warm[1].offset == len(token_ids)
    assert warm[1].keys.shape[2] >= len(token_ids) + 1

    arrays[0] = mx.zeros_like(arrays[0])
    kv.keys[..., :, :] = mx.zeros_like(kv.keys)
    _assert_allclose(warm[0][0], mx.ones((1, 3, 4)))
    _assert_allclose(
        warm[1].keys[..., : len(token_ids), :],
        mx.ones((1, 1, len(token_ids), 2)),
    )


def test_exact_cache_supports_rotating_and_chunked_kv_cache():
    from mlx_vlm.models.cache import ChunkedKVCache, KVCache, RotatingKVCache

    block_size = 16
    manager = APCManager(num_blocks=4, block_size=block_size)
    token_ids = list(range(48))

    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)

    rotating = RotatingKVCache(max_size=8, keep=2)
    rotating.keys = mx.arange(1 * 1 * 8 * 2, dtype=mx.float32).reshape(1, 1, 8, 2)
    rotating.values = rotating.keys + 100
    rotating.offset = len(token_ids)
    rotating._idx = 5

    chunked = ChunkedKVCache(chunk_size=12)
    chunked.keys = mx.ones((1, 1, 12, 2)) * 3
    chunked.values = mx.ones((1, 1, 12, 2)) * 4
    chunked.offset = len(token_ids)
    chunked.start_position = len(token_ids) - 12

    assert manager.store_exact_cache(
        token_ids,
        [kv, rotating, chunked],
        extra_hash=13,
    )
    warm, matched_tokens = manager.lookup_exact_cache(
        token_ids + [999],
        extra_hash=13,
    )

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert isinstance(warm[1], RotatingKVCache)
    assert warm[1].max_size == rotating.max_size
    assert warm[1].keep == rotating.keep
    assert warm[1].offset == rotating.offset
    assert warm[1]._idx == rotating._idx
    _assert_allclose(warm[1].keys, rotating.keys)
    _assert_allclose(warm[1].values, rotating.values)
    assert isinstance(warm[2], ChunkedKVCache)
    assert warm[2].chunk_size == chunked.chunk_size
    assert warm[2].offset == chunked.offset
    assert warm[2].start_position == chunked.start_position
    _assert_allclose(warm[2].keys, chunked.keys)
    _assert_allclose(warm[2].values, chunked.values)


def test_exact_cache_disk_restore_rebuilds_index(tmp_path, monkeypatch):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")

    token_ids = list(range(40))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)
    arrays = ArraysCache(size=2)
    arrays[0] = mx.ones((1, 3, 4))
    arrays[1] = mx.ones((1, 2, 3)) * 3

    disk = DiskBlockStore(tmp_path, namespace="exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [arrays, kv], extra_hash=11)
    disk._q.join()
    assert disk.num_exact_indexed == 1
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    warm, matched_tokens = manager.lookup_exact_cache(
        token_ids + [999],
        extra_hash=11,
    )

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert manager.stats_snapshot()["disk_hits"] == 1
    assert manager.stats_snapshot()["disk_exact_indexed"] == 1
    _assert_allclose(warm[0][0], arrays[0])
    _assert_allclose(warm[0][1], arrays[1])
    _assert_allclose(warm[1].keys[..., : len(token_ids), :], kv.keys)
    assert warm[1].offset == len(token_ids)
    assert warm[1].keys.shape[2] >= len(token_ids) + 1
    manager.close()


def test_exact_cache_disk_restore_preserves_rotating_kv(tmp_path, monkeypatch):
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")

    token_ids = list(range(40))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)
    rotating = RotatingKVCache(max_size=8, keep=0)
    rotating.keys = mx.arange(1 * 1 * 8 * 2, dtype=mx.float32).reshape(1, 1, 8, 2)
    rotating.values = rotating.keys + 10
    rotating.offset = len(token_ids)
    rotating._idx = 3

    disk = DiskBlockStore(tmp_path, namespace="rotating-exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [kv, rotating], extra_hash=17)
    disk._q.join()
    manager.close()

    disk = DiskBlockStore(tmp_path, namespace="rotating-exact")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    warm, matched_tokens = manager.lookup_exact_cache(
        token_ids + [999],
        extra_hash=17,
    )

    assert matched_tokens == len(token_ids)
    assert warm is not None
    assert manager.stats_snapshot()["disk_hits"] == 1
    assert isinstance(warm[1], RotatingKVCache)
    assert warm[1].max_size == rotating.max_size
    assert warm[1].keep == rotating.keep
    assert warm[1].offset == rotating.offset
    assert warm[1]._idx == rotating._idx
    _assert_allclose(warm[1].keys, rotating.keys)
    _assert_allclose(warm[1].values, rotating.values)
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


def test_mixed_warm_batch_cache_left_pads_cold_and_short_rows():
    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    full_tokens = list(range(2 * block_size))
    short_tokens = list(range(100, 100 + block_size))
    full_keys, full_values = _make_fake_kv(seq_len=len(full_tokens))
    short_keys, short_values = _make_fake_kv(seq_len=len(short_tokens))

    full_blocks = manager.store_kv_blocks(full_tokens, full_keys, full_values)
    short_blocks = manager.store_kv_blocks(short_tokens, short_keys, short_values)
    picks = [
        {"matched_blocks": full_blocks, "prefix_len": 2 * block_size},
        None,
        {"matched_blocks": short_blocks, "prefix_len": block_size},
    ]

    caches, max_prefix = make_warm_batch_kv_cache_multi(picks, num_layers=2)

    assert max_prefix == 2 * block_size
    assert len(caches) == 2
    assert caches[0].keys.shape == (3, 1, 2 * block_size, 4)
    assert caches[0]._idx == 2 * block_size
    assert caches[0].offset.tolist() == [2 * block_size, 0, block_size]
    assert caches[0].left_padding.tolist() == [0, 2 * block_size, block_size]

    manager.release(full_blocks + short_blocks)


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
        harvest_manager,
        caches,
        batch_idx=1,
        full_token_ids=short_token_ids,
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


def test_multimodal_token_ids_from_config():
    config = SimpleNamespace(
        image_token_id=None,
        image_token_index=42,
        video_token_id=77,
        video_token_index=None,
    )

    assert apc_module.multimodal_token_ids_from_config(config) == {42, 77}


def test_media_token_spans_are_contiguous_ranges():
    token_ids = [1, 42, 42, 2, 77, 77, 77, 3]

    assert apc_module.media_token_spans(token_ids, {42, 77}) == (
        (1, 3),
        (4, 7),
    )


def test_prefix_must_leave_text_only_suffix():
    token_ids = [1, 42, 42, 2, 77, 77, 3]

    assert not apc_module.prefix_leaves_text_only_suffix(token_ids, 3, {42, 77})
    assert not apc_module.prefix_leaves_text_only_suffix(token_ids, 5, {42, 77})
    assert apc_module.prefix_leaves_text_only_suffix(token_ids, 6, {42, 77})
    assert apc_module.prefix_leaves_text_only_suffix(token_ids, 7, {42, 77})


def test_adjust_prefix_moves_after_media_span():
    token_ids = [1] + [42] * 3072 + [2] * 30

    assert (
        apc_module.adjust_prefix_to_text_suffix_boundary(
            token_ids,
            desired_prefix_len=2958,
            media_token_ids={42},
            max_prefix_tokens=len(token_ids) - 1,
        )
        == 3073
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


def test_exact_disk_hit_is_promoted_to_memory(tmp_path, monkeypatch):
    """After a disk restore, the entry is written to _exact_cache so the next
    identical request is served from memory (disk_hits stays unchanged)."""
    from mlx_vlm.models.cache import KVCache

    token_ids = list(range(40))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)

    # Write to disk only (memory cache disabled).
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="promotion")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [kv], extra_hash=3)
    disk._q.join()
    manager.close()

    # Restart with an in-memory cache (4 slots) so promotion has somewhere to land.
    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
    disk = DiskBlockStore(tmp_path, namespace="promotion")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)

    # First lookup: cold start, disk hit.
    warm1, matched1 = manager.lookup_exact_cache(token_ids + [999], extra_hash=3)
    assert matched1 == len(token_ids)
    assert warm1 is not None
    snap1 = manager.stats_snapshot()
    assert snap1["disk_hits"] == 1
    assert snap1["exact_hits"] == 1

    # Second lookup: must be served from memory; disk_hits must not increase.
    warm2, matched2 = manager.lookup_exact_cache(token_ids + [999], extra_hash=3)
    assert matched2 == len(token_ids)
    assert warm2 is not None
    snap2 = manager.stats_snapshot()
    assert snap2["disk_hits"] == 1, "second hit should come from memory, not disk"
    assert snap2["exact_hits"] == 2

    # The two returned caches must be independent objects (cloned, not the same).
    assert warm1 is not warm2

    manager.close()


def test_exact_disk_hit_promotion_skipped_when_memory_disabled(tmp_path, monkeypatch):
    """When _exact_cache_max == 0 the disk hit still works; no promotion attempted."""
    from mlx_vlm.models.cache import KVCache

    token_ids = list(range(20))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="nopromo")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [kv], extra_hash=5)
    disk._q.join()
    manager.close()

    # Restart also with memory disabled.
    disk = DiskBlockStore(tmp_path, namespace="nopromo")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)

    warm, matched = manager.lookup_exact_cache(token_ids + [999], extra_hash=5)
    assert matched == len(token_ids)
    assert warm is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Second lookup should hit disk again (memory still disabled).
    warm2, matched2 = manager.lookup_exact_cache(token_ids + [999], extra_hash=5)
    assert matched2 == len(token_ids)
    assert warm2 is not None
    assert manager.stats_snapshot()["disk_hits"] == 2

    manager.close()


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


def test_exact_disk_hit_promotion_clone_is_independent_of_returned_cache(
    tmp_path, monkeypatch
):
    """The clone stored in _exact_cache must be a separate object from the
    cache returned to the caller.  Simulating token-generation mutations on the
    returned cache (advancing offset) must not corrupt the stored entry."""
    from mlx_vlm.models.cache import KVCache

    token_ids = list(range(25))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2))
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 2
    kv.offset = len(token_ids)

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="clone-independence")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [kv], extra_hash=0)
    disk._q.join()
    manager.close()

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
    disk = DiskBlockStore(tmp_path, namespace="clone-independence")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)

    # First lookup: disk hit + promotion.
    warm1, matched = manager.lookup_exact_cache(token_ids + [999], extra_hash=0)
    assert matched == len(token_ids)
    assert warm1 is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Simulate generate_step mutating the returned cache in-place.
    original_offset = warm1[0].offset
    warm1[0].offset += 10  # as if 10 tokens were generated

    # Second lookup: must come from memory, with the original offset intact.
    warm2, _ = manager.lookup_exact_cache(token_ids + [999], extra_hash=0)
    assert warm2 is not None
    assert manager.stats_snapshot()["disk_hits"] == 1  # served from memory
    assert (
        warm2[0].offset == original_offset
    ), "stored clone offset was corrupted by mutation of the returned cache"

    manager.close()


def test_exact_disk_hit_promotion_with_nonzero_extra_hash(tmp_path, monkeypatch):
    """Promotion must use the correct key when extra_hash != 0, so the
    promoted entry is found on subsequent lookups with the same extra_hash and
    not accidentally served for a different extra_hash."""
    from mlx_vlm.models.cache import KVCache

    token_ids = list(range(30))
    kv = KVCache()
    kv.keys = mx.ones((1, 1, len(token_ids), 2)) * 5
    kv.values = mx.ones((1, 1, len(token_ids), 2)) * 6
    kv.offset = len(token_ids)

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "0")
    disk = DiskBlockStore(tmp_path, namespace="extra-hash")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)
    assert manager.store_exact_cache(token_ids, [kv], extra_hash=42)
    disk._q.join()
    manager.close()

    monkeypatch.setenv("APC_EXACT_CACHE_ENTRIES", "4")
    disk = DiskBlockStore(tmp_path, namespace="extra-hash")
    manager = APCManager(num_blocks=1, block_size=16, disk=disk)

    # Disk hit with the correct extra_hash -> promoted.
    warm1, matched1 = manager.lookup_exact_cache(token_ids + [999], extra_hash=42)
    assert matched1 == len(token_ids)
    assert warm1 is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Second lookup with same extra_hash -> memory hit, no new disk hit.
    warm2, matched2 = manager.lookup_exact_cache(token_ids + [999], extra_hash=42)
    assert matched2 == len(token_ids)
    assert warm2 is not None
    assert manager.stats_snapshot()["disk_hits"] == 1

    # Lookup with a different extra_hash -> must miss (different namespace).
    warm_wrong, matched_wrong = manager.lookup_exact_cache(
        token_ids + [999], extra_hash=99
    )
    assert matched_wrong == 0
    assert warm_wrong is None

    manager.close()


def _tiny_exact_cache(tokens):
    from mlx_vlm.models.cache import ArraysCache

    c = ArraysCache(size=1)
    c[0] = mx.zeros((1, 2, max(1, len(tokens)), 8))
    return [c]


def test_partition_splits_a_hybrid_cache_by_pageability():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    cache = [KVCache(), ArraysCache(size=1), KVCache(), ArraysCache(size=1)]

    pageable, checkpointed = apc_module.partition_cache_by_pageability(cache)

    assert pageable == [0, 2]
    assert checkpointed == [1, 3]


def test_partition_of_a_dense_cache_is_every_layer():
    from mlx_vlm.models.cache import KVCache

    pageable, checkpointed = apc_module.partition_cache_by_pageability(
        [KVCache() for _ in range(4)]
    )

    assert pageable == [0, 1, 2, 3]
    assert checkpointed == []


def test_dense_store_records_the_dense_layer_range():
    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    stored = manager.store_kv_blocks(token_ids, layer_keys, layer_values)

    assert stored
    assert all(b.layer_indices == tuple(range(len(layer_keys))) for b in stored)
    manager.release(stored)


def test_sparse_store_records_the_layers_it_came_from():
    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids))

    stored = manager.store_kv_blocks(
        token_ids, layer_keys, layer_values, layer_indices=[2, 5]
    )

    assert stored
    assert all(b.layer_indices == (2, 5) for b in stored)
    manager.release(stored)


def test_harvest_skips_stateful_layers_only_when_asked():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    block_size = 16
    token_ids = list(range(2 * block_size))
    kv = KVCache()
    slab = mx.zeros((1, 2, len(token_ids), 32))
    kv.update_and_fetch(slab, slab)
    state = ArraysCache(size=1)
    state[0] = mx.zeros((1, 2, 4, 32))
    mixed = [state, kv]

    strict = APCManager(num_blocks=8, block_size=block_size)
    assert apc_module.harvest_blocks_from_batch_cache(strict, mixed, token_ids) == []

    partial = APCManager(num_blocks=8, block_size=block_size)
    blocks = apc_module.harvest_blocks_from_batch_cache(
        partial, mixed, token_ids, allow_partial_layers=True
    )

    assert blocks
    assert all(b.layer_indices == (1,) for b in blocks)
    partial.release(blocks)


def test_exact_lookup_picks_the_longest_stored_prefix():
    manager = APCManager(num_blocks=8, block_size=16)
    tokens = list(range(1, 600))
    for boundary in (128, 256, 384):
        manager.store_exact_cache(
            tokens[:boundary], _tiny_exact_cache(tokens[:boundary])
        )

    cache, reused = manager.lookup_exact_cache(tokens)

    assert cache is not None
    assert reused == 384


def test_exact_lookup_ignores_a_prefix_that_diverges():
    manager = APCManager(num_blocks=8, block_size=16)
    stored = list(range(1, 600))
    for boundary in (128, 256):
        manager.store_exact_cache(
            stored[:boundary], _tiny_exact_cache(stored[:boundary])
        )

    divergent = stored[:200] + [9999] * 400
    cache, reused = manager.lookup_exact_cache(divergent)

    assert reused == 128


def test_exact_lookup_honours_the_minimum_prefix():
    manager = APCManager(num_blocks=8, block_size=16)
    tokens = list(range(1, 600))
    for boundary in (128, 256):
        manager.store_exact_cache(
            tokens[:boundary], _tiny_exact_cache(tokens[:boundary])
        )

    _, reused = manager.lookup_exact_cache(tokens, min_prefix_tokens=200)

    assert reused == 256


def _mixed_template():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    return [ArraysCache(size=1), KVCache(), ArraysCache(size=1), KVCache()]


def test_composite_assembly_puts_paged_layers_back_in_place():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids), num_layers=2)
    stored = manager.store_kv_blocks(
        token_ids, layer_keys, layer_values, layer_indices=[1, 3]
    )
    template = _mixed_template()

    out = apc_module.make_warm_composite_cache(stored, template)

    assert out is not None
    assert isinstance(out[1], KVCache) and isinstance(out[3], KVCache)
    assert isinstance(out[0], ArraysCache) and isinstance(out[2], ArraysCache)
    assert out[0] is template[0] and out[2] is template[2]
    manager.release(stored)


def test_composite_assembly_rejects_blocks_from_different_layouts():
    block_size = 16
    manager = APCManager(num_blocks=16, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids), num_layers=2)
    a = manager.store_kv_blocks(
        token_ids, layer_keys, layer_values, layer_indices=[1, 3]
    )
    other = list(range(500, 500 + 2 * block_size))
    b = manager.store_kv_blocks(other, layer_keys, layer_values, layer_indices=[0, 2])

    assert apc_module.make_warm_composite_cache(a + b, _mixed_template()) is None
    manager.release(a)
    manager.release(b)


def test_composite_assembly_rejects_an_index_past_the_template():
    block_size = 16
    manager = APCManager(num_blocks=8, block_size=block_size)
    token_ids = list(range(2 * block_size))
    layer_keys, layer_values = _make_fake_kv(seq_len=len(token_ids), num_layers=2)
    stored = manager.store_kv_blocks(
        token_ids, layer_keys, layer_values, layer_indices=[1, 99]
    )

    assert apc_module.make_warm_composite_cache(stored, _mixed_template()) is None
    manager.release(stored)


def _exact_index_is_consistent(manager):
    from collections import Counter

    expected = Counter(len(e.token_ids) for e in manager._exact_cache.values())
    return manager._exact_lengths == sorted(
        expected
    ) and manager._exact_length_counts == dict(expected)


def _store_snapshot(manager, tokens):
    from mlx_vlm.models.cache import KVCache

    tokens = list(tokens)
    entry = KVCache()
    entry.keys = mx.zeros((1, 2, len(tokens), 8))
    entry.values = mx.zeros((1, 2, len(tokens), 8))
    entry.offset = len(tokens)
    return manager.store_exact_cache(tokens, [entry])


def test_exact_length_index_tracks_eviction():
    manager = APCManager(num_blocks=64, block_size=16)
    manager._exact_cache_max = 3
    manager.exact_cache_min_tokens = 1

    for length in (32, 64, 96, 128, 160):
        _store_snapshot(manager, range(length))

    assert len(manager._exact_cache) == 3
    assert _exact_index_is_consistent(manager)
    assert manager._exact_lengths == [96, 128, 160]


def test_exact_length_index_does_not_double_count_a_rewrite():
    manager = APCManager(num_blocks=64, block_size=16)
    manager._exact_cache_max = 4
    manager.exact_cache_min_tokens = 1
    tokens = list(range(48))

    _store_snapshot(manager, tokens)
    _store_snapshot(manager, tokens)

    assert len(manager._exact_cache) == 1
    assert manager._exact_lengths == [48]
    assert _exact_index_is_consistent(manager)


def test_exact_length_index_keeps_duplicate_lengths_until_all_are_gone():
    manager = APCManager(num_blocks=64, block_size=16)
    manager._exact_cache_max = 4
    manager.exact_cache_min_tokens = 1

    _store_snapshot(manager, range(32))
    _store_snapshot(manager, range(1000, 1032))

    assert manager._exact_lengths == [32]
    assert manager._exact_length_counts == {32: 2}

    manager._exact_cache_max = 1
    _store_snapshot(manager, range(2000, 2032))

    assert _exact_index_is_consistent(manager)


def test_clear_empties_the_exact_length_index():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    _store_snapshot(manager, range(64))

    manager.clear()

    assert manager._exact_lengths == []
    assert manager._exact_length_counts == {}


def test_longest_prefix_still_wins_after_eviction():
    manager = APCManager(num_blocks=64, block_size=16)
    manager._exact_cache_max = 2
    manager.exact_cache_min_tokens = 1
    tokens = list(range(256))

    _store_snapshot(manager, tokens[:64])
    _store_snapshot(manager, tokens[:128])
    _store_snapshot(manager, tokens[:192])

    cache, prefix_len = manager.lookup_exact_cache(tokens)

    assert cache is not None
    assert prefix_len == 192


def test_partial_harvest_skips_non_pageable_layers_that_still_expose_kv():
    """Four cache types expose .keys/.values but cannot be paged.

    Deciding by "does this layer yield KV" would harvest a ring buffer or a
    chunked layout into blocks as if it were linear, trimmable KV.
    """
    from mlx_vlm.models.cache import (
        ArraysCache,
        BufferedRotatingKVCache,
        ChunkedKVCache,
        ConcatenateKVCache,
        KVCache,
        RotatingKVCache,
    )

    factories = {
        "rotating": lambda: RotatingKVCache(max_size=64, keep=0),
        "chunked": lambda: ChunkedKVCache(64),
        "concatenate": lambda: ConcatenateKVCache(),
        "buffered_rotating": lambda: BufferedRotatingKVCache(max_size=64, keep=0),
    }

    for label, factory in factories.items():
        non_pageable = factory()
        non_pageable.update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
        plain = KVCache()
        plain.update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
        cache = [plain, non_pageable, ArraysCache(2)]

        manager = APCManager(num_blocks=64, block_size=16)
        blocks = harvest_blocks_from_batch_cache(
            manager, cache, list(range(64)), allow_partial_layers=True
        )

        assert blocks, label
        pageable, _ = apc_module.partition_cache_by_pageability(cache)
        assert blocks[0].layer_indices == tuple(pageable) == (0,), label
        assert not apc_module._cache_entry_supports_block_apc(non_pageable), label
        assert apc_module.layer_kv_for_apc(non_pageable)[0] is not None, label
        manager.release(blocks)


def test_partial_harvest_agrees_with_the_partition_helper():
    from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

    entries = []
    for index in range(8):
        if index % 3 == 0:
            entry = KVCache()
        elif index % 3 == 1:
            entry = RotatingKVCache(max_size=64, keep=0)
        else:
            entries.append(ArraysCache(2))
            continue
        entry.update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
        entries.append(entry)

    manager = APCManager(num_blocks=256, block_size=16)
    blocks = harvest_blocks_from_batch_cache(
        manager, entries, list(range(64)), allow_partial_layers=True
    )
    pageable, other = apc_module.partition_cache_by_pageability(entries)

    assert blocks[0].layer_indices == tuple(pageable)
    assert set(pageable).isdisjoint(other)
    manager.release(blocks)


def test_checkpoint_stride_uses_the_prefill_chunk_when_state_is_cheap():
    stride = apc_module.checkpoint_stride(
        state_bytes=80 * 1024, prompt_tokens=12000, budget_bytes=128 << 20
    )

    assert stride == 2048


def test_checkpoint_stride_coarsens_when_the_budget_is_tight():
    stride = apc_module.checkpoint_stride(
        state_bytes=40 << 20, prompt_tokens=12000, budget_bytes=128 << 20
    )

    assert stride == 4096
    assert -(-12000 // stride) * (40 << 20) <= (128 << 20)


def test_checkpoint_stride_returns_zero_when_one_checkpoint_will_not_fit():
    stride = apc_module.checkpoint_stride(
        state_bytes=348 << 20, prompt_tokens=12000, budget_bytes=128 << 20
    )

    assert stride == 0


def test_checkpoint_stride_is_always_a_whole_number_of_prefill_chunks():
    for state_mb in (1, 8, 17, 64, 127):
        stride = apc_module.checkpoint_stride(
            state_bytes=state_mb << 20,
            prompt_tokens=100_000,
            budget_bytes=128 << 20,
            prefill_chunk=512,
        )
        assert stride == 0 or stride % 512 == 0


def test_checkpoint_stride_stays_inside_the_budget_it_is_given():
    for state_mb in (1, 8, 17, 64, 127, 200):
        for prompt in (1000, 12000, 60000):
            budget = 128 << 20
            stride = apc_module.checkpoint_stride(
                state_bytes=state_mb << 20, prompt_tokens=prompt, budget_bytes=budget
            )
            if stride == 0:
                continue
            checkpoints = -(-prompt // stride)
            assert checkpoints * (state_mb << 20) <= budget


def test_checkpoint_stride_rejects_a_nonpositive_chunk():
    try:
        apc_module.checkpoint_stride(1, 100, budget_bytes=1 << 20, prefill_chunk=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_checkpoint_stride_declines_an_empty_budget():
    assert apc_module.checkpoint_stride(1024, 12000, budget_bytes=0) == 0


def test_checkpoint_state_bytes_counts_only_the_layers_that_cannot_page():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    plain = KVCache()
    plain.update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
    state = ArraysCache(1)
    state[0] = mx.zeros((1, 8, 16), dtype=mx.float32)

    measured = apc_module.checkpoint_state_bytes([plain, state])

    assert measured == 8 * 16 * 4


def test_composite_cache_places_the_state_entries_too():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    tokens = list(range(64))
    warm = [KVCache(), ArraysCache(1), KVCache()]
    for entry in (warm[0], warm[2]):
        entry.update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
    manager = APCManager(num_blocks=64, block_size=16)
    blocks = harvest_blocks_from_batch_cache(
        manager, warm, tokens, allow_partial_layers=True
    )
    state = ArraysCache(1)
    state[0] = mx.ones((1, 2, 2))

    template = [KVCache(), ArraysCache(1), KVCache()]
    out = apc_module.make_warm_composite_cache(blocks, template, state_entries=[state])

    assert out is not None
    assert out[1] is state
    assert out[0] is not template[0] and out[2] is not template[2]
    manager.release(blocks)


def test_composite_cache_refuses_a_state_list_of_the_wrong_length():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    warm = [KVCache(), ArraysCache(1)]
    warm[0].update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
    manager = APCManager(num_blocks=64, block_size=16)
    blocks = harvest_blocks_from_batch_cache(
        manager, warm, list(range(64)), allow_partial_layers=True
    )

    out = apc_module.make_warm_composite_cache(
        blocks,
        [KVCache(), ArraysCache(1)],
        state_entries=[ArraysCache(1), ArraysCache(1)],
    )

    assert out is None
    manager.release(blocks)


def test_composite_cache_refuses_blocks_whose_layout_the_template_disagrees_with():
    from mlx_vlm.models.cache import ArraysCache, KVCache

    warm = [KVCache(), ArraysCache(1)]
    warm[0].update_and_fetch(mx.zeros((1, 1, 64, 4)), mx.zeros((1, 1, 64, 4)))
    manager = APCManager(num_blocks=64, block_size=16)
    blocks = harvest_blocks_from_batch_cache(
        manager, warm, list(range(64)), allow_partial_layers=True
    )

    out = apc_module.make_warm_composite_cache(blocks, [ArraysCache(1), KVCache()])

    assert out is None
    manager.release(blocks)


def test_checkpoint_fits_enforces_the_budget_a_growing_state_would_break():
    budget = 128 << 20
    stride_priced_at = 40 << 20

    assert (
        apc_module.checkpoint_stride(stride_priced_at, 12000, budget_bytes=budget)
        == 4096
    )

    stored = 0
    grown = [40 << 20, 60 << 20, 90 << 20]
    taken = 0
    for cost in grown:
        if not apc_module.checkpoint_fits(stored, cost, budget_bytes=budget):
            break
        stored += cost
        taken += 1

    assert taken == 2
    assert stored <= budget


def test_checkpoint_fits_declines_an_empty_budget():
    assert not apc_module.checkpoint_fits(0, 1, budget_bytes=0)


def test_checkpoint_fits_reads_the_budget_from_the_environment(monkeypatch):
    monkeypatch.setenv("APC_CHECKPOINT_BUDGET_MB", "1")

    assert apc_module.checkpoint_fits(0, 1 << 20)
    assert not apc_module.checkpoint_fits(0, (1 << 20) + 1)


def _plan_for_mode(manager, tokens, mode):
    return apc_module.apc_lookup_plan(
        manager,
        tokens,
        extra_hash=0,
        apc_mode=mode,
        safe_lookup_min=0,
        suffix_is_text_only=lambda prefix_len: True,
        prefix_has_media=lambda prefix_len: False,
    )


def test_an_unknown_apc_mode_never_takes_the_block_path():
    """A third mode must decline, not fall through to block reuse.

    The block path assembles a warm cache assuming every layer came from
    blocks, which is wrong for any cache that only pages some of them.
    """
    manager = APCManager(num_blocks=64, block_size=16)
    tokens = list(range(64))
    keys = [mx.zeros((1, 1, len(tokens), 4))]
    values = [mx.zeros((1, 1, len(tokens), 4))]
    manager.release(manager.store_kv_blocks(tokens, keys, values))
    probe = tokens + list(range(900, 908))

    assert _plan_for_mode(manager, probe, "block") is not None
    for mode in ("composite", "future-mode", ""):
        assert _plan_for_mode(manager, probe, mode) is None, mode


def test_block_mode_still_reuses_after_the_membership_guard():
    manager = APCManager(num_blocks=64, block_size=16)
    tokens = list(range(64))
    keys = [mx.zeros((1, 1, len(tokens), 4))]
    values = [mx.zeros((1, 1, len(tokens), 4))]
    manager.release(manager.store_kv_blocks(tokens, keys, values))

    plan = _plan_for_mode(manager, tokens + list(range(900, 908)), "block")

    assert plan is not None and plan["prefix_len"] > 0
    manager.release(plan["matched_blocks"])


class _FakeLM:
    def __init__(self, entries):
        self._entries = entries

    def make_cache(self):
        from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache

        built = []
        for kind in self._entries:
            if kind == "kv":
                built.append(KVCache())
            elif kind == "state":
                built.append(ArraysCache(1))
            else:
                built.append(RotatingKVCache(max_size=64, keep=0))
        return built


def test_a_mixed_cache_stays_on_exact_mode_until_composite_is_enabled(monkeypatch):
    monkeypatch.delenv("APC_COMPOSITE", raising=False)

    assert model_apc_mode(_FakeLM(["kv", "state", "kv"])) == "exact"


def test_a_mixed_cache_selects_composite_when_enabled(monkeypatch):
    monkeypatch.setenv("APC_COMPOSITE", "1")

    assert model_apc_mode(_FakeLM(["kv", "state", "kv"])) == "composite"
    assert model_apc_mode(_FakeLM(["kv", "rotating"])) == "composite"


def test_a_dense_cache_is_block_mode_whether_or_not_composite_is_enabled(monkeypatch):
    monkeypatch.setenv("APC_COMPOSITE", "1")

    assert model_apc_mode(_FakeLM(["kv", "kv"])) == "block"


def test_a_cache_with_nothing_to_page_stays_exact(monkeypatch):
    monkeypatch.setenv("APC_COMPOSITE", "1")

    assert model_apc_mode(_FakeLM(["state", "state"])) == "exact"


def _lens(lens, total=100, callback=lambda n, c: None):
    from mlx_vlm.generate.ar import _prompt_cache_checkpoint_lens

    return _prompt_cache_checkpoint_lens(callback, lens, total)


def test_checkpoint_lens_accepts_a_single_position():
    assert _lens(32) == [32]


def test_checkpoint_lens_accepts_and_sorts_many():
    assert _lens([64, 16, 32]) == [16, 32, 64]


def test_checkpoint_lens_drops_positions_outside_the_prompt():
    assert _lens([0, -5, 50, 100, 250], total=100) == [50]


def test_checkpoint_lens_deduplicates():
    assert _lens([32, 32, 64]) == [32, 64]


def test_checkpoint_lens_is_empty_without_a_callback():
    assert _lens([10, 20], callback=None) == []
    assert _lens(None) == []


def _hybrid_cache(seq_len=64, state_shape=(1, 4, 4)):
    from mlx_vlm.models.cache import ArraysCache, KVCache

    kv = KVCache()
    kv.update_and_fetch(mx.zeros((1, 1, seq_len, 4)), mx.zeros((1, 1, seq_len, 4)))
    state = ArraysCache(1)
    state[0] = mx.ones(state_shape)
    return [kv, state]


def _checkpointer(manager, tokens, **kw):
    return apc_module.CompositeCheckpointer(manager, tokens, prefill_chunk=16, **kw)


def test_checkpointer_picks_a_stride_on_its_first_boundary():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    tokens = list(range(64))
    cp = _checkpointer(manager, tokens, budget_bytes=1 << 20)

    cp(16, _hybrid_cache())

    assert cp.stride == 16
    assert cp.stored == 1
    assert cp.decline is None


def test_checkpointer_declines_when_one_checkpoint_will_not_fit():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    cp = _checkpointer(manager, list(range(64)), budget_bytes=8)

    cp(16, _hybrid_cache())

    assert cp.stride == 0
    assert cp.stored == 0
    assert cp.decline == apc_module.CompositeDecline.BUDGET


def test_checkpointer_only_stores_on_stride_multiples():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    cp = _checkpointer(manager, list(range(128)), budget_bytes=1 << 20)

    cp(16, _hybrid_cache())
    cp(24, _hybrid_cache())
    cp(32, _hybrid_cache())

    assert cp.stride == 16
    assert cp.stored == 2


def test_checkpointer_stops_when_a_growing_state_exhausts_the_budget():
    """The stride is priced once, but a sliding window keeps growing.

    Pricing off the first, small measurement yields a fine stride; the running
    total must still stop before the budget is exceeded.
    """
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    small = apc_module.checkpoint_state_bytes(_hybrid_cache(state_shape=(1, 2, 2)))
    budget = small * 4
    cp = _checkpointer(manager, list(range(64)), budget_bytes=budget)

    cp(16, _hybrid_cache(state_shape=(1, 2, 2)))
    assert cp.stride == 16 and cp.stored == 1

    cp(32, _hybrid_cache(state_shape=(1, 8, 8)))

    assert cp.stored == 1
    assert cp.stored_bytes <= budget
    assert cp.decline == apc_module.CompositeDecline.BUDGET_EXHAUSTED


def test_checkpointer_declines_a_cache_with_nothing_to_page():
    from mlx_vlm.models.cache import ArraysCache

    manager = APCManager(num_blocks=64, block_size=16)
    cp = _checkpointer(manager, list(range(64)))

    cp(16, [ArraysCache(1), ArraysCache(1)])

    assert cp.decline == apc_module.CompositeDecline.NOT_MIXED
    assert cp.stored == 0


def _composite_plan(manager, tokens, template):
    return apc_module.apc_lookup_plan(
        manager,
        tokens,
        extra_hash=0,
        apc_mode="composite",
        safe_lookup_min=0,
        suffix_is_text_only=lambda n: True,
        prefix_has_media=lambda n: False,
        cache_template=template,
    )


def test_composite_lookup_declines_without_a_template():
    manager = APCManager(num_blocks=64, block_size=16)

    assert _composite_plan(manager, list(range(64)), None) is None
    assert manager.stats_snapshot()["composite_declines"] == {
        apc_module.CompositeDecline.LAYOUT_MISMATCH: 1
    }


def test_composite_lookup_declines_when_nothing_is_stored():
    manager = APCManager(num_blocks=64, block_size=16)

    assert _composite_plan(manager, list(range(64)), _hybrid_cache()) is None
    assert manager.stats_snapshot()["composite_declines"] == {
        apc_module.CompositeDecline.NO_BLOCKS: 1
    }


def test_composite_lookup_declines_when_only_blocks_are_stored():
    manager = APCManager(num_blocks=64, block_size=16)
    tokens = list(range(64))
    warm = _hybrid_cache()
    manager.release(
        harvest_blocks_from_batch_cache(
            manager, warm, tokens, allow_partial_layers=True
        )
    )

    plan = _composite_plan(manager, tokens + list(range(900, 908)), _hybrid_cache())

    assert plan is None
    assert manager.stats_snapshot()["composite_declines"] == {
        apc_module.CompositeDecline.NO_STATE_CHECKPOINT: 1
    }


def test_composite_lookup_returns_a_plan_when_both_halves_exist():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    tokens = list(range(64))
    warm = _hybrid_cache()
    manager.release(
        harvest_blocks_from_batch_cache(
            manager, warm, tokens, allow_partial_layers=True
        )
    )
    _, checkpointed = apc_module.partition_cache_by_pageability(warm)
    assert manager.store_exact_cache(tokens[:32], [warm[i] for i in checkpointed])

    plan = _composite_plan(manager, tokens + list(range(900, 908)), _hybrid_cache())

    assert plan is not None
    assert plan["prefix_len"] == 32
    assert plan["warm_cache"] is not None
    assert manager.stats_snapshot()["composite_declines"] == {}
    manager.release(plan["matched_blocks"])


def test_store_final_keeps_composite_no_worse_than_exact_without_chunking():
    """A prefill that never chunks must still leave a usable checkpoint.

    Hidden-state speculative decoding runs the prompt in one pass, so the
    stride-based checkpoints never fire. Without a fallback, composite would
    store nothing where exact mode would have stored its snapshot.
    """
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    tokens = list(range(64))
    cache = _hybrid_cache()
    cp = _checkpointer(manager, tokens, budget_bytes=1 << 20)

    assert cp.stored == 0
    assert cp.store_final(48, cache) is True
    assert cp.stored == 1

    restored, prefix_len = manager.lookup_exact_cache(tokens)
    assert restored is not None
    assert prefix_len == 48


def test_store_final_respects_the_budget():
    manager = APCManager(num_blocks=64, block_size=16)
    manager.exact_cache_min_tokens = 1
    cp = _checkpointer(manager, list(range(64)), budget_bytes=8)

    assert cp.store_final(48, _hybrid_cache()) is False
    assert cp.stored == 0
    assert cp.decline == apc_module.CompositeDecline.BUDGET


def test_store_final_declines_a_cache_with_nothing_to_page():
    from mlx_vlm.models.cache import ArraysCache

    manager = APCManager(num_blocks=64, block_size=16)
    cp = _checkpointer(manager, list(range(64)))

    assert cp.store_final(48, [ArraysCache(1)]) is False
    assert cp.decline == apc_module.CompositeDecline.NOT_MIXED


def test_last_reuse_tokens_reports_the_current_request_not_a_running_total():
    manager = APCManager(num_blocks=64, block_size=16)
    tokens = list(range(64))
    keys = [mx.zeros((1, 1, len(tokens), 4))]
    values = [mx.zeros((1, 1, len(tokens), 4))]
    manager.release(manager.store_kv_blocks(tokens, keys, values))

    assert manager.stats_snapshot()["last_reuse_tokens"] == 0

    plan = _plan_for_mode(manager, tokens + list(range(900, 908)), "block")
    assert plan is not None
    hit = manager.stats_snapshot()["last_reuse_tokens"]
    assert hit == plan["prefix_len"] > 0
    manager.release(plan["matched_blocks"])

    assert _plan_for_mode(manager, list(range(500, 564)), "block") is None
    assert manager.stats_snapshot()["last_reuse_tokens"] == hit


def test_a_batched_cache_is_not_classified_as_composite(monkeypatch):
    from mlx_vlm.models.cache import BatchKVCache, BatchRotatingKVCache

    monkeypatch.setenv("APC_COMPOSITE", "1")

    class BatchLM:
        def make_cache(self):
            return [BatchRotatingKVCache(32, [0]), BatchKVCache([0])]

    assert model_apc_mode(BatchLM()) == "exact"


def test_kv_bits_leaves_a_sliding_window_cache_alone_instead_of_raising():
    """A rotating cache has no quantized form, and does not need one.

    Its window is bounded, so it is not what grows under a long prompt, and
    asking it to quantize raised NotImplementedError and took the request with
    it. The heterogeneous and turboquant paths already skip it.
    """
    from mlx_vlm.generate.common import maybe_quantize_kv_cache
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    shape = (1, 1, 48, 64)
    rotating = RotatingKVCache(max_size=64, keep=0)
    rotating.update_and_fetch(mx.zeros(shape), mx.zeros(shape))
    plain = KVCache()
    plain.update_and_fetch(mx.zeros(shape), mx.zeros(shape))
    tail = KVCache()
    tail.update_and_fetch(mx.zeros(shape), mx.zeros(shape))
    prompt_cache = [plain, rotating, tail]

    maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=0,
        kv_group_size=32,
        kv_bits=8,
    )

    assert isinstance(prompt_cache[1], RotatingKVCache)
    assert type(prompt_cache[0]).__name__ != "KVCache"
