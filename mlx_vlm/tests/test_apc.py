from __future__ import annotations

import os
import subprocess
import sys

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
    from mlx_lm.models.cache import ArraysCache, KVCache

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
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

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
    from mlx_lm.models.cache import ArraysCache, KVCache

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
    from mlx_lm.models.cache import ChunkedKVCache, KVCache, RotatingKVCache

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
    from mlx_lm.models.cache import ArraysCache, KVCache

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
    from mlx_lm.models.cache import KVCache, RotatingKVCache

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
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

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
