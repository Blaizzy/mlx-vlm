"""APC memory budgets, disk persistence, and block storage."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import apc
from mlx_vlm.apc import APCManager, DiskBlockStore, _cache_nbytes, from_env
from mlx_vlm.apc_storage import KVBlockHandle
from mlx_vlm.models.cache import ArraysCache, KVCache, QuantizedKVCache, RotatingKVCache

# Memory budgets and disk eviction


def _kv(length, value=1):
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
def manager_factory(monkeypatch, tmp_path):
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
def disk_reader(manager_factory):
    """Seed a disk checkpoint and return a manager with no resident entries."""

    def make(tokens, caches, *, budget=1 << 20):
        writer = manager_factory(budget=budget, disk=True)
        assert writer.store_exact_cache(tokens, caches)
        writer.disk.flush()
        return manager_factory(budget=budget, disk=True)

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
def test_kv_growth_ignores_unused_capacity(manager_factory, make_cache):
    capacity = 256
    cache = make_cache()
    cache.step = 1
    tensor = mx.ones((1, 1, capacity, 64))
    cache.update_and_fetch(tensor, tensor + 1)
    cache.trim(capacity - 16)
    cache.step = 256
    allocated_bytes = _cache_nbytes(cache)
    per_token = allocated_bytes // capacity
    manager = manager_factory(budget=1 << 20)
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
    reader = disk_reader(tokens, [_kv(16)], budget=4096)
    monkeypatch.setattr(reader, "_memory_headroom", lambda: 4096)
    monkeypatch.setattr(
        KVCache, "prefix_cache_reserve", lambda *a: pytest.fail("expanded")
    )
    assert reader.lookup_exact_cache(tokens + [99] * 6000) == (None, 0)
    assert reader.stats.memory_skips == 1


def test_padded_kv_admission_counts_allocated_buffers(manager_factory, monkeypatch):
    cache = _kv(256)
    cache.offset = 16
    manager = manager_factory(budget=1024)
    monkeypatch.setattr(
        apc, "_clone_prompt_cache_for_apc", lambda *a, **kw: pytest.fail("cloned")
    )
    # Admission includes the unused portion of the 8 KiB buffer.
    assert not manager.store_exact_cache(list(range(16)), [cache])
    assert manager.resident_bytes() == 0
    assert manager.stats.memory_skips == 1


def test_batch_checkpoint_skips_extraction_without_headroom(
    manager_factory, monkeypatch
):
    manager = manager_factory(disk=True)
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
    manager_factory, monkeypatch, exact
):
    manager = manager_factory(budget=0, disk=True)
    source = _kv(32)
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
    manager_factory, monkeypatch
):
    manager = manager_factory()
    tokens = list(range(16))
    assert manager.store_exact_cache(tokens, [_kv(16)])
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 4096)
    monkeypatch.setattr(
        apc, "_clone_prompt_cache_for_apc", lambda *a, **kw: pytest.fail("cloned")
    )
    # The stored 512 bytes fit, but allocating capacity for 1,000 tokens does not.
    assert manager.lookup_exact_cache(tokens + [99] * 984) == (None, 0)
    assert manager.stats_snapshot()["memory_skips"] == 1


def test_byte_eviction_preserves_leased_blocks(manager_factory):
    manager = manager_factory(budget=1024)
    source = _kv(32)
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
    manager_factory, monkeypatch
):
    manager = manager_factory(budget=2 << 20, disk=True)
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
        cache = _kv(length, i)
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
    caches = [_kv(16), _kv(16, 2)]
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


def test_failed_direct_spill_does_not_report_a_store(manager_factory, monkeypatch):
    manager = manager_factory(budget=0, disk=True)

    def fail_write(*args):
        raise OSError("full")

    monkeypatch.setattr(manager.disk, "_write_exact_cache_snapshot", fail_write)
    assert not manager.store_exact_cache(list(range(32)), [_kv(32)])
    stats = manager.stats_snapshot()
    assert stats["exact_stores"] == stats["resident_bytes"] == 0
    assert stats["disk_write_failures"] == 1
    assert not manager.disk._in_flight


@pytest.mark.parametrize("synchronous", [True, False])
def test_completed_oversized_disk_write_obeys_cap(tmp_path, synchronous):
    disk = DiskBlockStore(tmp_path, max_bytes=512)
    try:
        disk.save_exact_cache(1, [1] * 32, 0, [_kv(32)], synchronous=synchronous)
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


# Storage handles


@pytest.fixture
def _seed_block_storage():
    mx.random.seed(0)


@pytest.mark.usefixtures("_seed_block_storage")
def test_kv_block_handle_empty():
    handle = KVBlockHandle()
    assert handle.resident_bytes() == 0
