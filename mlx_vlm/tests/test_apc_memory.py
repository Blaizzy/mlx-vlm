"""APC memory admission, eviction, and bounded disk persistence."""

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import apc
from mlx_vlm.apc import APCManager, DiskBlockStore, _cache_nbytes, from_env
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache


def _kv(length, value=1):
    cache = KVCache()
    cache.keys = mx.full((1, 1, length, 4), value, dtype=mx.float32)
    cache.values = mx.full((1, 1, length, 4), value + 1, dtype=mx.float32)
    cache.offset = length
    mx.eval(cache.state)
    return cache


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


def test_exact_resident_bytes_and_byte_lru(manager_factory):
    manager = manager_factory(budget=1536)
    a, b, c = [1] * 16, [2] * 16, [3] * 32
    assert manager.store_exact_cache(a, [_kv(16, 1)])
    assert manager.store_exact_cache(b, [_kv(16, 2)])
    assert manager.resident_bytes() == 1024
    assert manager.lookup_exact_cache(a + [9])[1] == 16  # A becomes most recent.
    assert manager.store_exact_cache(c, [_kv(32, 3)])
    assert [entry.token_ids for entry in manager._exact_cache.values()] == [
        tuple(a),
        tuple(c),
    ]
    stats = manager.stats_snapshot()
    assert stats["resident_bytes"] == stats["exact_resident_bytes"] == 1536
    assert stats["memory_evictions"] == 1
    assert manager.lookup_exact_cache(b + [9]) == (None, 0)
    manager.clear()
    assert manager.resident_bytes() == 0


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


@pytest.mark.parametrize("budget", [0, 256])
def test_oversized_checkpoint_spills_without_clone(
    manager_factory, monkeypatch, budget
):
    manager = manager_factory(budget=budget, disk=True)
    tokens = list(range(32))
    source = _kv(32)

    def no_clone(*args, **kwargs):
        raise AssertionError("Spilling must not allocate another complete checkpoint")

    monkeypatch.setattr(apc, "_clone_prompt_cache_for_apc", no_clone)
    coordinator = manager.coordinator(
        SimpleNamespace(make_cache=lambda: [ArraysCache(1)])
    )
    assert coordinator.store_checkpoint(tokens, [source])
    assert manager.resident_bytes() == manager.disk.pending_bytes == 0
    assert manager.disk.num_exact_indexed == 1  # Finished before producer returns.
    assert manager.stats_snapshot()["disk_writes"] == 1
    source.keys = mx.zeros_like(source.keys)
    restored, count = manager.lookup_exact_cache(tokens + [99])
    assert count == 32
    assert mx.all(restored[0].state[0] == 1).item()
    assert manager.resident_bytes() == 0  # Disk promotion obeys the byte cap too.


def test_oversized_checkpoint_without_disk_does_not_clone(manager_factory, monkeypatch):
    manager = manager_factory(budget=1)
    monkeypatch.setattr(
        apc, "_clone_prompt_cache_for_apc", lambda *a, **kw: pytest.fail("cloned")
    )
    assert not manager.store_exact_cache(list(range(32)), [_kv(32)])
    assert manager.resident_bytes() == 0
    assert manager.stats_snapshot()["memory_skips"] == 1


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


def test_prefill_evicts_oldest_before_new_allocation(manager_factory, monkeypatch):
    manager = manager_factory()
    assert manager.store_exact_cache([1] * 16, [_kv(16)])
    assert manager.store_exact_cache([2] * 16, [_kv(16)])
    # 3 KiB available including APC. A 40-token prefill reserves 2.5 KiB,
    # leaving room for only one retained 512-byte cache.
    monkeypatch.setattr(
        manager, "_memory_headroom", lambda: 3072 - manager.resident_bytes()
    )
    manager.prepare_prefill(40)
    assert manager.resident_bytes() == 512
    assert next(iter(manager._exact_cache.values())).token_ids == (2,) * 16
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 2560
    # An intermediate checkpoint may not immediately consume the freed space.
    monkeypatch.setattr(
        manager, "_memory_headroom", lambda: 3072 - manager.resident_bytes() - 1280
    )
    assert not manager.store_exact_cache([3] * 40, [_kv(40)])
    assert manager.resident_bytes() == 0


@pytest.mark.parametrize("sequence_count", [1, 3])
def test_short_hybrid_prompt_does_not_inflate_long_prefill_reserve(
    manager_factory, monkeypatch, sequence_count
):
    manager = manager_factory(budget=1 << 20)
    recurrent = ArraysCache(1)
    recurrent[0] = mx.zeros((1, 128, 128))
    # Only 16 tokens occupy a capacity-rounded KV buffer. Nested caches model
    # GLM's latent attention and indexer layout.
    kv = _kv(256)
    kv.offset = 16
    prompt_cache = [recurrent, CacheList(kv)]
    assert manager.store_exact_cache(list(range(16)), prompt_cache)
    fixed = recurrent.nbytes + (256 - 16) * 32
    expected = 2 * (fixed * sequence_count + 30_000 * 32)
    monkeypatch.setattr(manager, "_memory_headroom", lambda: expected + (1 << 20))
    manager.prepare_prefill(30_000, sequence_count=sequence_count)
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == expected
    assert manager.stats_snapshot()["memory_evictions"] == 0
    # Observing a long checkpoint preserves a per-token estimate of 32 bytes,
    # while the fixed recurrent state remains counted once per sequence.
    manager._observe_cache_size(
        recurrent.nbytes + 30_000 * 32, 30_000, [recurrent, CacheList(_kv(30_000))]
    )
    assert manager._bytes_per_token == 32


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


@pytest.mark.parametrize("disk", [False, True])
def test_hybrid_short_prefix_restore_counts_fixed_state_once(
    manager_factory, monkeypatch, disk
):
    manager = manager_factory(budget=4 << 20, disk=disk)
    recurrent = ArraysCache(1)
    recurrent[0] = mx.ones((1, 512, 512))
    tokens = list(range(16))
    assert manager.store_exact_cache(tokens, [recurrent, CacheList(_kv(16))])
    if disk:
        manager.disk.flush()
        manager.clear()
        # A fresh manager must read the persisted growth profile before loading
        # tensors; no in-memory observations survive a server restart.
        manager = manager_factory(budget=4 << 20, disk=True)
    monkeypatch.setattr(manager, "_memory_headroom", lambda: 6 << 20)
    manager.prepare_prefill(4096)
    restored, length = manager.lookup_exact_cache(tokens + [99] * (4096 - 16))
    assert length == 16
    assert mx.array_equal(restored[0][0], recurrent[0]).item()
    assert restored[1].caches[0].keys.shape[2] >= 4096
    assert manager.stats_snapshot()["memory_skips"] == 0


def test_prefill_waits_for_writes_before_evicting(manager_factory, monkeypatch):
    manager = manager_factory(disk=True)
    manager.store_exact_cache([1] * 16, [_kv(16)])
    original_flush = manager.disk.flush
    flushed = []

    def flush():
        original_flush()
        assert manager.disk.pending_bytes == 0
        flushed.append(True)

    monkeypatch.setattr(manager.disk, "flush", flush)
    manager.memory_max_bytes = 0
    manager.prepare_prefill(100_000)
    assert flushed and manager.resident_bytes() == 0
    assert manager.disk.num_exact_indexed == 1


def test_byte_eviction_preserves_leased_blocks(manager_factory):
    manager = manager_factory(budget=1024)
    source = _kv(32)
    leased = manager.store_kv_blocks(list(range(32)), [source.keys], [source.values])
    assert len(leased) == 2
    manager.release(leased[:1])
    manager.memory_max_bytes = 0
    manager.prepare_prefill(100_000)
    assert manager.resident_bytes() == 512
    assert leased[1].ref_cnt == 1 and leased[1].keys is not None
    manager.release(leased[1:])
    manager.prepare_prefill(100_000)
    assert manager.resident_bytes() == 0


def test_dense_budget_keeps_disk_prefix_complete(manager_factory):
    manager = manager_factory(budget=512, disk=True)
    source = _kv(64)
    tokens = list(range(64))
    blocks = manager.store_kv_blocks(tokens, [source.keys], [source.values])
    manager.release(blocks)
    assert manager.resident_bytes() <= 512
    manager.disk.flush()
    assert manager.disk.num_blocks_indexed == 4
    restored, count = manager.lookup_prefix_disk_cache(
        tokens + [99], allow_memory_overlap=True
    )
    assert count == 64
    assert restored[0].offset == 64


def test_sequential_long_prefixes_remain_bounded_and_replay_from_disk(
    manager_factory, monkeypatch
):
    manager = manager_factory(budget=2 << 20, disk=True)
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
        manager.prepare_prefill(length)
        cache = _kv(length, i)
        live_bytes[0] = cache.nbytes
        assert manager.store_exact_cache([i] * length, [cache])
        assert manager.resident_bytes() <= manager.memory_max_bytes
        assert manager.disk.pending_bytes <= manager.disk.queue_max_bytes
        del cache
        live_bytes[0] = 0
        mx.clear_cache()

    manager.prepare_prefill(100_001)
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


def test_failed_disk_write_releases_pending_bytes(tmp_path, monkeypatch):
    disk = DiskBlockStore(tmp_path)
    monkeypatch.setattr(
        disk,
        "_write_exact_cache_snapshot",
        lambda *a: (_ for _ in ()).throw(OSError("full")),
    )
    try:
        disk.save_exact_cache(1, [1] * 16, 0, [_kv(16)])
        disk.flush()
        assert disk.pending_bytes == 0
        assert not disk._in_flight
        assert disk.num_exact_indexed == 0
    finally:
        disk.close()


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
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("APC_ENABLED", "1")
    if opt_out == "environment":
        monkeypatch.setenv("APC_DISK_ENABLED", "0")
        overrides = None
    else:
        overrides = {"disk_path": ""}
    manager = from_env(overrides=overrides)
    assert manager.disk is None
    assert not (tmp_path / "mlx-vlm").exists()


def test_automatic_budget_and_overrides(monkeypatch):
    monkeypatch.setattr(apc, "_metal_working_set_bytes", lambda: 40 << 30)
    monkeypatch.delenv("APC_MEMORY_MAX_GB", raising=False)
    monkeypatch.delenv("APC_MEMORY_RESERVE_GB", raising=False)
    manager = APCManager(num_blocks=1)
    assert manager.memory_max_bytes == manager.memory_reserve_bytes == 4 << 30
    monkeypatch.setenv("APC_MEMORY_MAX_GB", "0")
    monkeypatch.setenv("APC_MEMORY_RESERVE_GB", "2")
    manager = APCManager(num_blocks=1)
    assert manager.memory_max_bytes == 0
    assert manager.memory_reserve_bytes == 2 << 30
