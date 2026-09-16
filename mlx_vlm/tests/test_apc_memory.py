"""APC memory admission, eviction, and bounded disk persistence."""

import threading
import weakref
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_vlm import apc
from mlx_vlm.apc import APCManager, DiskBlockStore, _cache_nbytes, from_env
from mlx_vlm.apc_adapters import cache_memory_components, clone_cache_entry
from mlx_vlm.apc_coordinator import PrefillMemoryPlan
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchPoolingCache,
    BatchQuantizedKVCache,
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
)
from mlx_vlm.models.hy_v4.cache import HyV4KVCache
from mlx_vlm.models.minimax_m3_vl.language import (
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
)
from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache
from mlx_vlm.models.z1t.config import ModelConfig as Z1TConfig
from mlx_vlm.models.z1t.language import AFTConv, Z1TCache


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


def _hybrid(kv, composite=False):
    state = ArraysCache(1)
    state[0] = mx.ones((1, 256, 1024))  # 1 MiB fixed state.
    return [CacheList(state, kv)] if composite else [state, kv]


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


@pytest.mark.parametrize("composite", [False, True])
def test_short_hybrid_checkpoint_does_not_block_long_batch_reuse(
    manager_factory, monkeypatch, composite
):
    def make_cache(length=0):
        return _hybrid(BatchKVCache.merge([_kv(length)]), composite)

    manager = manager_factory(budget=4 << 20)
    monkeypatch.setattr(
        manager, "_memory_headroom", lambda: (8 << 20) - manager.resident_bytes()
    )
    coordinator = manager.coordinator(SimpleNamespace(make_cache=make_cache))
    for tokens in (list(range(18)), [42] * 6000):
        coordinator.prepare_prefill(len(tokens) + 1)
        assert coordinator.store_checkpoint(tokens, make_cache(len(tokens)))
    coordinator.prepare_prefill(6001)
    restored, count = manager.lookup_exact_cache(tokens + [9])
    assert restored is not None and count == 6000
    assert manager.stats.exact_stores == 2
    assert manager.stats.memory_skips == 0


def test_disk_fixed_state_reserve_counts_every_prefill_sequence(disk_reader):
    state = ArraysCache(1)
    state[0] = mx.ones((1, 256, 1024), dtype=mx.float32)
    tokens = list(range(18))
    manager = disk_reader(tokens, [state], budget=4 << 20)
    assert manager.lookup_exact_cache(tokens + [99])[1] == len(tokens)
    coordinator = _coordinator(manager, [state])
    coordinator.prepare_prefill([2000, 2000, 2000])
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 6 << 20
    coordinator.prepare_prefill(0)
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 0


def test_unknown_checkpoint_state_keeps_conservative_growth_estimate(
    manager_factory, monkeypatch
):
    class GrowingCache:
        state = mx.ones((1, 256, 1024))
        meta_state = ()

    manager = manager_factory(budget=4 << 20)
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


def test_disk_restore_capacity_does_not_inflate_growth(disk_reader):
    cache = _kv(16)
    cache.step = 256
    tokens = list(range(16))
    reader = disk_reader(tokens, [CacheList(cache)])
    restored, count = reader.lookup_exact_cache(tokens + [99] * 6000)
    assert count == 16 and restored[0][0].keys.shape[2] >= 6016
    coordinator = _coordinator(reader, restored)
    coordinator.prepare_prefill(6016)
    assert reader.stats_snapshot()["prefill_reserve_bytes"] == 2 * 6144 * 32


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_short_hybrid_checkpoint_extends_with_bounded_memory(
    manager_factory, disk_reader, monkeypatch, disk, composite
):
    entries = _hybrid(_kv(18), composite)
    tokens = list(range(18))
    if disk:
        reader = disk_reader(tokens, entries, budget=4 << 20)
    else:
        reader = manager_factory(budget=4 << 20)
        assert reader.store_exact_cache(tokens, entries)
    coordinator = _coordinator(reader, entries)
    monkeypatch.setattr(reader, "_memory_headroom", lambda: 8 << 20)
    coordinator.prepare_prefill(6001)

    restored, matched = reader.lookup_exact_cache(tokens + [99] * 5983)

    assert matched == 18
    leaves = restored[0].caches if composite else restored
    expected = entries[0].caches if composite else entries
    assert leaves[1].keys.shape[2] >= 6001
    assert mx.array_equal(leaves[0][0], expected[0][0]).item()
    assert _cache_nbytes(restored) < 2 << 20
    assert reader.stats.memory_skips == 0
    assert reader.stats.disk_hits == int(disk)


def test_disk_expansion_is_admitted_before_reserving_capacity(disk_reader, monkeypatch):
    tokens = list(range(16))
    reader = disk_reader(tokens, [_kv(16)], budget=4096)
    monkeypatch.setattr(reader, "_memory_headroom", lambda: 4096)
    monkeypatch.setattr(
        KVCache, "prefix_cache_reserve", lambda *a: pytest.fail("expanded")
    )
    assert reader.lookup_exact_cache(tokens + [99] * 6000) == (None, 0)
    assert reader.stats.memory_skips == 1


def test_rejected_disk_expansion_falls_back_to_memory(disk_reader, monkeypatch):
    tokens = list(range(1024))
    reader = disk_reader(tokens[:32], [_kv(32)])
    assert reader.store_exact_cache(tokens[:16], [_kv(16)])
    coordinator = _coordinator(reader, [KVCache()])
    coordinator.prepare_prefill(len(tokens))
    reserve = reader.stats_snapshot()["prefill_reserve_bytes"]
    disk_caches = []
    load = reader.disk.load_exact_cache

    def track_load(*args, **kwargs):
        loaded = load(*args, **kwargs)
        assert loaded is not None and len(loaded[0]) == 32
        disk_caches.append(weakref.ref(loaded[2][0]))
        return loaded

    # Disk expansion needs two buffers. Freeing the loaded 1 KiB lets
    # the memory hit fit with one clone.
    monkeypatch.setattr(reader.disk, "load_exact_cache", track_load)
    monkeypatch.setattr(
        reader,
        "_memory_headroom",
        lambda: reserve + 1000 - sum(_cache_nbytes(ref()) for ref in disk_caches),
    )
    restored, matched = reader.lookup_exact_cache(tokens)

    assert restored is not None and matched == 16
    assert mx.all(restored[0].state[0] == 1).item()
    assert restored[0].keys.shape[2] >= len(tokens)
    assert len(disk_caches) == 1 and disk_caches[0]() is None
    assert reader.stats_snapshot()["prefill_reserve_bytes"] == reserve
    assert reader.stats.exact_hits == 1
    assert reader.stats.disk_hits == 0
    assert reader.stats.memory_skips == 1


def test_prefill_budget_covers_unequal_length_batch_padding(manager_factory):
    manager = manager_factory(budget=4 << 20)
    cache = _kv(16)
    cache.step = 256
    assert manager.store_exact_cache(list(range(16)), [cache])
    coordinator = _coordinator(manager, [cache])
    lengths = [6000, 16, 16]
    coordinator.prepare_prefill(lengths)
    batch = BatchKVCache([max(lengths) - n for n in lengths])
    keys = mx.ones((len(lengths), 1, max(lengths), 4))
    batch.update_and_fetch(keys, keys + 1)
    assert manager.stats_snapshot()["prefill_reserve_bytes"] >= 2 * batch.nbytes


@pytest.mark.parametrize("make_cache", [BatchKVCache, BatchQuantizedKVCache])
def test_runtime_memory_is_learned_before_any_checkpoint(manager_factory, make_cache):
    manager = manager_factory(budget=4 << 20)
    cache = make_cache([0, 5984])
    coordinator = _coordinator(manager, [cache])
    coordinator.prepare_prefill([6000, 16])
    keys = mx.ones((2, 1, 16, 64))
    cache.update_and_fetch(keys, keys + 1)
    live_bytes = _cache_nbytes(cache)
    coordinator.observe_cache([cache], 16, batch_size=2)
    reserve = manager.stats_snapshot()["prefill_reserve_bytes"]

    future = make_cache([0, 5984])
    keys = mx.ones((2, 1, 6000, 64))
    future.update_and_fetch(keys, keys + 1)
    allocated = _cache_nbytes(future.keys) + _cache_nbytes(future.values)
    assert reserve + live_bytes >= 2 * allocated
    assert manager.stats.exact_stores == 0


@pytest.mark.parametrize("lengths", [(16, 6000), (6000, 16)])
def test_windowed_memory_budget_is_independent_of_checkpoint_order(
    manager_factory, lengths
):
    manager = manager_factory(budget=4 << 20)
    for length in lengths:
        cache = RotatingKVCache(max_size=512)
        keys = mx.ones((1, 1, length, 4))
        cache.update_and_fetch(keys, keys + 1)
        assert manager.store_exact_cache([length] * length, [cache])
    coordinator = _coordinator(manager, [cache])
    coordinator.prepare_prefill(6001, prefill_step_size=2048)
    # Window plus one chunk, rounded to 256 slots.
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 2 * 2560 * 32


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


def test_oversized_checkpoint_spills_without_clone(manager_factory, monkeypatch):
    manager = manager_factory(budget=256, disk=True)
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
    coordinator = _coordinator(manager, [KVCache()])
    assert manager.store_exact_cache([1] * 16, [_kv(16)])
    assert manager.store_exact_cache([2] * 16, [_kv(16)])
    # 3 KiB available including APC. A 40-token prefill reserves 2.5 KiB,
    # leaving room for only one retained 512-byte cache.
    monkeypatch.setattr(
        manager, "_memory_headroom", lambda: 3072 - manager.resident_bytes()
    )
    coordinator.prepare_prefill(40)
    assert manager.resident_bytes() == 512
    assert next(iter(manager._exact_cache.values())).token_ids == (2,) * 16
    assert manager.stats_snapshot()["prefill_reserve_bytes"] == 2560
    # An intermediate checkpoint may not immediately consume the freed space.
    monkeypatch.setattr(
        manager, "_memory_headroom", lambda: 3072 - manager.resident_bytes() - 1280
    )
    assert not manager.store_exact_cache([3] * 40, [_kv(40)])
    assert manager.resident_bytes() == 0


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


def test_prefill_waits_for_writes_before_evicting(manager_factory, monkeypatch):
    manager = manager_factory(disk=True)
    coordinator = _coordinator(manager, [KVCache()])
    manager.store_exact_cache([1] * 16, [_kv(16)])
    original_flush = manager.disk.flush
    flushed = []

    def flush():
        original_flush()
        assert manager.disk.pending_bytes == 0
        flushed.append(True)

    monkeypatch.setattr(manager.disk, "flush", flush)
    manager.memory_max_bytes = 0
    coordinator.prepare_prefill(100_000)
    assert flushed and manager.resident_bytes() == 0
    assert manager.disk.num_exact_indexed == 1


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


@pytest.mark.parametrize(
    "make_cache", [KVCache, HyV4KVCache, lambda: RingSlidingKVCache(16)]
)
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
    assert (
        cache.nbytes
        <= profile.footprint(6000, chunk_size, prefix_tokens=16)
        <= 2 * cache.nbytes
    )


def test_z1t_prefill_memory_is_fixed():
    layer = AFTConv(Z1TConfig(hidden_size=8, aft_heads=2, aft_ksize=4))
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


@pytest.mark.parametrize("factory", [KVCache, HyV4KVCache])
def test_allocation_forecast_releases_cache_and_resets_between_requests(factory):
    cache = factory()
    keys = mx.ones((1, 1, 6000, 4))
    cache.update_and_fetch(keys, keys)
    state = ArraysCache(1)
    state[0] = mx.ones((1, 4))
    plan = PrefillMemoryPlan()
    plan.observe_cache([cache, state], 6000)
    reference = weakref.ref(cache)
    del cache
    assert reference() is None

    small = factory()
    keys = keys[..., :16, :]
    small.update_and_fetch(keys, keys)
    expected = 2 * (small.nbytes + state.nbytes)
    assert plan.prepare([16], chunk_size=37) == expected
    assert plan.observe_cache([factory(), state], 0) == expected - state.nbytes


@pytest.mark.parametrize("factory", [KVCache, HyV4KVCache])
def test_allocation_forecast_uses_live_capacity_after_trim(factory):
    cache = factory()
    keys = mx.ones((1, 1, 512, 4))
    cache.update_and_fetch(keys, keys)
    cache.trim(256)
    plan = PrefillMemoryPlan()
    plan.prepare([768], chunk_size=512)
    previous_bytes = cache.nbytes
    reserve = plan.observe_cache([cache], 256)
    cache.update_and_fetch(keys, keys)
    assert reserve == 2 * cache.nbytes - previous_bytes
