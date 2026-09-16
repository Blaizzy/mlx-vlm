"""Tests for APC integration with quantized KV caches.

Uses real QuantizedKVCache and BatchQuantizedKVCache objects with small
dimensions. No mocking of cache behavior — these tests exercise the same
code paths production uses.

TDD: written BEFORE the implementation. Each test targets a specific
behavior required for issue #1174 (APC + KV-cache quantization).
"""

from __future__ import annotations

import os
from typing import List

import mlx.core as mx
import pytest

from mlx_vlm.apc import (
    APCManager,
    DiskBlockStore,
    _clone_cache_entry_for_apc,
    _clone_prompt_cache_for_apc,
    harvest_blocks_from_batch_cache,
    make_warm_batch_exact_cache_multi,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
    make_warm_kv_cache,
    snapshot_prompt_cache_row,
)
from mlx_vlm.apc_adapters import apc_exact_eligible
from mlx_vlm.generate.ar import _extend_cache, _make_cache
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    should_quantize_kv_layer,
)
from mlx_vlm.turboquant import BatchTurboQuantKVCache

# Small dimensions: fast tests, no GPU pressure
B, H, D = 1, 2, 32
GROUP_SIZE = 32
BITS = 8
BLOCK_SIZE = 16


def _rand_kv(batch=B, seq_len=32, heads=H, dim=D):
    """Generate random K/V tensors with realistic scale."""
    k = mx.random.normal((batch, heads, seq_len, dim))
    v = mx.random.normal((batch, heads, seq_len, dim))
    mx.eval(k, v)
    return k, v


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


# ---------------------------------------------------------------------------
# Test 1: QuantizedKVCache.dequantize_for_apc() roundtrip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 2: BatchQuantizedKVCache.dequantize_for_apc()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 3: harvest_blocks_from_batch_cache with quantized caches
# This is THE test that catches issue #1174's crash.
# ---------------------------------------------------------------------------


class TestHarvestFromQuantizedCache:
    def test_with_left_padding(self):
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


# ---------------------------------------------------------------------------
# Test 4: make_warm_kv_cache with quantization config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 5: Full store→lookup→restore roundtrip with quantized caches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 6: _cache_entry_supports_* with quantized types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 7: make_warm_batch_kv_cache with quantized config
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 8: model_apc_mode with quantized caches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 9: Guard removal regression test
# ---------------------------------------------------------------------------


class TestGuardRemoval:
    def test_apc_not_disabled_when_kv_bits_set(self):
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


# ---------------------------------------------------------------------------
# Test 10: Exact-mode store/restore with quantized entries
# ---------------------------------------------------------------------------


class TestExactModeQuantized:
    def test_hybrid_batch_kv_and_quantized_exact_store(self):
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

        warm, matched_tokens = manager.lookup_exact_cache(
            token_ids + [999], extra_hash=0
        )
        assert matched_tokens == len(token_ids)
        assert warm is not None
        assert len(warm) == 3


class TestNativePackedExactCheckpoints:
    def test_uniform_roundtrip_and_merge_stay_packed(self, tmp_path, monkeypatch):
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

        restored, matched = _disk_roundtrip(
            tmp_path, "native-uniform", token_ids, snapshot
        )
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
        self, tmp_path, monkeypatch, bits, key_bits, value_bits
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


# ---------------------------------------------------------------------------
# Test 11: Stored blocks are decoupled from quantized source
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 12: Empty-cache guard (regression for review finding #2)
# ---------------------------------------------------------------------------


class TestEmptyCacheGuard:
    def test_dequantize_for_apc_returns_none_when_empty(self):
        """dequantize_for_apc() returns (None, None) on an empty cache."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        dk, dv = cache.dequantize_for_apc()
        assert dk is None
        assert dv is None

    def test_turboquant_dequantize_for_apc_returns_none_when_empty(self):
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

    def test_harvest_handles_empty_quantized_cache(self):
        """harvest_blocks_from_batch_cache returns [] for empty quantized caches."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        empty_cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        token_ids = list(range(BLOCK_SIZE))
        blocks = harvest_blocks_from_batch_cache(
            manager, [empty_cache], batch_idx=0, full_token_ids=token_ids
        )
        assert blocks == []


# ---------------------------------------------------------------------------
# Test 13: Production path wires kv_quant_config (regression for review finding #1)
# ---------------------------------------------------------------------------


class TestProductionPathQuantConfig:
    def test_int_coercion_on_float_bits(self):
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


# ---------------------------------------------------------------------------
# Test 14: Multi-row batch with mixed warm/cold rows
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 15: Harvest from batch_idx > 0
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 16: Type homogeneity — warm and cold caches are same type for extend()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 17: Warm restore last-layer policy + staggered join (#1562 residual)
# ---------------------------------------------------------------------------
# Kept here (not a new file) so APC+kv tests stay co-located with #1174 suite.
# See also PR #1568 review: prefer existing test modules over new files.


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


class TestWarmRestoreLayerTypesMatchMakeCache:
    """APC warm path must match ``_make_cache`` last-layer policy (#1562)."""

    def test_make_warm_batch_kv_cache_multi_matches_make_cache_types(self):
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


class TestExactHybridColdStaggeredJoin:
    """#1579: exact multi warm must match live quant layout for staggered join.

    Hybrid models (Qwen3.5-class ArraysCache + full-attn) use exact APC.
    Live cold rows are BatchQuantized on full-attn layers; exact merge alone
    yields float BatchKVCache — _extend_cache then blows up. Requant on
    make_warm_batch_exact_cache_multi(kv_quant_config=...) fixes it.
    """

    def _hybrid_row_caches(self, seq_len: int, *, n_full_attn: int = 3):
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

    def _live_hybrid_batch(self, seq_len: int, n_full_attn: int = 3):
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

    def test_exact_warm_with_kv_config_extends_live_quant(self):
        seq_len = 16
        live = self._live_hybrid_batch(seq_len)
        row = self._hybrid_row_caches(seq_len)
        warm, _ = make_warm_batch_exact_cache_multi(
            [row], [seq_len], kv_quant_config=KV_CFG
        )
        assert _layer_type_names(live) == _layer_type_names(warm)
        extended = _extend_cache(live, warm)
        assert int(extended[1].offset.shape[0]) == 2
        assert isinstance(extended[1], BatchQuantizedKVCache)
        assert isinstance(extended[-1], BatchKVCache)


class TestTurboQuantWarmRestoreLayout:
    """APC warm must use BatchTurboQuant when live gen does (#1579 TQ follow-up).

    Live ``_make_cache`` selects TurboQuant via ``turboquant_enabled(bits, scheme)``.
    Warm helpers must not force uniform ``BatchQuantizedKVCache(bits=int(3.5))``
    — that packing fails for some head dims and cannot ``extend`` with TQ peers.
    """

    def _pure_attn_row(self, seq_len: int, num_layers: int = 4, head_dim: int = D):
        rows = []
        for _ in range(num_layers):
            c = KVCache()
            k = mx.random.normal((1, H, seq_len, head_dim))
            v = mx.random.normal((1, H, seq_len, head_dim))
            mx.eval(k, v)
            c.keys, c.values, c.offset = k, v, seq_len
            rows.append(c)
        return rows

    def _live_tq_batch(self, seq_len: int, num_layers: int = 4, head_dim: int = D):
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

    def test_block_warm_multi_turboquant_matches_make_cache_types(self):
        num_layers = 4
        seq_len = 16
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            # Store float blocks (APC always float); restore with TQ config.
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=TQ_CFG)
            live = self._live_tq_batch(seq_len, num_layers=num_layers)
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
