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
    _cache_entry_supports_block_apc,
    _cache_entry_supports_exact_apc,
    _clone_cache_entry_for_apc,
    _clone_prompt_cache_for_apc,
    harvest_blocks_from_batch_cache,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
    make_warm_kv_cache,
    model_apc_mode,
)
from mlx_vlm.generate.ar import _extend_cache, _make_cache
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    KVCache,
    QuantizedKVCache,
    should_quantize_kv_layer,
)

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


def _max_abs_error(a: mx.array, b: mx.array) -> float:
    return mx.max(mx.abs(a - b)).item()


# ---------------------------------------------------------------------------
# Test 1: QuantizedKVCache.dequantize_for_apc() roundtrip
# ---------------------------------------------------------------------------


class TestQuantizedCacheDequantize:
    def test_roundtrip_within_tolerance(self):
        """quantize → dequantize produces values within 8-bit tolerance."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(seq_len=64)
        cache.update_and_fetch(k, v)

        dk, dv = cache.dequantize_for_apc()
        mx.eval(dk, dv)

        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert dk.dtype in (mx.float16, mx.float32, mx.bfloat16)
        # 8-bit quantization: max error should be small
        assert _max_abs_error(dk, k) < 0.1
        assert _max_abs_error(dv, v) < 0.1

    def test_respects_offset(self):
        """dequantize_for_apc() returns only tokens up to offset, not full buffer."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(seq_len=10)
        cache.update_and_fetch(k, v)

        dk, dv = cache.dequantize_for_apc()
        mx.eval(dk, dv)

        # Should be 10 tokens, not the full 256-step allocation
        assert dk.shape[2] == 10
        assert dv.shape[2] == 10

    def test_incremental_fill(self):
        """Works correctly after multiple update_and_fetch calls."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        k1, v1 = _rand_kv(seq_len=20)
        k2, v2 = _rand_kv(seq_len=12)
        cache.update_and_fetch(k1, v1)
        cache.update_and_fetch(k2, v2)

        dk, dv = cache.dequantize_for_apc()
        mx.eval(dk, dv)

        assert dk.shape[2] == 32  # 20 + 12
        assert dv.shape[2] == 32
        # First 20 tokens should approximate k1
        assert _max_abs_error(dk[:, :, :20, :], k1) < 0.1


# ---------------------------------------------------------------------------
# Test 2: BatchQuantizedKVCache.dequantize_for_apc()
# ---------------------------------------------------------------------------


class TestBatchQuantizedCacheDequantize:
    def test_roundtrip(self):
        """Batch quantized cache dequantize roundtrip."""
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=48)
        cache.update_and_fetch(k, v)

        dk, dv = cache.dequantize_for_apc()
        mx.eval(dk, dv)

        assert dk.shape == (1, H, 48, D)
        assert dv.shape == (1, H, 48, D)
        assert _max_abs_error(dk, k) < 0.1
        assert _max_abs_error(dv, v) < 0.1

    def test_respects_idx(self):
        """Returns only up to _idx, not full allocation."""
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=7)
        cache.update_and_fetch(k, v)

        dk, dv = cache.dequantize_for_apc()
        mx.eval(dk, dv)

        assert dk.shape[2] == 7
        assert dv.shape[2] == 7


# ---------------------------------------------------------------------------
# Test 3: harvest_blocks_from_batch_cache with quantized caches
# This is THE test that catches issue #1174's crash.
# ---------------------------------------------------------------------------


class TestHarvestFromQuantizedCache:
    def test_does_not_crash(self):
        """harvest_blocks_from_batch_cache works with BatchQuantizedKVCache.

        Previously crashed with: TypeError on keys[batch_idx:...] because
        .keys is a tuple (packed, scales, biases), not an array.
        """
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE  # 32 tokens = 2 full blocks

        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        num_layers = 4
        batch_caches = [
            BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
            for _ in range(num_layers)
        ]
        for c in batch_caches:
            ki, vi = _rand_kv(batch=1, seq_len=seq_len)
            c.update_and_fetch(ki, vi)
            mx.eval(c.keys)

        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, batch_caches, batch_idx=0, full_token_ids=token_ids
        )

        assert len(blocks) == 2  # 32 tokens / 16 block_size
        for block in blocks:
            assert len(block.keys) == num_layers
            assert len(block.values) == num_layers
            for k_layer in block.keys:
                assert isinstance(k_layer, mx.array)
                assert k_layer.shape == (1, H, BLOCK_SIZE, D)
            for v_layer in block.values:
                assert isinstance(v_layer, mx.array)
                assert v_layer.shape == (1, H, BLOCK_SIZE, D)
        manager.release(blocks)

    def test_harvested_values_approximate_originals(self):
        """Harvested float blocks approximate the original input values."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = BLOCK_SIZE  # 1 full block

        original_k, original_v = _rand_kv(batch=1, seq_len=seq_len)

        batch_caches = []
        for _ in range(2):
            c = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
            c.update_and_fetch(original_k, original_v)
            mx.eval(c.keys)
            batch_caches.append(c)

        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, batch_caches, batch_idx=0, full_token_ids=token_ids
        )

        assert len(blocks) == 1
        # Harvested block should approximate the original
        assert _max_abs_error(blocks[0].keys[0], original_k) < 0.1
        assert _max_abs_error(blocks[0].values[0], original_v) < 0.1
        manager.release(blocks)

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


class TestRestoreIntoQuantizedCache:
    def test_creates_quantized_cache(self):
        """make_warm_kv_cache with kv_quant_config returns QuantizedKVCache."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE

        # Store blocks from plain float K/V
        layer_keys, layer_values = _rand_kv(seq_len=seq_len)
        # Duplicate for 3 layers
        lk = [layer_keys, mx.array(layer_keys), mx.array(layer_keys)]
        lv = [layer_values, mx.array(layer_values), mx.array(layer_values)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, matched_tokens = manager.lookup_prefix(token_ids)
        assert matched_tokens == seq_len

        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm = make_warm_kv_cache(matched, kv_quant_config=quant_config)

        assert len(warm) == 3
        # n=3 > 2: last layer stays float (matches stream / batch last-layer policy)
        for i, c in enumerate(warm):
            if should_quantize_kv_layer(i, 3):
                assert isinstance(c, QuantizedKVCache)
                assert c.offset == seq_len
                assert c.bits == BITS
                assert c.group_size == GROUP_SIZE
            else:
                assert isinstance(c, KVCache)
                assert c.offset == seq_len
        manager.release(matched)

    def test_restored_values_approximate_stored(self):
        """Restored quantized cache dequantizes back to stored values."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = BLOCK_SIZE

        layer_keys, layer_values = _rand_kv(seq_len=seq_len)
        lk = [layer_keys]
        lv = [layer_values]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm = make_warm_kv_cache(matched, kv_quant_config=quant_config)

        dk, dv = warm[0].dequantize_for_apc()
        mx.eval(dk, dv)

        # One quantization cycle on restore
        assert _max_abs_error(dk, layer_keys) < 0.1
        assert _max_abs_error(dv, layer_values) < 0.1
        manager.release(matched)


# ---------------------------------------------------------------------------
# Test 5: Full store→lookup→restore roundtrip with quantized caches
# ---------------------------------------------------------------------------


class TestFullRoundtripQuantized:
    def test_store_quantized_lookup_restore_quantized(self):
        """End-to-end: harvest from quantized → store → lookup → restore into quantized.

        Two quantization cycles (harvest dequant + restore requant).
        """
        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 3 * BLOCK_SIZE
        num_layers = 2

        # Simulate prefill: fill quantized batch caches
        originals_k = []
        originals_v = []
        batch_caches = []
        for _ in range(num_layers):
            c = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
            k, v = _rand_kv(batch=1, seq_len=seq_len)
            c.update_and_fetch(k, v)
            mx.eval(c.keys)
            originals_k.append(k)
            originals_v.append(v)
            batch_caches.append(c)

        # Harvest
        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, batch_caches, batch_idx=0, full_token_ids=token_ids
        )
        assert len(blocks) == 3
        manager.release(blocks)

        # Lookup
        matched, matched_tokens = manager.lookup_prefix(token_ids)
        assert matched_tokens == seq_len

        # Restore into quantized
        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm = make_warm_kv_cache(matched, kv_quant_config=quant_config)

        assert len(warm) == num_layers
        for i, c in enumerate(warm):
            assert isinstance(c, QuantizedKVCache)
            assert c.offset == seq_len
            dk, dv = c.dequantize_for_apc()
            mx.eval(dk, dv)
            # Two quant cycles: slightly more error but still bounded
            assert _max_abs_error(dk, originals_k[i]) < 0.2
            assert _max_abs_error(dv, originals_v[i]) < 0.2
        manager.release(matched)

    def test_restored_cache_accepts_new_tokens(self):
        """Restored quantized cache can accept additional tokens via update_and_fetch."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = BLOCK_SIZE

        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, [cache], batch_idx=0, full_token_ids=token_ids
        )
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm = make_warm_kv_cache(matched, kv_quant_config=quant_config)

        # Feed more tokens into the restored cache
        new_k, new_v = _rand_kv(batch=1, seq_len=5)
        warm[0].update_and_fetch(new_k, new_v)
        assert warm[0].offset == seq_len + 5
        manager.release(matched)


# ---------------------------------------------------------------------------
# Test 6: _cache_entry_supports_* with quantized types
# ---------------------------------------------------------------------------


class TestCacheEntrySupport:
    def test_quantized_supports_block_apc(self):
        """QuantizedKVCache should be recognized as block-APC compatible."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        assert _cache_entry_supports_block_apc(cache) is True

    def test_quantized_supports_exact_apc(self):
        """QuantizedKVCache should be recognized as exact-APC compatible."""
        cache = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
        assert _cache_entry_supports_exact_apc(cache) is True

    def test_plain_kv_still_works(self):
        """Existing KVCache support is not broken."""
        cache = KVCache()
        assert _cache_entry_supports_block_apc(cache) is True
        assert _cache_entry_supports_exact_apc(cache) is True


# ---------------------------------------------------------------------------
# Test 7: make_warm_batch_kv_cache with quantized config
# ---------------------------------------------------------------------------


class TestMakeWarmBatchQuantized:
    def test_creates_batch_quantized_cache(self):
        """make_warm_batch_kv_cache with kv_quant_config returns BatchQuantizedKVCache."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE
        num_layers = 2

        lk = []
        lv = []
        for _ in range(num_layers):
            k, v = _rand_kv(seq_len=seq_len)
            lk.append(k)
            lv.append(v)
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm = make_warm_batch_kv_cache(matched, kv_quant_config=quant_config)

        assert len(warm) == num_layers
        for c in warm:
            assert isinstance(c, BatchQuantizedKVCache)
        manager.release(matched)


# ---------------------------------------------------------------------------
# Test 8: model_apc_mode with quantized caches
# ---------------------------------------------------------------------------


class TestModelApcModeQuantized:
    def test_returns_block_for_quantized_model(self):
        """model_apc_mode returns 'block' when make_cache produces QuantizedKVCache."""

        class FakeModel:
            def make_cache(self):
                return [
                    QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
                    QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
                ]

        mode = model_apc_mode(FakeModel())
        assert mode == "block"

    def test_returns_exact_for_mixed_quantized_and_rotating(self):
        """model_apc_mode returns 'exact' for mixed quantized + non-block-eligible."""
        from mlx_vlm.models.cache import RotatingKVCache

        class FakeHybridModel:
            def make_cache(self):
                return [
                    QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
                    RotatingKVCache(max_size=128, keep=0),
                    QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS),
                ]

        mode = model_apc_mode(FakeHybridModel())
        assert mode == "exact"


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
    def test_store_and_lookup_quantized_in_exact_cache(self):
        """store_exact_cache / lookup_exact_cache works with QuantizedKVCache."""
        manager = APCManager(num_blocks=4, block_size=BLOCK_SIZE)
        token_ids = list(range(2 * BLOCK_SIZE))

        # Build a prompt_cache with QuantizedKVCache entries
        caches = []
        for _ in range(2):
            c = QuantizedKVCache(group_size=GROUP_SIZE, bits=BITS)
            k, v = _rand_kv(seq_len=len(token_ids))
            c.update_and_fetch(k, v)
            caches.append(c)
        mx.eval([c.keys for c in caches])

        stored = manager.store_exact_cache(token_ids, caches, extra_hash=0)
        assert stored is True

        warm, matched_tokens = manager.lookup_exact_cache(
            token_ids + [999], extra_hash=0
        )
        assert matched_tokens == len(token_ids)
        assert warm is not None
        assert len(warm) == 2

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
        arrays.cache = [
            mx.zeros((1, seq_len, D)),
            mx.zeros((1, seq_len, D)),
        ]
        arrays.left_padding = mx.array([0])

        batch_kv = BatchKVCache([0])
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        batch_kv.update_and_fetch(k, v)

        batch_q = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        kq, vq = _rand_kv(batch=1, seq_len=seq_len)
        batch_q.update_and_fetch(kq, vq)
        mx.eval(batch_kv.keys, batch_q.keys)

        prompt_cache = [arrays, batch_kv, batch_q]
        assert all(_cache_entry_supports_exact_apc(c) for c in prompt_cache)

        # Clone path: BatchKVCache collapses to KVCache; quant dequants to KVCache.
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
        assert isinstance(cloned[2], KVCache)

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


# ---------------------------------------------------------------------------
# Test 11: Stored blocks are decoupled from quantized source
# ---------------------------------------------------------------------------


class TestBlockDecoupling:
    def test_harvested_blocks_independent_of_source_cache(self):
        """After harvest, mutating the source cache doesn't affect stored blocks."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = BLOCK_SIZE

        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        k, v = _rand_kv(batch=1, seq_len=seq_len)
        cache.update_and_fetch(k, v)
        mx.eval(cache.keys)

        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, [cache], batch_idx=0, full_token_ids=token_ids
        )
        assert len(blocks) == 1

        # Snapshot block value before mutating source
        block_k_snapshot = mx.array(blocks[0].keys[0])
        mx.eval(block_k_snapshot)

        # Mutate source cache by adding more tokens
        k2, v2 = _rand_kv(batch=1, seq_len=10)
        cache.update_and_fetch(k2, v2)
        mx.eval(cache.keys)

        # Block should be unchanged
        assert _max_abs_error(blocks[0].keys[0], block_k_snapshot) == 0.0
        manager.release(blocks)


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

    def test_batch_dequantize_for_apc_returns_none_when_empty(self):
        """BatchQuantizedKVCache.dequantize_for_apc() returns (None, None) when empty."""
        cache = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
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
    def test_make_warm_batch_kv_cache_multi_with_quant_config(self):
        """make_warm_batch_kv_cache_multi creates BatchQuantizedKVCache when config is passed."""
        from mlx_vlm.apc import make_warm_batch_kv_cache_multi

        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE

        # Store blocks
        lk = [_rand_kv(seq_len=seq_len)[0] for _ in range(2)]
        lv = [_rand_kv(seq_len=seq_len)[1] for _ in range(2)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        picks = [{"matched_blocks": matched, "prefix_len": seq_len}]

        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm, max_prefix = make_warm_batch_kv_cache_multi(
            picks, num_layers=2, kv_quant_config=quant_config
        )

        assert max_prefix == seq_len
        assert len(warm) == 2
        for c in warm:
            assert isinstance(c, BatchQuantizedKVCache)
            assert c._idx == seq_len
            assert c.bits == BITS
        manager.release(matched)

    def test_make_warm_batch_kv_cache_multi_without_quant_stays_plain(self):
        """Without kv_quant_config, make_warm_batch_kv_cache_multi returns BatchKVCache."""
        from mlx_vlm.apc import make_warm_batch_kv_cache_multi
        from mlx_vlm.models.cache import BatchKVCache

        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE

        lk = [_rand_kv(seq_len=seq_len)[0] for _ in range(2)]
        lv = [_rand_kv(seq_len=seq_len)[1] for _ in range(2)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        picks = [{"matched_blocks": matched, "prefix_len": seq_len}]

        warm, max_prefix = make_warm_batch_kv_cache_multi(
            picks, num_layers=2, kv_quant_config=None
        )

        assert len(warm) == 2
        for c in warm:
            assert isinstance(c, BatchKVCache)
        manager.release(matched)

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


class TestMultiRowMixedWarmCold:
    def test_mixed_picks_produces_correct_types_and_shapes(self):
        """make_warm_batch_kv_cache_multi with some None picks (cold) works.

        This is the exact production scenario: some rows hit APC (warm),
        others miss (cold, get zero-padded). All rows must produce the
        same cache type so extend() works.
        """
        from mlx_vlm.apc import make_warm_batch_kv_cache_multi

        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE
        num_layers = 2

        lk = [_rand_kv(seq_len=seq_len)[0] for _ in range(num_layers)]
        lv = [_rand_kv(seq_len=seq_len)[1] for _ in range(num_layers)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)

        # Row 0 = warm (APC hit), Row 1 = cold (miss)
        picks = [
            {"matched_blocks": matched, "prefix_len": seq_len},
            None,
        ]

        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm, max_prefix = make_warm_batch_kv_cache_multi(
            picks, num_layers=num_layers, kv_quant_config=quant_config
        )

        assert max_prefix == seq_len
        assert len(warm) == num_layers
        for c in warm:
            assert isinstance(c, BatchQuantizedKVCache)
            # _idx covers the full max_prefix (warm row content + cold row zeros)
            assert c._idx == max_prefix
            # left_padding: row 0 has 0 (full hit), row 1 has max_prefix (all cold)
            lp = c.left_padding.tolist()
            assert lp[0] == 0
            assert lp[1] == max_prefix
        manager.release(matched)

    def test_mixed_picks_without_quant_produces_batch_kv_cache(self):
        """Without quant config, mixed warm/cold still produces BatchKVCache."""
        from mlx_vlm.apc import make_warm_batch_kv_cache_multi
        from mlx_vlm.models.cache import BatchKVCache

        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE

        lk = [_rand_kv(seq_len=seq_len)[0] for _ in range(2)]
        lv = [_rand_kv(seq_len=seq_len)[1] for _ in range(2)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        picks = [{"matched_blocks": matched, "prefix_len": seq_len}, None]

        warm, _ = make_warm_batch_kv_cache_multi(
            picks, num_layers=2, kv_quant_config=None
        )

        for c in warm:
            assert isinstance(c, BatchKVCache)
        manager.release(matched)


# ---------------------------------------------------------------------------
# Test 15: Harvest from batch_idx > 0
# ---------------------------------------------------------------------------


class TestHarvestNonZeroBatchIdx:
    def test_harvest_batch_idx_1(self):
        """harvest_blocks_from_batch_cache correctly extracts row 1 from multi-row cache."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        seq_len = BLOCK_SIZE
        num_layers = 2

        # Create a 2-row batch cache with different values per row
        row0_k, row0_v = _rand_kv(batch=1, seq_len=seq_len)
        row1_k, row1_v = _rand_kv(batch=1, seq_len=seq_len)
        batch_k = mx.concatenate([row0_k, row1_k], axis=0)  # [2, H, seq_len, D]
        batch_v = mx.concatenate([row0_v, row1_v], axis=0)
        mx.eval(batch_k, batch_v)

        batch_caches = []
        for _ in range(num_layers):
            c = BatchQuantizedKVCache([0, 0], group_size=GROUP_SIZE, bits=BITS)
            c.update_and_fetch(batch_k, batch_v)
            mx.eval(c.keys)
            batch_caches.append(c)

        # Harvest row 1 (not row 0)
        token_ids = list(range(seq_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, batch_caches, batch_idx=1, full_token_ids=token_ids
        )

        assert len(blocks) == 1
        # Block should contain row 1's data, not row 0's
        harvested_k = blocks[0].keys[0]
        mx.eval(harvested_k)
        # Verify it's closer to row1 than row0
        error_vs_row1 = _max_abs_error(harvested_k, row1_k)
        error_vs_row0 = _max_abs_error(harvested_k, row0_k)
        assert error_vs_row1 < 0.1  # should match row1 within quant tolerance
        assert error_vs_row0 > error_vs_row1  # should NOT match row0
        manager.release(blocks)

    def test_harvest_batch_idx_1_with_left_padding(self):
        """Harvest row 1 with different left-padding per row."""
        manager = APCManager(num_blocks=8, block_size=BLOCK_SIZE)
        content_len = BLOCK_SIZE
        num_layers = 2

        # Row 0: left_pad=0, Row 1: left_pad=5
        left_padding = [0, 5]
        total_len = content_len + 5  # both rows same buffer length

        batch_k = mx.random.normal((2, H, total_len, D))
        batch_v = mx.random.normal((2, H, total_len, D))
        mx.eval(batch_k, batch_v)

        batch_caches = []
        for _ in range(num_layers):
            c = BatchQuantizedKVCache(left_padding, group_size=GROUP_SIZE, bits=BITS)
            c.update_and_fetch(batch_k, batch_v)
            mx.eval(c.keys)
            batch_caches.append(c)

        # Harvest row 1 — should skip 5 left-padding tokens
        token_ids = list(range(content_len))
        blocks = harvest_blocks_from_batch_cache(
            manager, batch_caches, batch_idx=1, full_token_ids=token_ids
        )

        assert len(blocks) == 1
        # Row 1 has content_len tokens after skipping left_pad=5
        assert blocks[0].keys[0].shape[2] == BLOCK_SIZE
        manager.release(blocks)


# ---------------------------------------------------------------------------
# Test 16: Type homogeneity — warm and cold caches are same type for extend()
# ---------------------------------------------------------------------------


class TestCacheTypeHomogeneity:
    def test_warm_quantized_matches_cold_quantized_type(self):
        """Warm APC cache (with kv_quant_config) is same type as cold cache from _make_cache.

        This ensures extend() won't crash when merging warm and cold batches.
        """
        from mlx_vlm.apc import make_warm_batch_kv_cache_multi
        from mlx_vlm.generate.ar import _make_cache
        from mlx_vlm.models.cache import BatchQuantizedKVCache

        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        seq_len = 2 * BLOCK_SIZE

        lk = [_rand_kv(seq_len=seq_len)[0] for _ in range(2)]
        lv = [_rand_kv(seq_len=seq_len)[1] for _ in range(2)]
        mx.eval(lk + lv)

        token_ids = list(range(seq_len))
        blocks = manager.store_kv_blocks(token_ids, lk, lv)
        manager.release(blocks)

        matched, _ = manager.lookup_prefix(token_ids)
        picks = [{"matched_blocks": matched, "prefix_len": seq_len}]

        quant_config = {"bits": BITS, "group_size": GROUP_SIZE}
        warm, _ = make_warm_batch_kv_cache_multi(
            picks, num_layers=2, kv_quant_config=quant_config
        )

        # Simulate what cold path produces
        class FakeModel:
            class layers:
                pass

            layers = [None, None]

        cold = _make_cache(FakeModel(), [0], kv_bits=BITS, kv_group_size=GROUP_SIZE)

        # Both should be the same type
        for warm_layer, cold_layer in zip(warm, cold):
            assert type(warm_layer) == type(cold_layer), (
                f"Type mismatch: warm={type(warm_layer).__name__}, "
                f"cold={type(cold_layer).__name__}"
            )
            assert isinstance(warm_layer, BatchQuantizedKVCache)
        manager.release(matched)


# ---------------------------------------------------------------------------
# Test 17: Warm restore last-layer policy + staggered join (#1562 residual)
# ---------------------------------------------------------------------------
# Kept here (not a new file) so APC+kv tests stay co-located with #1174 suite.
# See also PR #1568 review: prefer existing test modules over new files.


KV_CFG = {"bits": BITS, "group_size": GROUP_SIZE}


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

    def test_make_cache_skips_last_layer_for_n_gt_2(self):
        assert _expected_make_cache_types(4) == [
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchKVCache",
        ]

    def test_make_cache_quantizes_all_when_n_le_2(self):
        assert _expected_make_cache_types(2) == [
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
        ]

    def test_make_warm_batch_kv_cache_matches_make_cache_types(self):
        num_layers = 4
        seq_len = 2 * BLOCK_SIZE
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=KV_CFG)
            assert _layer_type_names(warm) == _expected_make_cache_types(num_layers)
            assert isinstance(warm[-1], BatchKVCache)
            assert isinstance(warm[0], BatchQuantizedKVCache)
        finally:
            manager.close()

    def test_make_warm_batch_kv_cache_multi_matches_make_cache_types(self):
        num_layers = 4
        seq_len = 2 * BLOCK_SIZE
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
            pick = {"matched_blocks": matched, "prefix_len": seq_len}
            warm, max_prefix = make_warm_batch_kv_cache_multi(
                [pick, None],
                num_layers=num_layers,
                kv_quant_config=KV_CFG,
            )
            assert max_prefix == seq_len
            assert _layer_type_names(warm) == _expected_make_cache_types(num_layers)
            assert isinstance(warm[-1], BatchKVCache)
            assert warm[-1].left_padding.tolist() == [0, seq_len]
        finally:
            manager.close()

    def test_make_warm_without_kv_config_all_float(self):
        num_layers = 3
        seq_len = BLOCK_SIZE
        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=None)
            assert all(isinstance(c, BatchKVCache) for c in warm)
        finally:
            manager.close()


class TestExtendGenBatchWithWarmRestoredRow:
    """``_extend_cache`` must join live ``_make_cache`` row with APC-warm row."""

    def _live_gen_cache(self, num_layers: int, seq_len: int, left_padding=(0,)):
        class FakeLayer:
            pass

        class FakeModel:
            layers = [FakeLayer() for _ in range(num_layers)]

        caches = _make_cache(
            FakeModel(),
            list(left_padding),
            kv_bits=float(BITS),
            kv_group_size=GROUP_SIZE,
            kv_quant_scheme="uniform",
        )
        for c in caches:
            k, v = _rand_kv(batch=len(left_padding), seq_len=seq_len)
            c.update_and_fetch(k, v)
        return caches

    def test_extend_live_make_cache_with_warm_single_row(self):
        num_layers = 4
        seq_len = 2 * BLOCK_SIZE
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=KV_CFG)
            live = self._live_gen_cache(num_layers, seq_len=seq_len)

            assert _layer_type_names(live) == _layer_type_names(warm)

            extended = _extend_cache(live, warm)
            assert len(extended) == num_layers
            for c in extended:
                assert int(c.offset.shape[0]) == 2
            assert isinstance(extended[-1], BatchKVCache)
            assert not isinstance(extended[-1].keys, tuple)
        finally:
            manager.close()

    def test_extend_mismatched_last_layer_types_raises(self):
        """Historical failure mode: BatchKVCache.extend vs quantized tuple keys."""
        live = BatchKVCache([0])
        k, v = _rand_kv(seq_len=8)
        live.update_and_fetch(k, v)

        other = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        ok, ov = _rand_kv(seq_len=8)
        other.update_and_fetch(ok, ov)
        assert isinstance(other.keys, tuple)

        with pytest.raises(AttributeError, match="shape"):
            live.extend(other)


class TestStaggeredJoinSynthetic:
    """Simulate A live then B APC-warm join under kv-bits (server concurrent shape)."""

    def test_staggered_warm_join_layer_types_compatible(self):
        num_layers = 4
        seq_len = 2 * BLOCK_SIZE
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_prefix_blocks(manager, num_layers, seq_len, token_ids)

            class FakeLayer:
                pass

            class FakeModel:
                layers = [FakeLayer() for _ in range(num_layers)]

            live = _make_cache(
                FakeModel(),
                [0],
                kv_bits=float(BITS),
                kv_group_size=GROUP_SIZE,
                kv_quant_scheme="uniform",
            )
            for c in live:
                k, v = _rand_kv(seq_len=seq_len + 4)
                c.update_and_fetch(k, v)

            warm = make_warm_batch_kv_cache(matched, kv_quant_config=KV_CFG)
            assert _layer_type_names(live) == _layer_type_names(warm), (
                f"layer type mismatch blocks continuous-batching join: "
                f"live={_layer_type_names(live)} warm={_layer_type_names(warm)}"
            )
            extended = _extend_cache(live, warm)
            assert int(extended[0].offset.shape[0]) == 2
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
