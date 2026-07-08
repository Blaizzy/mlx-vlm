"""Tests for APC integration with quantized KV caches.

Uses real QuantizedKVCache and BatchQuantizedKVCache objects with small
dimensions. No mocking of cache behavior — these tests exercise the same
code paths production uses.

TDD: written BEFORE the implementation. Each test targets a specific
behavior required for issue #1174 (APC + KV-cache quantization).
"""

from __future__ import annotations

import mlx.core as mx

from mlx_vlm.apc import (
    APCManager,
    _cache_entry_supports_block_apc,
    _cache_entry_supports_exact_apc,
    harvest_blocks_from_batch_cache,
    make_warm_batch_kv_cache,
    make_warm_kv_cache,
    model_apc_mode,
)
from mlx_vlm.models.cache import BatchQuantizedKVCache, KVCache, QuantizedKVCache

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
        for c in warm:
            assert isinstance(c, QuantizedKVCache)
            assert c.offset == seq_len
            assert c.bits == BITS
            assert c.group_size == GROUP_SIZE
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
