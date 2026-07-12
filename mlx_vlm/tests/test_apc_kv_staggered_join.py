"""Regression tests for APC + kv-bits continuous-batching join (#1562 residual).

Context
-------
#1568 fixed multi-row quantized SDPA mask broadcast and
``BatchQuantizedKVCache.prepare`` / ``finalize``. Simultaneous
``BatchGenerator.insert([a, b])`` with ``kv_bits`` works.

A residual remains on the **server / continuous-batching join** path:

1. Request A starts with ``_make_cache`` → last full-attention layer is
   **unquantized** ``BatchKVCache`` (n > 2).
2. Request A stores APC blocks after prefill.
3. Request B joins mid-flight with an APC hit; warm restore goes through
   ``make_warm_batch_kv_cache*`` which historically built
   **all** layers as ``BatchQuantizedKVCache``.
4. ``_extend_cache`` calls ``BatchKVCache.extend`` on the last layer with
   ``other.keys`` a quantized **tuple** →
   ``AttributeError: 'tuple' object has no attribute 'shape'``.

These tests pin the contract:

* Warm restore layer types must match ``_make_cache`` under the same
  ``kv_bits`` policy (last layer float when n > 2).
* Extending a live gen batch with a warm-restored row must succeed.
* Staggered cold→warm join under APC + kv-bits must not crash (synthetic
  unit path; optional live model test is marked).

TDD: written to fail on main until warm restore respects last-layer policy
(or extend is made type-safe across the mismatch).
"""

from __future__ import annotations

from typing import List

import mlx.core as mx
import pytest

from mlx_vlm.apc import (
    APCManager,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
)
from mlx_vlm.generate.ar import _extend_cache, _make_cache
from mlx_vlm.models.cache import BatchKVCache, BatchQuantizedKVCache

B, H, D = 1, 2, 32
GROUP_SIZE = 32
BITS = 8
BLOCK_SIZE = 16
KV_CFG = {"bits": BITS, "group_size": GROUP_SIZE}


def _rand_kv(batch: int = B, seq_len: int = 32, heads: int = H, dim: int = D):
    k = mx.random.normal((batch, heads, seq_len, dim))
    v = mx.random.normal((batch, heads, seq_len, dim))
    mx.eval(k, v)
    return k, v


def _store_blocks(
    manager: APCManager, num_layers: int, seq_len: int, token_ids: List[int]
):
    """Store float APC blocks for a synthetic multi-layer prefix."""
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
    """Mirror ``_make_cache`` policy when model has plain layers + kv_bits."""

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


# ---------------------------------------------------------------------------
# Layer-type parity: warm restore vs _make_cache
# ---------------------------------------------------------------------------


class TestWarmRestoreLayerTypesMatchMakeCache:
    """APC warm path must not quantize the last layer when _make_cache does not."""

    def test_make_cache_skips_last_layer_for_n_gt_2(self):
        types = _expected_make_cache_types(4)
        assert types == [
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchQuantizedKVCache",
            "BatchKVCache",
        ]

    def test_make_cache_quantizes_all_when_n_le_2(self):
        # Document baseline: n<=2 keeps last layer quantized too.
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
            matched = _store_blocks(manager, num_layers, seq_len, token_ids)
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
            matched = _store_blocks(manager, num_layers, seq_len, token_ids)
            pick = {"matched_blocks": matched, "prefix_len": seq_len}
            # One warm row + one cold row (server mixed-prefill shape)
            warm, max_prefix = make_warm_batch_kv_cache_multi(
                [pick, None],
                num_layers=num_layers,
                kv_quant_config=KV_CFG,
            )
            assert max_prefix == seq_len
            assert _layer_type_names(warm) == _expected_make_cache_types(num_layers)
            assert isinstance(warm[-1], BatchKVCache)
            # Multi-row left padding on cold row
            assert warm[-1].left_padding.tolist() == [0, seq_len]
        finally:
            manager.close()

    def test_make_warm_without_kv_config_all_float(self):
        num_layers = 3
        seq_len = BLOCK_SIZE
        manager = APCManager(num_blocks=16, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=None)
            assert all(isinstance(c, BatchKVCache) for c in warm)
        finally:
            manager.close()


# ---------------------------------------------------------------------------
# Extend-join: gen batch (_make_cache) + warm-restored row
# ---------------------------------------------------------------------------


class TestExtendGenBatchWithWarmRestoredRow:
    """``_extend_cache`` must join a live gen row with an APC-warm row under kv-bits."""

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
        # Fill like a completed prefill so keys are non-None
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
            matched = _store_blocks(manager, num_layers, seq_len, token_ids)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=KV_CFG)
            live = self._live_gen_cache(num_layers, seq_len=seq_len)

            assert _layer_type_names(live) == _layer_type_names(warm)

            extended = _extend_cache(live, warm)
            assert len(extended) == num_layers
            # Batch dim grows 1 → 2
            for c in extended:
                assert int(c.offset.shape[0]) == 2
            # Last layer remains dense BatchKVCache
            assert isinstance(extended[-1], BatchKVCache)
            assert not isinstance(extended[-1].keys, tuple)
        finally:
            manager.close()

    def test_extend_fails_today_if_last_layer_types_mismatch(self):
        """Guardrail: document the exact failure mode from server concurrent APC+kv.

        If someone "fixes" types incorrectly by quantizing the live last layer
        only, this still records the extend crash on the historical mismatch.
        """
        live = BatchKVCache([0])
        k, v = _rand_kv(seq_len=8)
        live.update_and_fetch(k, v)

        other = BatchQuantizedKVCache([0], group_size=GROUP_SIZE, bits=BITS)
        ok, ov = _rand_kv(seq_len=8)
        other.update_and_fetch(ok, ov)
        assert isinstance(other.keys, tuple)

        with pytest.raises(AttributeError, match="shape"):
            live.extend(other)


# ---------------------------------------------------------------------------
# Integration-style: staggered join using BatchGenerator (no HF download)
# ---------------------------------------------------------------------------


class TestStaggeredJoinSynthetic:
    """Simulate A generating then B joining with APC warm under kv-bits."""

    def test_staggered_warm_join_layer_types_compatible(self):
        num_layers = 4
        seq_len = 2 * BLOCK_SIZE
        manager = APCManager(num_blocks=32, block_size=BLOCK_SIZE)
        try:
            token_ids = list(range(seq_len))
            matched = _store_blocks(manager, num_layers, seq_len, token_ids)

            # A: live gen path
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
                k, v = _rand_kv(seq_len=seq_len + 4)  # prefill + a few decode steps
                c.update_and_fetch(k, v)

            # B: APC hit warm restore (server second client)
            warm = make_warm_batch_kv_cache(matched, kv_quant_config=KV_CFG)

            # Contract under test
            assert _layer_type_names(live) == _layer_type_names(warm), (
                f"layer type mismatch blocks continuous-batching join: "
                f"live={_layer_type_names(live)} warm={_layer_type_names(warm)}"
            )
            extended = _extend_cache(live, warm)
            assert int(extended[0].offset.shape[0]) == 2
        finally:
            manager.close()


# ---------------------------------------------------------------------------
# Optional live-model smoke (skipped unless RUN_LIVE_APC_KV_JOIN=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_LIVE_APC_KV_JOIN", "0") != "1",
    reason="Set RUN_LIVE_APC_KV_JOIN=1 to run live model staggered join smoke",
)
def test_live_batch_generator_staggered_apc_kv_join():
    """Live repro of server concurrent failure (cold A, then B joins after A stores)."""
    import os

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
