"""Regression tests for exact-mode APC reuse on the DFlash speculative path.

Covered behavior (all pure cache math, no model load, no downloads):
  * store order            -- _store_row stores checkpoints FIRST and the
                              full-length entry LAST, so a row does not evict
                              its OWN full entry out of the 2-entry LRU.
  * store floor            -- _store_row refuses rows below STORE_MIN_TOKENS
                              (APC_DFLASH_STORE_MIN_TOKENS, read at import).
  * rotating-window snapshot compaction
                           -- RotatingKVCacheCloneAdapter.clone keeps only the
                              last max_size positions of a one-shot retained-all
                              buffer (O(window) not O(prompt) per entry), and a
                              warm merge/decode round-trip through the compacted
                              snapshot matches the cold window exactly.
  * keep>0 guard           -- attention-sink caches (keep > 0) are refused at
                              store (_snapshot_reuse_safe), refused at mode
                              probe (dflash_apc_mode), and never compacted.
  * identical re-store dedup
                           -- store_exact_cache skips the multi-GB reclone on a
                              byte-identical re-store, counts it in
                              stats.exact_store_dedups, keeps the entry object.
  * clone=False ownership  -- store_exact_cache(clone=False) takes the caller's
                              snapshot as-is (identity), while lookup still
                              hands out clones.
  * static-prefix boundary -- store_batch pins a checkpoint at
                              static_prefix_boundary (second harmony <|start|>
                              header, min STATIC_PREFIX_MIN_TOKENS) through the
                              same _store_row path as guard + full entries.
  * composite/sink probes  -- compacted-entry re-clone round-trip, in-place
                              decode continuation on a restored compacted
                              cache, store_prefix_checkpoint success path,
                              cross-row `seen` dedup in store_batch, and the
                              dflash_apc_mode gates (single make_cache build).

Like test_apc.py this exercises APC mechanics with synthetic arrays; the
tokenizer is a tiny fake that renders the harmony "<|start|>" header as one id.
"""

from __future__ import annotations

import importlib
import os

# The store floor is read once when apc_dflash is imported. Set the knob first;
# if another test module in this session already imported apc_dflash with a
# different floor, reload it so the override takes.
os.environ["APC_DFLASH_STORE_MIN_TOKENS"] = "32"
# Deliberately NOT setting APC_EXACT_CACHE_ENTRIES here: the LRU-order test
# needs the default cap of 2. One boundary test widens it for a single manager.

import mlx.core as mx

from mlx_vlm.apc import APCManager, snapshot_prompt_cache_row
from mlx_vlm.models.cache import (
    BatchKVCache,
    BatchRotatingKVCache,
    CacheList,
    KVCache,
    RotatingKVCache,
)
from mlx_vlm.speculative import apc_dflash

if apc_dflash.STORE_MIN_TOKENS != 32:
    apc_dflash = importlib.reload(apc_dflash)


# ============================================================================
# Helpers
# ============================================================================


def kv(pos, n):
    """Deterministic K/V of shape (1, 2, n, 4): slot t encodes position pos+t."""
    base = mx.arange(pos, pos + n, dtype=mx.float32).reshape(1, 1, n, 1)
    k = mx.contiguous(mx.broadcast_to(base, (1, 2, n, 4)))
    v = mx.contiguous(mx.broadcast_to(base + 0.5, (1, 2, n, 4)))
    return k, v


def one_shot_kvcache(n):
    c = KVCache()
    c.update_and_fetch(*kv(0, n))
    return c


def one_shot_rot(n, max_size, keep=0):
    c = RotatingKVCache(max_size=max_size, keep=keep)
    c.update_and_fetch(*kv(0, n))
    return c


def aeq(a, b):
    return bool(mx.array_equal(a, b))


def new_manager():
    return APCManager(num_blocks=4, block_size=16)


class FakeHarmonyTok:
    """encode('<|start|>') -> [7], so static_prefix_boundary looks for id 7."""

    bos_token_id = None

    def encode(self, s, **kw):
        if s == "<|start|>":
            return [7]
        return [1, 2, 3]


# ============================================================================
# 1. store order / full-entry survival
# ============================================================================
# With full-then-checkpoints store order the guard checkpoint would evict the
# row's OWN full entry out of the 2-entry LRU -- the one a growing
# conversation needs next turn. Checkpoints first, full last.


def test_store_order_full_entry_survives():
    mgr = new_manager()
    assert mgr._exact_cache_max == 2, "default LRU cap is 2"
    ids = list(range(200))
    c = one_shot_kvcache(200)
    ok = apc_dflash._store_row(mgr, ids, [c], 0, 7, boundaries=None, seen=set())
    assert ok is True, "_store_row returns True"
    hit, plen = mgr.lookup_exact_cache(ids + [9999], extra_hash=7)
    assert hit is not None, "full entry survived the LRU (hit)"
    assert plen == 200, "full entry prefix_len == 200"
    hit2, plen2 = mgr.lookup_exact_cache(ids, extra_hash=7)
    assert (hit2 is not None, plen2) == (True, 184), "guard checkpoint hit at 200-16"


# ============================================================================
# 2. store floor
# ============================================================================


def test_store_floor():
    assert apc_dflash.STORE_MIN_TOKENS == 32, "STORE_MIN_TOKENS env override took"
    mgr = new_manager()
    ids = list(range(20))  # < 32 floor
    c = one_shot_kvcache(20)
    ok = apc_dflash._store_row(mgr, ids, [c], 0, 7, boundaries=None, seen=set())
    assert ok is False, "sub-floor _store_row refused"
    hit, plen = mgr.lookup_exact_cache(ids + [1], extra_hash=7)
    assert (hit, plen) == (None, 0), "sub-floor lookup misses"


# ============================================================================
# 3. rotating snapshot compaction
# ============================================================================
# A one-shot prefill leaves a RotatingKVCache retaining ALL tokens; the
# snapshot must keep only the last max_size positions (keep == 0 only).


def test_snapshot_compaction():
    c = one_shot_rot(512, 64)
    snap = snapshot_prompt_cache_row([c], 0)
    e = snap[0]
    assert int(e.keys.shape[2]) == 64, "compacted slot count == window (64)"
    assert int(e.offset) == 512, "compacted offset keeps logical length (512)"
    assert int(e._idx) == 64, "compacted _idx == window (64)"
    lk, lv = kv(511, 1)
    assert aeq(e.keys[..., -1:, :], lk) and aeq(
        e.values[..., -1:, :], lv
    ), "last slot holds position 511"
    assert int(c.keys.shape[2]) == 512, "source cache untouched (512 slots)"

    c40 = one_shot_rot(40, 64)
    s40 = snapshot_prompt_cache_row([c40], 0)
    assert int(s40[0].keys.shape[2]) == 40, "no compaction below window (40 slots)"

    ck = one_shot_rot(512, 64, keep=4)
    sk = snapshot_prompt_cache_row([ck], 0)
    assert int(sk[0].keys.shape[2]) == 512, "keep>0 snapshot NOT compacted (512 slots)"


# ============================================================================
# 4. warm round-trip equivalence (compaction active)
# ============================================================================
# Cold: one-shot 320 tokens. Warm: one-shot 300 -> compacted snapshot ->
# batch merge -> prefill the 20-token suffix -> extract. Windows must match.


def test_warm_round_trip():
    W, cp_len, suf, total = 64, 300, 20, 320

    cold = one_shot_rot(total, W)
    cold_k = cold._temporal_order(cold.keys)[..., -W:, :]
    cold_v = cold._temporal_order(cold.values)[..., -W:, :]

    warm_src = one_shot_rot(cp_len, W)
    snap = snapshot_prompt_cache_row([warm_src], 0)
    assert int(snap[0].keys.shape[2]) == W, "snapshot compacted to window"
    assert int(snap[0].offset) == cp_len, "snapshot logical offset == 300"

    batch = BatchRotatingKVCache.merge(snap)
    batch.prepare(right_padding=[0], lengths=[suf])
    batch.update_and_fetch(*kv(cp_len, suf))  # suffix positions 300..319
    batch.finalize()
    out = batch.extract(0)

    assert int(out.offset) == total, "warm extract offset == 320"
    assert aeq(out.keys[..., -W:, :], cold_k), "warm last-64 keys == cold window"
    assert aeq(out.values[..., -W:, :], cold_v), "warm last-64 values == cold window"


# ============================================================================
# 5. keep>0 store guard
# ============================================================================


def test_keep_store_guard():
    mgr = new_manager()
    ids = list(range(200))
    ck = one_shot_rot(200, 64, keep=4)
    ok = apc_dflash._store_row(mgr, ids, [ck], 0, 7, boundaries=None, seen=set())
    assert ok is False, "keep=4 _store_row refused"
    hit, plen = mgr.lookup_exact_cache(ids + [1], extra_hash=7)
    assert (hit, plen) == (None, 0), "keep=4 lookup misses"
    assert (
        apc_dflash._snapshot_reuse_safe([RotatingKVCache(max_size=64, keep=4)])
        is False
    ), "keep=4 unsafe"
    assert (
        apc_dflash._snapshot_reuse_safe([RotatingKVCache(max_size=64, keep=0)])
        is True
    ), "keep=0 safe"
    assert apc_dflash._snapshot_reuse_safe([KVCache()]) is True, "KVCache safe"
    assert apc_dflash._snapshot_reuse_safe([None]) is True, "None entry safe"


# ============================================================================
# 6. keep>0 lookup gate
# ============================================================================


def test_keep_lookup_gate():
    class SinkLM:
        def make_cache(self):
            return [RotatingKVCache(max_size=64, keep=4)]

    class PlainLM:
        def make_cache(self):
            return [RotatingKVCache(max_size=64, keep=0)]

    assert (
        apc_dflash.dflash_apc_mode(SinkLM()) is None
    ), "dflash_apc_mode refuses keep=4 model"
    assert (
        apc_dflash.dflash_apc_mode(PlainLM()) == "exact"
    ), "dflash_apc_mode allows keep=0 model"


# ============================================================================
# 7. identical re-store dedup
# ============================================================================


def test_restore_dedup():
    mgr = new_manager()
    ids = list(range(200))
    c = one_shot_kvcache(200)
    ok1 = mgr.store_exact_cache(ids, [c], extra_hash=5)  # clone defaults True
    assert ok1 is True, "first store ok"
    entry = next(e for e in mgr._exact_cache.values() if e.extra_hash == 5)
    pc_before = entry.prompt_cache
    ok2 = mgr.store_exact_cache(ids, [c], extra_hash=5)
    assert ok2 is True, "identical re-store returns True"
    assert mgr.stats.exact_store_dedups == 1, "exact_store_dedups == 1"
    entry_after = next(e for e in mgr._exact_cache.values() if e.extra_hash == 5)
    assert entry_after.prompt_cache is pc_before, "entry prompt_cache identity kept"
    ok3 = mgr.store_exact_cache(ids, [c], extra_hash=6)
    assert ok3 is True, "different extra_hash stores (True)"
    assert mgr.stats.exact_store_dedups == 1, "different extra_hash did NOT dedup"
    stats = mgr.stats.snapshot(mgr.num_blocks, mgr.block_size)
    assert "exact_store_dedups" in stats, "exact_store_dedups in stats snapshot"


# ============================================================================
# 8. clone=False ownership
# ============================================================================


def test_clone_false_ownership():
    mgr = new_manager()
    ids = list(range(200))
    c = one_shot_kvcache(200)
    snap = snapshot_prompt_cache_row([c], 0)
    ok = mgr.store_exact_cache(ids, snap, extra_hash=3, clone=False)
    assert ok is True, "clone=False store ok"
    entry = next(e for e in mgr._exact_cache.values() if e.extra_hash == 3)
    assert entry.prompt_cache[0] is snap[0], "stored entry owns snap[0] (identity)"
    hit, plen = mgr.lookup_exact_cache(ids + [1], extra_hash=3)
    assert plen == 200, "lookup hits at 200"
    assert hit is not None, "lookup returned a cache"
    assert hit[0] is not snap[0], "lookup hands out a clone (not snap[0])"
    assert aeq(hit[0].keys[..., :200, :], snap[0].keys[..., :200, :]) and aeq(
        hit[0].values[..., :200, :], snap[0].values[..., :200, :]
    ), "clone content equals stored snapshot"


# ============================================================================
# 9. static-prefix boundary unification in store_batch
# ============================================================================
# store_batch pins a checkpoint at the SECOND harmony <|start|> header id when
# it sits at >= STATIC_PREFIX_MIN_TOKENS (2048, frozen at def time).


def test_store_batch_boundary_positive():
    fake = FakeHarmonyTok()

    # Stores land as boundary(2050), guard(2084), full(2100); the default
    # 2-entry LRU would evict the boundary, so this ONE manager is built with
    # APC_EXACT_CACHE_ENTRIES=4 (read in APCManager.__init__).
    ids = [7] + [1] * 2049 + [7] + [2] * 49  # len 2100, second 7 at index 2050
    assert (len(ids), ids[2050]) == (2100, 7), "positive ids shape sane"
    assert (
        apc_dflash.static_prefix_boundary(fake, ids) == 2050
    ), "static_prefix_boundary finds index 2050"
    os.environ["APC_EXACT_CACHE_ENTRIES"] = "4"
    try:
        mgr = APCManager(num_blocks=4, block_size=16)
    finally:
        del os.environ["APC_EXACT_CACHE_ENTRIES"]
    assert mgr._exact_cache_max == 4, "widened LRU cap took"
    c = one_shot_kvcache(2100)
    apc_dflash.store_batch(
        mgr,
        model=None,
        processor=fake,
        all_input_ids=[ids],
        prompt_cache=[c],
        extra_hashes=[11],
    )
    hit, plen = mgr.lookup_exact_cache(ids[:2050] + [8888], extra_hash=11)
    assert (hit is not None, plen) == (True, 2050), "boundary checkpoint hit at 2050"
    hit, plen = mgr.lookup_exact_cache(ids + [8888], extra_hash=11)
    assert plen == 2100, "full entry also present (2100)"
    hit, plen = mgr.lookup_exact_cache(ids, extra_hash=11)
    assert plen == 2084, "guard checkpoint also present (2084)"


def test_store_batch_boundary_negative():
    fake = FakeHarmonyTok()
    # Second header below the 2048 minimum -> no boundary checkpoint.
    mgr_n = new_manager()  # default cap 2 is fine: only guard + full stored
    ids_n = [7] + [1] * 99 + [7] + [2] * 99  # len 200, second 7 at index 100
    assert (len(ids_n), ids_n[100]) == (200, 7), "negative ids shape sane"
    assert (
        apc_dflash.static_prefix_boundary(fake, ids_n) is None
    ), "static_prefix_boundary rejects sub-min index"
    c_n = one_shot_kvcache(200)
    apc_dflash.store_batch(
        mgr_n,
        model=None,
        processor=fake,
        all_input_ids=[ids_n],
        prompt_cache=[c_n],
        extra_hashes=[12],
    )
    hit, plen = mgr_n.lookup_exact_cache(ids_n[:100] + [8888], extra_hash=12)
    assert (hit, plen) == (None, 0), "no checkpoint at sub-min boundary"
    hit, plen = mgr_n.lookup_exact_cache(ids_n, extra_hash=12)
    assert plen == 184, "negative row still stored guard (184)"


# ============================================================================
# 10. trimmed-checkpoint exactness
# ============================================================================
# _store_trimmed on an unwrapped RotatingKVCache must store exactly
# tokens[:cp]. (STORE_MIN_TOKENS does not apply to _store_trimmed.)


def test_trimmed_exactness():
    mgr = new_manager()
    ids = list(range(40))
    c = one_shot_rot(40, 64)  # unwrapped: offset 40 <= max_size 64
    ok = apc_dflash._store_trimmed(mgr, ids, [c], 0, 3, 30, 40)
    assert ok is True, "_store_trimmed returns True"
    hit, plen = mgr.lookup_exact_cache(ids, extra_hash=3)
    assert (hit is not None, plen) == (True, 30), "lookup hits at cp=30"
    row = BatchRotatingKVCache.merge(hit).extract(0)
    ek, ev = kv(0, 30)
    assert aeq(row.keys, ek), "restored keys == positions 0..29"
    assert aeq(row.values, ev), "restored values == positions 0..29"
    assert int(row.offset) == 30, "restored offset == 30"


# ============================================================================
# 11. second-clone path round-trip
# ============================================================================
# lookup_exact_cache clones the stored entry through
# RotatingKVCacheCloneAdapter AGAIN. On an already-compacted entry (offset 512
# != _idx 64) the compaction branch must NOT re-fire -- it requires offset ==
# _idx -- so the clone is a verbatim fresh copy that still merges/extracts to
# the correct window.


def test_second_clone_round_trip():
    mgr = new_manager()
    ids = list(range(512))
    c = one_shot_rot(512, 64)
    snap = snapshot_prompt_cache_row([c], 0)  # compacted: 64 slots, offset 512
    ok = mgr.store_exact_cache(ids, snap, extra_hash=21, clone=False)
    assert ok is True, "compacted store ok"
    hit, hlen = mgr.lookup_exact_cache(ids + [9999], extra_hash=21)
    assert (hit is not None, hlen) == (True, 512), "lookup hits at 512"
    assert int(hit[0].keys.shape[2]) == 64, "re-clone keeps window slots (64)"
    assert int(hit[0].offset) == 512, "re-clone keeps logical offset (512)"
    assert int(hit[0]._idx) == 64, "re-clone keeps _idx (64)"
    assert hit[0].keys is not snap[0].keys, "re-clone is a fresh copy"
    row = BatchRotatingKVCache.merge(hit).extract(0)
    ek, ev = kv(448, 64)
    assert aeq(row.keys, ek), "merged extract keys == positions 448..511"
    assert aeq(row.values, ev), "merged extract values == positions 448..511"


# ============================================================================
# 12. in-place decode continuation on a restored compacted cache
# ============================================================================
# Warm: the lookup clone of a compacted 512-token snapshot (64 slots, offset
# 512). Cold: a live one-shot 512-token cache (512 retained slots). One S=1
# decode step at position 512 must converge them bit-identically: the cold
# path trims its buffer to the 64-token window and rotates; the warm cache
# starts at exactly that window.


def test_decode_continuation():
    mgr = new_manager()
    ids = list(range(512))
    src = one_shot_rot(512, 64)
    snap = snapshot_prompt_cache_row([src], 0)
    mgr.store_exact_cache(ids, snap, extra_hash=22, clone=False)
    hit, hlen = mgr.lookup_exact_cache(ids + [9999], extra_hash=22)
    assert (hit is not None, hlen) == (True, 512), "warm lookup hits at 512"
    warm = hit[0]
    cold = one_shot_rot(512, 64)  # live cache, NOT compacted (512 slots)
    assert int(cold.keys.shape[2]) == 512, "cold retains all slots pre-decode"
    dk, dv = kv(512, 1)  # single decode token at position 512, shape (1,2,1,4)
    warm.update_and_fetch(dk, dv)
    cold.update_and_fetch(dk, dv)
    assert aeq(warm.keys, cold.keys), "warm keys == cold keys after step"
    assert aeq(warm.values, cold.values), "warm values == cold values after step"
    assert (int(warm.offset), int(cold.offset)) == (513, 513), "offsets converge"
    assert int(warm._idx) == int(cold._idx), "_idx converges"


# ============================================================================
# 13. mid-prefill checkpoint success path
# ============================================================================
# store_prefix_checkpoint stores the cache state captured DURING prefill at
# cp, keyed as ids[:cp] -- no trim, so the _store_trimmed wrap guard does not
# apply.


def test_prefix_checkpoint_success():
    mgr = new_manager()
    ids = list(range(60))  # FULL prompt is longer than cp
    c = one_shot_rot(40, 64)  # cache state mid-prefill at cp=40
    fake = FakeHarmonyTok()
    ok = apc_dflash.store_prefix_checkpoint(
        mgr, None, fake, ids, [c], 40, extra_hash=23
    )
    assert ok is True, "store_prefix_checkpoint returns True"
    hit, plen = mgr.lookup_exact_cache(ids, extra_hash=23)
    assert (hit is not None, plen) == (True, 40), "lookup hits at cp=40"
    row = BatchRotatingKVCache.merge(hit).extract(0)
    ek, ev = kv(0, 40)
    assert aeq(row.keys, ek), "restored keys == positions 0..39"
    assert aeq(row.values, ev), "restored values == positions 0..39"
    assert int(row.offset) == 40, "restored offset == 40"


# ============================================================================
# 14. cross-row `seen` dedup in store_batch
# ============================================================================
# Two IDENTICAL rows in one batched cache: row 0 stores guard(184) +
# full(200); row 1's identical stores are suppressed by the cross-row `seen`
# set BEFORE reaching the manager, so exact_stores stays 2 and the
# manager-level dedup counter never fires. (No boundary checkpoint: these ids
# carry no second harmony start header at >= 2048; guard is the default 16.)


def test_cross_row_seen_dedup():
    mgr = new_manager()
    fake = FakeHarmonyTok()
    ids = list(range(1000, 1200))  # len 200, no header id 7 anywhere
    k, v = kv(0, 200)
    batch = BatchKVCache([0, 0])
    batch.update_and_fetch(
        mx.concatenate([k, k], axis=0), mx.concatenate([v, v], axis=0)
    )  # two identical rows, shape (2, 2, 200, 4)
    apc_dflash.store_batch(
        mgr,
        model=None,
        processor=fake,
        all_input_ids=[ids, ids],
        prompt_cache=[batch],
        extra_hashes=[27, 27],
    )
    assert mgr.stats.exact_stores == 2, "exact_stores == 2 (row 0 only)"
    assert mgr.stats.exact_store_dedups == 0, "manager-level dedups == 0"
    hit, plen = mgr.lookup_exact_cache(ids + [1], extra_hash=27)
    assert (hit is not None, plen) == (True, 200), "full entry survived (200)"
    hit, plen = mgr.lookup_exact_cache(ids, extra_hash=27)
    assert (hit is not None, plen) == (True, 184), "guard checkpoint present (184)"


# ============================================================================
# 15. mode-classification gates (single make_cache build)
# ============================================================================
# dflash_apc_mode builds make_cache() ONCE and classifies block/exact itself.
# All-block-eligible models, models without make_cache, and models whose
# make_cache raises must all degrade to None (cold path); a mixed keep=0
# Rotating+KVCache list is "exact".


def test_mode_classification_gates():
    class BlockLM:
        def make_cache(self):
            return [KVCache()]  # all block-eligible -> "block" -> refused here

    class NoMakeCacheLM:
        pass

    class RaisingLM:
        def make_cache(self):
            raise RuntimeError("boom")

    class MixedLM:
        def make_cache(self):
            return [RotatingKVCache(max_size=64, keep=0), KVCache()]

    assert apc_dflash.dflash_apc_mode(BlockLM()) is None, "all-block model refused"
    assert (
        apc_dflash.dflash_apc_mode(NoMakeCacheLM()) is None
    ), "missing make_cache refused"
    assert (
        apc_dflash.dflash_apc_mode(RaisingLM()) is None
    ), "raising make_cache refused"
    assert (
        apc_dflash.dflash_apc_mode(MixedLM()) == "exact"
    ), "mixed keep=0 Rotating+KVCache -> exact"


# ============================================================================
# 16. composite sink guard + single-probe accounting
# ============================================================================
# Exact eligibility recurses into CacheList/tuple composites, so the keep>0
# sink guard must recurse too -- a composite hiding a RotatingKVCache(keep>0)
# must be refused at lookup AND at store. The whole-batch gate probes
# make_cache exactly once per lookup_batch call, and the mode probe runs
# AFTER the cheap request-local guards in the public lookup().


def test_composite_sink_and_probe_count():
    class CompositeSinkLM:
        def make_cache(self):
            return [CacheList(RotatingKVCache(max_size=64, keep=4)), KVCache()]

    class TupleSinkLM:
        def make_cache(self):
            return [(RotatingKVCache(max_size=64, keep=4), KVCache())]

    class CompositeCleanLM:
        def make_cache(self):
            return [CacheList(RotatingKVCache(max_size=64, keep=0)), KVCache()]

    assert (
        apc_dflash.dflash_apc_mode(CompositeSinkLM()) is None
    ), "CacheList-nested keep=4 refused"
    assert (
        apc_dflash.dflash_apc_mode(TupleSinkLM()) is None
    ), "tuple-nested keep=4 refused"
    assert (
        apc_dflash.dflash_apc_mode(CompositeCleanLM()) == "exact"
    ), "CacheList-nested keep=0 still exact"
    assert (
        apc_dflash._snapshot_reuse_safe(
            [CacheList(RotatingKVCache(max_size=64, keep=4))]
        )
        is False
    ), "nested sink unsafe"
    assert (
        apc_dflash._snapshot_reuse_safe(
            [CacheList(RotatingKVCache(max_size=64, keep=0)), None]
        )
        is True
    ), "nested keep=0 safe"

    class CountingLM:
        def __init__(self):
            self.calls = 0

        def make_cache(self):
            self.calls += 1
            return [RotatingKVCache(max_size=64, keep=0), KVCache()]

    mgr = APCManager(num_blocks=4, block_size=16)
    lm = CountingLM()
    ids = list(range(200))
    plan = apc_dflash.lookup_batch(
        mgr, lm, None, FakeHarmonyTok(), [ids, ids, ids], [None, None, None]
    )
    assert plan is None, "empty-manager lookup_batch returns None"
    assert lm.calls == 1, "make_cache probed once for the whole batch"

    lm2 = CountingLM()
    assert (
        apc_dflash.lookup(None, lm2, None, FakeHarmonyTok(), ids, None) is None
        and lm2.calls == 0
    ), "lookup(None manager) short-circuits before mode probe"
