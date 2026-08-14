from __future__ import annotations

import mlx.core as mx

from mlx_vlm.apc import APCManager, ReuseOutcome, apc_lookup_decision, apc_lookup_plan

BLOCK = 16
LAYERS, HEADS, DIM = 2, 2, 8


def _kv(seq_len):
    keys = [mx.zeros((1, HEADS, seq_len, DIM)) for _ in range(LAYERS)]
    return keys, [mx.zeros((1, HEADS, seq_len, DIM)) for _ in range(LAYERS)]


def _decide(manager, tokens, *, apc_mode="block", suffix_ok=True, prefix_media=False):
    return apc_lookup_decision(
        manager,
        tokens,
        extra_hash=0,
        apc_mode=apc_mode,
        safe_lookup_min=0,
        suffix_is_text_only=lambda prefix_len: suffix_ok,
        prefix_has_media=lambda prefix_len: prefix_media,
    )


def test_short_prompt_reports_prompt_too_short():
    manager = APCManager(num_blocks=8, block_size=BLOCK)

    decision = _decide(manager, [7])

    assert not decision.reused
    assert decision.outcome == ReuseOutcome.PROMPT_TOO_SHORT


def test_cold_lookup_reports_no_stored_prefix():
    manager = APCManager(num_blocks=8, block_size=BLOCK)

    decision = _decide(manager, list(range(4 * BLOCK)))

    assert not decision.reused
    assert decision.outcome == ReuseOutcome.NO_STORED_PREFIX


def test_block_hit_reports_block_and_its_length():
    manager = APCManager(num_blocks=8, block_size=BLOCK)
    stored = list(range(4 * BLOCK))
    keys, values = _kv(len(stored))
    manager.release(manager.store_kv_blocks(stored, keys, values))

    decision = _decide(manager, stored + list(range(900, 908)))

    assert decision.reused
    assert decision.outcome == ReuseOutcome.BLOCK
    assert decision.prefix_len > 0
    manager.release(decision.plan["matched_blocks"])


def test_media_in_the_suffix_reports_media_suffix_not_a_cold_miss():
    manager = APCManager(num_blocks=8, block_size=BLOCK)
    stored = list(range(4 * BLOCK))
    keys, values = _kv(len(stored))
    manager.release(manager.store_kv_blocks(stored, keys, values))

    decision = _decide(manager, stored + list(range(900, 908)), suffix_ok=False)

    assert not decision.reused
    assert decision.outcome == ReuseOutcome.MEDIA_SUFFIX


def test_media_in_the_prefix_is_distinct_from_nothing_stored():
    manager = APCManager(num_blocks=8, block_size=BLOCK)
    stored = list(range(4 * BLOCK))
    keys, values = _kv(len(stored))
    manager.release(manager.store_kv_blocks(stored, keys, values))

    decision = _decide(manager, stored + list(range(900, 908)), prefix_media=True)

    assert not decision.reused
    assert decision.outcome == ReuseOutcome.MEDIA_PREFIX


def test_outcomes_accumulate_on_the_manager():
    manager = APCManager(num_blocks=8, block_size=BLOCK)

    _decide(manager, [1])
    _decide(manager, list(range(4 * BLOCK)))
    _decide(manager, list(range(4 * BLOCK)))

    counts = manager.stats_snapshot()["reuse_outcomes"]
    assert counts[ReuseOutcome.PROMPT_TOO_SHORT] == 1
    assert counts[ReuseOutcome.NO_STORED_PREFIX] == 2


def test_plan_helper_still_returns_a_plan_or_none():
    manager = APCManager(num_blocks=8, block_size=BLOCK)

    plan = apc_lookup_plan(
        manager,
        [1],
        extra_hash=0,
        apc_mode="block",
        safe_lookup_min=0,
        suffix_is_text_only=lambda n: True,
        prefix_has_media=lambda n: False,
    )

    assert plan is None
    assert manager.stats_snapshot()["reuse_outcomes"]
