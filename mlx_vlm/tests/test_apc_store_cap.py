"""Unit tests for APC_EXACT_MAX_PROMPT_TOKENS (exact-store size cap).

The cap changes only STORE eligibility in
BatchGenerator._store_apc_exact_checkpoints: rows whose checkpoint length
exceeds it skip the store and are marked done (so they are not re-examined
every step). Lookups, under-cap rows, and the unset/0 default are untouched.

Tested against a stub batch object (the method's dependencies are plain
attributes/callables), so no model instantiation is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx_vlm.generate.ar import (
    PromptProcessingBatch,
    _get_apc_exact_max_prompt_tokens,
)


class _RecordingManager:
    def __init__(self):
        self.stored = []

    def store_exact_cache(self, input_ids, prompt_cache, extra_hash=0):
        self.stored.append((tuple(input_ids), extra_hash))
        return True


def _stub_generator(metas):
    manager = _RecordingManager()
    gen = SimpleNamespace(
        _apc_manager=manager,
        _apc_mode="exact",
        _apc_meta=metas,
        _row_real_tokens_processed=lambda batch_idx: int(
            metas[batch_idx].get("checkpoint_len") or 0
        ),
        _apc_prompt_cache_for_store=lambda batch_idx: ["cache"],
    )
    return gen, manager


def _meta(n):
    return {
        "checkpoint_len": n,
        "full_input_ids": list(range(n)),
        "extra_hash": 0,
    }


def test_env_helper_parses_and_defaults(monkeypatch):
    monkeypatch.delenv("APC_EXACT_MAX_PROMPT_TOKENS", raising=False)
    assert _get_apc_exact_max_prompt_tokens() == 0
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "32768")
    assert _get_apc_exact_max_prompt_tokens() == 32768
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "-5")
    assert _get_apc_exact_max_prompt_tokens() == 0
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "not-a-number")
    assert _get_apc_exact_max_prompt_tokens() == 0


def test_over_cap_row_skips_store_and_marks_done(monkeypatch):
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "100")
    gen, manager = _stub_generator([_meta(101)])
    PromptProcessingBatch._store_apc_exact_checkpoints(gen)
    assert manager.stored == []
    # marked done: not re-examined on subsequent steps
    assert gen._apc_meta[0]["checkpoint_done"] is True
    PromptProcessingBatch._store_apc_exact_checkpoints(gen)
    assert manager.stored == []


def test_under_cap_row_stores(monkeypatch):
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "100")
    gen, manager = _stub_generator([_meta(100)])
    PromptProcessingBatch._store_apc_exact_checkpoints(gen)
    assert len(manager.stored) == 1
    assert gen._apc_meta[0]["checkpoint_done"] is True


@pytest.mark.parametrize("raw", [None, "0"])
def test_unset_or_zero_is_stock_behavior(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("APC_EXACT_MAX_PROMPT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", raw)
    gen, manager = _stub_generator([_meta(1_000_000)])
    PromptProcessingBatch._store_apc_exact_checkpoints(gen)
    assert len(manager.stored) == 1


def test_mixed_batch_caps_only_over_cap_rows(monkeypatch):
    monkeypatch.setenv("APC_EXACT_MAX_PROMPT_TOKENS", "100")
    gen, manager = _stub_generator([_meta(50), _meta(500), _meta(99)])
    PromptProcessingBatch._store_apc_exact_checkpoints(gen)
    assert len(manager.stored) == 2
    assert gen._apc_meta[1]["checkpoint_done"] is True
