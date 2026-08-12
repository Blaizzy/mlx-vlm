"""Model-free tests for the DFlash warm-hit drafter-context fix.

After an APC warm hit the drafter used to receive a SINGLE hidden-state
position (each row's last real position). ``assemble_warm_drafter_context``
now hands it each row's FULL un-padded suffix hidden states plus per-row
absolute RoPE offsets (``context_offsets[i] = prefix_lens[i]``), and
``_apply_round1_window`` enforces the drafter's training-time sliding window
on the round-1 context while keeping RoPE positions absolute via the additive
per-cache ``pos_shift``.

Covered behavior (all pure array math, no model load, no downloads):
  * default path        -- ragged rows returned un-padded with correct
                           per-row lengths; context_offsets == prefix_lens.
  * MUSE_WARMCTX=0      -- reproduces the pre-fix single-position
                           take_along_axis slice byte-identically.
  * _apply_round1_window -- keeps the LAST ``keep`` positions and bumps
                           pos_shift by the number skipped; no-op (same
                           object, no shift) when already within the window;
                           composes additively with a warm-hit offset.
  * _round1_window_limit -- window - 1; None for windowless configs; None
                           under the MUSE_ROUND1_TRUNC=0 kill switch.
  * _env_flag           -- call-time parse semantics backing both hatches.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_vlm.speculative.dflash import (
    _apply_round1_window,
    _drafter_pos_shift,
    _env_flag,
    _round1_window_limit,
    assemble_warm_drafter_context,
)


# ============================================================================
# Helpers
# ============================================================================


def _hs(B, S, D):
    """Hidden states with a distinct value at every (row, position, dim) so
    any slicing/indexing mistake changes the result."""
    return mx.arange(B * S * D, dtype=mx.float32).reshape(B, S, D)


class _CacheStub:
    """Drafter layer-cache stand-in for _drafter_pos_shift's contract: any
    object accepting attribute assignment. ``pos_shift`` is deliberately NOT
    pre-declared -- the real caches also grow it on first bump (getattr
    default 0)."""


class _DrafterStub:
    def __init__(self, sliding_window=None):
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.sliding_window = sliding_window


# ============================================================================
# assemble_warm_drafter_context
# ============================================================================


def test_warm_context_default_returns_ragged_unpadded_rows():
    B, S, D = 3, 7, 4
    hs = _hs(B, S, D)
    right_pad = [0, 3, 5]
    prefix_lens = [1000, 250, 0]

    rows, offsets = assemble_warm_drafter_context(hs, right_pad, prefix_lens)

    assert offsets == prefix_lens
    assert isinstance(rows, list) and len(rows) == B
    for i in range(B):
        expect = hs[i : i + 1, : S - right_pad[i], :]
        assert rows[i].shape == (1, S - right_pad[i], D)
        assert mx.array_equal(rows[i], expect).item()


def test_warm_context_offsets_are_plain_ints():
    hs = _hs(2, 4, 3)
    _, offsets = assemble_warm_drafter_context(hs, [0, 1], [mx.array(17), 5])
    assert offsets == [17, 5]
    assert all(type(o) is int for o in offsets)


def test_warm_context_kill_switch_matches_single_position_slice(monkeypatch):
    monkeypatch.setenv("MUSE_WARMCTX", "0")
    B, S, D = 3, 6, 5
    hs = _hs(B, S, D)
    right_pad = [0, 2, 4]  # non-uniform: per-row last real position differs
    prefix_lens = [10, 20, 30]

    hidden, offsets = assemble_warm_drafter_context(hs, right_pad, prefix_lens)

    assert offsets is None
    # The exact pre-fix computation from server/generation.py.
    li = mx.array([S - 1 - rp for rp in right_pad], dtype=mx.int32)
    idx = mx.broadcast_to(li[:, None, None], (B, 1, D))
    expect = mx.take_along_axis(hs, idx, axis=1)
    assert hidden.shape == (B, 1, D)
    assert mx.array_equal(hidden, expect).item()


# ============================================================================
# _apply_round1_window / _drafter_pos_shift
# ============================================================================


def test_apply_round1_window_keeps_tail_and_bumps_pos_shift():
    S, keep = 9, 4
    hidden = _hs(1, S, 3)
    caches = [_CacheStub(), _CacheStub()]

    out = _apply_round1_window(hidden, caches, keep)

    assert out.shape == (1, keep, 3)
    assert mx.array_equal(out, hidden[:, -keep:, :]).item()
    assert all(c.pos_shift == S - keep for c in caches)


def test_apply_round1_window_noop_within_window():
    hidden = _hs(1, 3, 2)
    caches = [_CacheStub()]

    out = _apply_round1_window(hidden, caches, 4)

    assert out is hidden  # untouched, not a copy
    assert getattr(caches[0], "pos_shift", 0) == 0


def test_pos_shift_composes_warm_offset_then_truncation():
    # A warm suffix longer than the window: the APC offset lands first, the
    # round-1 truncation adds the skipped count on top (additive shifts).
    caches = [_CacheStub()]
    _drafter_pos_shift(caches, 100)  # APC warm hit: cached_tokens
    hidden = _hs(1, 8, 2)
    out = _apply_round1_window(hidden, caches, 5)  # skips 3
    assert out.shape == (1, 5, 2)
    assert caches[0].pos_shift == 103


# ============================================================================
# _round1_window_limit
# ============================================================================


def test_round1_window_limit_is_window_minus_one():
    assert _round1_window_limit(_DrafterStub(2048)) == 2047


def test_round1_window_limit_none_for_windowless_configs():
    assert _round1_window_limit(_DrafterStub(None)) is None
    assert _round1_window_limit(_DrafterStub(0)) is None
    assert _round1_window_limit(_DrafterStub(1)) is None


def test_round1_window_limit_kill_switch(monkeypatch):
    monkeypatch.setenv("MUSE_ROUND1_TRUNC", "0")
    assert _round1_window_limit(_DrafterStub(2048)) is None


# ============================================================================
# _env_flag (call-time hatch semantics)
# ============================================================================


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", " OFF "])
def test_env_flag_falsy_spellings(monkeypatch, raw):
    monkeypatch.setenv("MUSE_TEST_FLAG", raw)
    assert _env_flag("MUSE_TEST_FLAG", True) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
def test_env_flag_truthy_spellings(monkeypatch, raw):
    monkeypatch.setenv("MUSE_TEST_FLAG", raw)
    assert _env_flag("MUSE_TEST_FLAG", False) is True


def test_env_flag_unset_returns_default(monkeypatch):
    monkeypatch.delenv("MUSE_TEST_FLAG", raising=False)
    assert _env_flag("MUSE_TEST_FLAG", True) is True
    assert _env_flag("MUSE_TEST_FLAG", False) is False


# ============================================================================
# Server-rounds dispatch (B==1)
# ============================================================================


def test_warm_hit_batch_size_one_dispatches_batch_rounds(monkeypatch):
    """A B==1 APC warm hit (ragged hidden + context_offsets) must route
    through _dflash_rounds_batch — the only loop that consumes the offsets
    and per-row ragged hidden — while cold B==1 keeps the singleton fast
    path (covered by test_speculative's dispatch test)."""
    from types import SimpleNamespace

    from mlx_vlm.speculative import utils as speculative_utils

    calls = []

    def fake_batch(*args, **kwargs):
        calls.append((args, kwargs))
        yield [3], None

    def fake_single(*args, **kwargs):
        raise AssertionError("warm B==1 must not take the singleton path")

    monkeypatch.setattr(speculative_utils, "_dflash_rounds", fake_single)
    monkeypatch.setattr(speculative_utils, "_dflash_rounds_batch", fake_batch)

    result = list(
        speculative_utils.run_speculative_server_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            SimpleNamespace(),
            prompt_cache=[],
            hidden=[mx.zeros((1, 5, 1), dtype=mx.float32)],  # ragged, one row
            draft_kind="dflash",
            first_bonus=mx.array([2], dtype=mx.int32),
            max_tokens=4,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            token_dtype=mx.int32,
            context_offsets=[1000],
        )
    )

    assert result == [([3], None)]
    assert calls and calls[0][1]["context_offsets"] == [1000]
