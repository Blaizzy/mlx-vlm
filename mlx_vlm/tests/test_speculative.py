"""Speculative decoding regressions.

This file groups drafter-kind resolution, Gemma 4 assistant mask handling,
and Qwen3.5 DFlash cache rollback coverage in one place.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

import mlx_vlm.models.qwen3_5.language as qwen_language
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.speculative.drafters import (
    DEFAULT_DRAFTER_KIND,
    DRAFTER_KIND_BY_MODEL_TYPE,
    KNOWN_DRAFTER_KINDS,
    resolve_drafter_kind,
)
from mlx_vlm.speculative.drafters.gemma4_assistant.masks import (
    make_drafter_masks,
    normalize_batched_shared_kv_states,
)


def _make_conv_input(batch_size: int, layer_offset: int, length: int = 5) -> mx.array:
    rows = []
    for row in range(batch_size):
        rows.append([[layer_offset * 100 + row * 10 + t] for t in range(length)])
    return mx.array(rows, dtype=mx.float32)


def _make_gdn_state(
    batch_size: int, layer_offset: int, *, init_state: mx.array | None
) -> tuple:
    q = mx.full((batch_size, 3, 3, 4), layer_offset + 0.1, dtype=mx.float32)
    k = mx.full((batch_size, 3, 3, 4), layer_offset + 0.2, dtype=mx.float32)
    v = mx.full((batch_size, 3, 3, 5), layer_offset + 0.3, dtype=mx.float32)
    a = mx.full((batch_size, 3, 3), layer_offset + 0.4, dtype=mx.float32)
    b = mx.full((batch_size, 3, 3), layer_offset + 0.5, dtype=mx.float32)
    A_log = mx.full((3,), layer_offset + 0.6, dtype=mx.float32)
    dt_bias = mx.full((3,), layer_offset + 0.7, dtype=mx.float32)
    conv_input = _make_conv_input(batch_size, layer_offset)
    return (
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        init_state,
        None,
        conv_input,
        4,
    )


def _make_drafter_dir(tmp_path: Path, model_type: str | None) -> Path:
    d = tmp_path / "drafter"
    d.mkdir()
    cfg = {} if model_type is None else {"model_type": model_type}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_qwen_rollback_speculative_cache_flattens_batch_per_layer():
    batch_size = 2
    accepted = mx.array([0, 1], dtype=mx.int32)
    caches = [ArraysCache(size=2), ArraysCache(size=2)]
    state0 = mx.full((batch_size, 3, 5, 4), 10.0, dtype=mx.float32)
    state1 = mx.full((batch_size, 3, 5, 4), 20.0, dtype=mx.float32)
    gdn_states = [
        _make_gdn_state(batch_size, 0, init_state=state0),
        _make_gdn_state(batch_size, 1, init_state=state1),
    ]
    captured = {}

    def fake_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True
    ):
        del k, v, a, b, use_kernel
        captured["q_shape"] = q.shape
        captured["A_log_shape"] = A_log.shape
        captured["dt_bias_shape"] = dt_bias.shape
        captured["mask"] = mask
        row_ids = mx.arange(state.shape[0], dtype=mx.float32).reshape(-1, 1, 1, 1)
        states_out = mx.broadcast_to(row_ids, state.shape)
        return mx.zeros((q.shape[0], q.shape[1], 3, 5), dtype=mx.float32), states_out

    with patch.object(
        qwen_language, "gated_delta_update", side_effect=fake_gated_delta_update
    ):
        max_a = qwen_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), caches, gdn_states, accepted, block_size=3
        )

    assert max_a == 1
    assert captured["q_shape"] == (4, 2, 3, 4)
    assert captured["A_log_shape"] == (4, 1, 3)
    assert captured["dt_bias_shape"] == (4, 1, 3)
    assert captured["mask"].tolist() == [
        [True, False],
        [True, True],
        [True, False],
        [True, True],
    ]
    assert caches[0][1][:, 0, 0, 0].tolist() == [0.0, 1.0]
    assert caches[1][1][:, 0, 0, 0].tolist() == [2.0, 3.0]
    assert caches[0][0][:, :, 0].tolist() == [[1.0, 2.0, 3.0], [12.0, 13.0, 14.0]]
    assert caches[1][0][:, :, 0].tolist() == [
        [101.0, 102.0, 103.0],
        [112.0, 113.0, 114.0],
    ]


def test_qwen_rollback_speculative_cache_zero_inits_missing_state():
    accepted = mx.array([1, 0], dtype=mx.int32)
    caches = [ArraysCache(size=2)]
    gdn_states = [_make_gdn_state(batch_size=2, layer_offset=0, init_state=None)]
    captured = {}

    def fake_gated_delta_update(
        q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True
    ):
        del q, k, v, a, b, A_log, dt_bias, mask, use_kernel
        captured["state"] = state
        return mx.zeros((state.shape[0], 2, 3, 5), dtype=mx.float32), state

    with patch.object(
        qwen_language, "gated_delta_update", side_effect=fake_gated_delta_update
    ):
        qwen_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), caches, gdn_states, accepted, block_size=3
        )

    assert captured["state"].shape == (2, 3, 5, 4)
    assert float(mx.sum(mx.abs(captured["state"])).item()) == 0.0


def test_mtp_drafter_masks_support_batched_offsets():
    kv = (mx.zeros((2, 1, 8, 4)), mx.zeros((2, 1, 8, 4)))

    masks = make_drafter_masks(
        {"sliding_attention": kv, "full_attention": kv},
        query_len=1,
        query_offset=mx.array([5, 8]),
        sliding_window=4,
    )

    assert masks["sliding_attention"].shape == (2, 1, 1, 8)
    assert masks["full_attention"].shape == (2, 1, 1, 8)
    full = masks["full_attention"].tolist()
    assert full[0][0][0][5] == -float("inf")
    assert full[1][0][0][7] == 0.0


def test_normalize_batched_shared_kv_states_repacks_left_padded_rows():
    keys = mx.array(
        [
            [[[0], [0], [0], [10], [11], [12], [13], [14]]],
            [[[20], [21], [22], [23], [24], [25], [26], [27]]],
        ],
        dtype=mx.float32,
    )
    values = keys + 100

    normalized = normalize_batched_shared_kv_states(
        {"full_attention": (keys, values)},
        kv_valid_len=mx.array([5, 8]),
        left_padding=mx.array([3, 0]),
    )

    norm_keys, norm_values = normalized["full_attention"]
    assert norm_keys[:, 0, :, 0].tolist() == [
        [10.0, 11.0, 12.0, 13.0, 14.0, 0.0, 0.0, 0.0],
        [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
    ]
    assert norm_values[:, 0, :, 0].tolist() == [
        [110.0, 111.0, 112.0, 113.0, 114.0, 0.0, 0.0, 0.0],
        [120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0],
    ]


def test_normalize_batched_shared_kv_states_drops_tail_zero_slack():
    keys = mx.array(
        [
            [[[0], [0], [0], [10], [11], [12], [13], [14], [15], [16], [17], [0]]],
            [[[20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]],
        ],
        dtype=mx.float32,
    )
    values = keys + 100

    normalized = normalize_batched_shared_kv_states(
        {"sliding_attention": (keys, values)},
        kv_valid_len=mx.array([8, 12]),
        left_padding=mx.array([3, 0]),
    )

    norm_keys, norm_values = normalized["sliding_attention"]
    assert norm_keys[:, 0, :, 0].tolist() == [
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 0.0, 0.0, 0.0, 0.0],
        [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0],
    ]
    assert norm_values[:, 0, :, 0].tolist() == [
        [110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 0.0, 0.0, 0.0, 0.0],
        [
            120.0,
            121.0,
            122.0,
            123.0,
            124.0,
            125.0,
            126.0,
            127.0,
            128.0,
            129.0,
            130.0,
            131.0,
        ],
    ]


def test_gemma4_assistant_overrides_dflash_to_mtp(tmp_path, caplog):
    path = _make_drafter_dir(tmp_path, "gemma4_assistant")
    with caplog.at_level("WARNING"):
        resolved = resolve_drafter_kind(path, "dflash")
    assert resolved == "mtp"
    assert any("requires --draft-kind='mtp'" in r.getMessage() for r in caplog.records)


def test_explicit_mtp_kept_for_gemma4_assistant(tmp_path, caplog):
    path = _make_drafter_dir(tmp_path, "gemma4_assistant")
    with caplog.at_level("WARNING"):
        resolved = resolve_drafter_kind(path, "mtp")
    assert resolved == "mtp"
    assert caplog.records == []


def test_unknown_model_type_keeps_caller_kind(tmp_path):
    path = _make_drafter_dir(tmp_path, "qwen3_dflash")
    assert resolve_drafter_kind(path, "dflash") == "dflash"


def test_missing_config_keeps_caller_kind(tmp_path):
    d = tmp_path / "drafter"
    d.mkdir()
    assert resolve_drafter_kind(d, "dflash") == "dflash"


def test_malformed_config_keeps_caller_kind(tmp_path):
    d = tmp_path / "drafter"
    d.mkdir()
    (d / "config.json").write_text("not valid json {")
    assert resolve_drafter_kind(d, "dflash") == "dflash"


def test_kind_table_only_uses_known_kinds():
    for mt, kind in DRAFTER_KIND_BY_MODEL_TYPE.items():
        assert kind in KNOWN_DRAFTER_KINDS, f"{mt} maps to unknown kind {kind}"


@pytest.mark.parametrize("model_type", [None])
def test_resolver_handles_no_model_type_field(tmp_path, model_type):
    path = _make_drafter_dir(tmp_path, model_type)
    assert resolve_drafter_kind(path, "dflash") == "dflash"


def test_kind_none_autodetects_mtp_for_gemma4_assistant(tmp_path):
    path = _make_drafter_dir(tmp_path, "gemma4_assistant")
    assert resolve_drafter_kind(path, None) == "mtp"
    assert resolve_drafter_kind(path) == "mtp"


def test_kind_none_falls_back_to_default_for_unknown_model_type(tmp_path):
    path = _make_drafter_dir(tmp_path, "qwen3_dflash")
    assert resolve_drafter_kind(path, None) == DEFAULT_DRAFTER_KIND


def test_kind_none_falls_back_to_default_for_missing_config(tmp_path):
    d = tmp_path / "drafter"
    d.mkdir()
    assert resolve_drafter_kind(d, None) == DEFAULT_DRAFTER_KIND
