"""Speculative decoding regressions.

This file groups drafter-kind resolution, Gemma 4 assistant mask handling,
and Qwen3.5 DFlash cache rollback coverage in one place.
"""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.qwen3_5.language as qwen_language
from mlx_vlm.generate import (
    _effective_mtp_block_size,
    _format_speculative_stats,
    _mtp_draft_block_active,
    _mtp_draft_hidden,
    _mtp_next_block_size,
    _mtp_rounds,
    _mtp_shared_kv_from_prompt_cache,
    _mtp_verify_target,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_deferred_greedy,
    _speculative_walk_deferred_greedy,
)
from mlx_vlm.models.cache import ArraysCache, BufferedRotatingKVCache, RotatingKVCache
from mlx_vlm.speculative.drafters import (
    DEFAULT_DRAFTER_KIND,
    DRAFTER_KIND_BY_MODEL_TYPE,
    KNOWN_DRAFTER_KINDS,
    resolve_drafter_kind,
)
from mlx_vlm.speculative.drafters.gemma4_assistant.masked_embedder import MaskedEmbedder
from mlx_vlm.speculative.drafters.gemma4_assistant.masks import (
    make_drafter_masks,
    normalize_batched_shared_kv_states,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp

generate_module = importlib.import_module("mlx_vlm.generate")


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

    def fake_gated_delta_state_update(
        k, v, a, b, A_log, dt_bias, state, steps, mask, use_kernel=True
    ):
        del v, a, b, use_kernel
        captured["k_shape"] = k.shape
        captured["A_log_shape"] = A_log.shape
        captured["dt_bias_shape"] = dt_bias.shape
        captured["steps"] = steps
        captured["mask"] = mask
        row_ids = mx.arange(state.shape[0], dtype=mx.float32).reshape(-1, 1, 1, 1)
        return mx.broadcast_to(row_ids, state.shape)

    with patch.object(
        qwen_language,
        "gated_delta_state_update",
        side_effect=fake_gated_delta_state_update,
    ):
        max_a = qwen_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), caches, gdn_states, accepted, block_size=3
        )

    assert max_a == 1
    assert captured["k_shape"] == (4, 2, 3, 4)
    assert captured["A_log_shape"] == (4, 1, 3)
    assert captured["dt_bias_shape"] == (4, 1, 3)
    assert captured["steps"].tolist() == [1, 2, 1, 2]
    assert captured["mask"] is None
    assert caches[0][1][:, 0, 0, 0].tolist() == [0.0, 1.0]
    assert caches[1][1][:, 0, 0, 0].tolist() == [2.0, 3.0]
    assert caches[0][0][:, :, 0].tolist() == [[1.0, 2.0, 3.0], [12.0, 13.0, 14.0]]
    assert caches[1][0][:, :, 0].tolist() == [
        [101.0, 102.0, 103.0],
        [112.0, 113.0, 114.0],
    ]


def test_qwen_rollback_speculative_cache_uses_intermediate_states():
    batch_size = 2
    accepted = mx.array([0, 1], dtype=mx.int32)
    caches = [ArraysCache(size=2)]
    state = mx.arange(batch_size * 3 * 3 * 5 * 4, dtype=mx.float32).reshape(
        batch_size, 3, 3, 5, 4
    )
    gdn_states = [_make_gdn_state(batch_size, 0, init_state=None) + (state,)]

    with patch.object(
        qwen_language,
        "gated_delta_state_update",
        side_effect=AssertionError("state replay should not run"),
    ):
        max_a = qwen_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), caches, gdn_states, accepted, block_size=3
        )

    assert max_a == 1
    expected_state = mx.stack([state[0, 0], state[1, 1]])
    assert caches[0][1].tolist() == expected_state.tolist()
    assert caches[0][0][:, :, 0].tolist() == [
        [1.0, 2.0, 3.0],
        [12.0, 13.0, 14.0],
    ]


def test_qwen_rollback_speculative_cache_zero_inits_missing_state():
    accepted = mx.array([1, 0], dtype=mx.int32)
    caches = [ArraysCache(size=2)]
    gdn_states = [_make_gdn_state(batch_size=2, layer_offset=0, init_state=None)]
    captured = {}

    def fake_gated_delta_state_update(
        k, v, a, b, A_log, dt_bias, state, steps, mask, use_kernel=True
    ):
        del k, v, a, b, A_log, dt_bias, steps, mask, use_kernel
        captured["state"] = state
        return state

    with patch.object(
        qwen_language,
        "gated_delta_state_update",
        side_effect=fake_gated_delta_state_update,
    ):
        qwen_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), caches, gdn_states, accepted, block_size=3
        )

    assert captured["state"].shape == (2, 3, 5, 4)
    assert float(mx.sum(mx.abs(captured["state"])).item()) == 0.0


def test_speculative_walk_accepts_until_first_mismatch():
    accepted, new_tokens = _speculative_walk(
        mx.array([[11, 12, 13]], dtype=mx.int32),
        mx.array([[11, 99, 77, 55]], dtype=mx.int32),
        budget=4,
    )

    assert accepted == 1
    assert new_tokens == [11, 99]


def test_speculative_walk_uses_target_bonus_when_all_drafts_match():
    accepted, new_tokens = _speculative_walk(
        mx.array([[11, 12]], dtype=mx.int32),
        mx.array([[11, 12, 42]], dtype=mx.int32),
        budget=3,
    )

    assert accepted == 2
    assert new_tokens == [11, 12, 42]


def test_speculative_walk_batch_matches_per_row_acceptance():
    accepted, new_tokens = _speculative_walk_batch(
        mx.array([[11, 12], [21, 22]], dtype=mx.int32),
        mx.array([[11, 90, 91], [21, 22, 23]], dtype=mx.int32),
        budgets=[3, 2],
    )

    assert accepted == [1, 2]
    assert new_tokens == [[11, 90], [21, 22]]


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


def test_mtp_drafter_sliding_mask_accounts_for_tail_offset():
    kv = (mx.zeros((1, 1, 6, 4)), mx.zeros((1, 1, 6, 4)))

    masks = make_drafter_masks(
        {"sliding_attention": kv},
        query_len=1,
        query_offset=10,
        sliding_window=4,
    )

    row = masks["sliding_attention"][0, 0, 0].tolist()
    assert row[:3] == [-float("inf"), -float("inf"), -float("inf")]
    assert row[3:] == [0.0, 0.0, 0.0]


def test_mtp_drafter_sliding_mask_accepts_single_row_array_offsets():
    kv = (mx.zeros((1, 1, 6, 4)), mx.zeros((1, 1, 6, 4)))

    masks = make_drafter_masks(
        {"sliding_attention": kv},
        query_len=1,
        query_offset=10,
        sliding_window=4,
        kv_valid_len=mx.array([10]),
    )

    row = masks["sliding_attention"][0, 0, 0].tolist()
    assert row[:3] == [-float("inf"), -float("inf"), -float("inf")]
    assert row[3:] == [0.0, 0.0, 0.0]


def test_buffered_rotating_cache_matches_temporal_multitoken_tail_and_trim():
    base = RotatingKVCache(max_size=4, keep=0)
    keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    values = keys + 10
    base.update_and_fetch(keys, values)

    buffered = BufferedRotatingKVCache.from_cache(base, buffer_size=2)
    new_keys = mx.array([[[[4.0], [5.0], [6.0]]]])
    new_values = new_keys + 10
    out_keys, _ = buffered.update_and_fetch(new_keys, new_values)

    assert buffered.start_position == 0
    assert buffered.offset == 7
    assert out_keys.reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    assert buffered.trim(2) == 2
    assert buffered.offset == 5
    assert buffered.state[0].reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_mtp_drafter_sliding_mask_uses_local_rotating_cache_offset():
    kv = (mx.zeros((1, 1, 8, 4)), mx.zeros((1, 1, 8, 4)))

    masks = make_drafter_masks(
        {"sliding_attention": kv},
        query_len=1,
        query_offset=128,
        sliding_window=4,
    )

    mask = masks["sliding_attention"].tolist()[0][0][0]
    assert mask[:5] == [-float("inf")] * 5
    assert mask[5:] == [0.0, 0.0, 0.0]


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


def test_speculative_walk_mtp_deferred_greedy_stops_after_first_mismatch():
    class FakeEmbed:
        def __init__(self):
            self.calls = 0

        def as_linear(self, hidden):
            self.calls += 1
            return hidden

    fake_head = FakeEmbed()
    lm = SimpleNamespace(speculative_logits_from_hidden=fake_head.as_linear)
    target_hidden = mx.array(
        [
            [
                [0, 0, 9, 0],
                [0, 0, 0, 9],
                [0, 9, 0, 0],
                [9, 0, 0, 0],
            ]
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[2, 1, 3]], dtype=mx.int32)
    accepted, new_tokens = _speculative_walk_deferred_greedy(
        lm,
        target_hidden,
        draft_tokens,
        lambda logits: mx.argmax(logits, axis=-1),
        budget=4,
    )

    expected_accepted, expected_tokens = _speculative_walk(
        draft_tokens, mx.argmax(target_hidden, axis=-1), budget=4
    )
    assert accepted == expected_accepted
    assert new_tokens == expected_tokens
    assert accepted == 1
    assert new_tokens == [2, 3]
    assert fake_head.calls == 2


def test_speculative_walk_batch_deferred_greedy_matches_batch_walk():
    class FakeEmbed:
        def __init__(self):
            self.calls = 0

        def as_linear(self, hidden):
            self.calls += 1
            return hidden

    fake_head = FakeEmbed()
    lm = SimpleNamespace(speculative_logits_from_hidden=fake_head.as_linear)
    target_hidden = mx.array(
        [
            [
                [0, 0, 9, 0],
                [0, 9, 0, 0],
                [0, 0, 0, 9],
            ],
            [
                [9, 0, 0, 0],
                [0, 0, 9, 0],
                [0, 9, 0, 0],
            ],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[2, 3], [0, 2]], dtype=mx.int32)

    accepted, new_tokens = _speculative_walk_batch_deferred_greedy(
        lm,
        target_hidden,
        draft_tokens,
        lambda logits: mx.argmax(logits, axis=-1),
        budgets=[3, 2],
    )

    expected_accepted, expected_tokens = _speculative_walk_batch(
        draft_tokens,
        mx.argmax(target_hidden, axis=-1),
        budgets=[3, 2],
    )
    assert accepted == expected_accepted
    assert new_tokens == expected_tokens
    assert accepted == [1, 2]
    assert new_tokens == [[2, 1], [0, 2]]
    assert fake_head.calls == 3


def test_mtp_draft_hidden_uses_model_hook():
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    lm = SimpleNamespace(speculative_draft_hidden=lambda h: h * 2)

    assert _mtp_draft_hidden(lm, hidden).tolist() == [[[2.0, 4.0]]]


def test_mtp_draft_hidden_defaults_to_identity():
    hidden = mx.array([[[1.0, 2.0]]], dtype=mx.float32)

    assert _mtp_draft_hidden(SimpleNamespace(), hidden).tolist() == hidden.tolist()


def test_mtp_verify_target_uses_model_logits_hook():
    verify_input = mx.array([[7, 8]], dtype=mx.int32)
    hidden = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
    target_tokens = mx.array([[3, 4]], dtype=mx.int32)
    calls = []

    def verify_logits(inputs, cache, sampler):
        calls.append((inputs, cache, sampler))
        return hidden, {"full": ("k", "v")}, ["gdn"], target_tokens

    lm = SimpleNamespace(
        speculative_verify_logits=verify_logits,
        speculative_logits_from_hidden=lambda _: (_ for _ in ()).throw(
            AssertionError("deferred logits should not be used")
        ),
    )

    result = _mtp_verify_target(
        lm,
        verify_input,
        prompt_cache=["cache"],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )

    assert calls[0][0] is verify_input
    assert calls[0][1] == ["cache"]
    assert result.hidden is hidden
    assert result.shared_kv_states == {"full": ("k", "v")}
    assert result.gdn_states == ["gdn"]
    assert result.target_tokens is target_tokens


def test_mtp_rounds_rolls_back_gemma_without_gdn_states():
    class Draft:
        def __init__(self):
            self.config = SimpleNamespace(block_size=3)
            self.accept_lens = []
            self.draft_lens = []

        def set_shared_kv(self, *args, **kwargs):
            pass

        def reset(self, model):
            pass

        def draft_block(self, *args, **kwargs):
            return mx.array([[7, 8]], dtype=mx.int32)

    rollback_calls = []

    class LM:
        def rollback_speculative_cache(self, *args):
            rollback_calls.append(args)

    lm = LM()
    model = SimpleNamespace(language_model=lm)
    verify = generate_module._MTPVerifyResult(
        hidden=mx.zeros((1, 3, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=None,
    )

    with (
        patch.object(generate_module, "_mtp_verify_target", return_value=verify),
        patch.object(
            generate_module,
            "_mtp_acceptance_walk",
            return_value=(0, [9]),
        ),
    ):
        list(
            _mtp_rounds(
                model,
                Draft(),
                [SimpleNamespace(offset=0)],
                mx.zeros((1, 1, 2), dtype=mx.float32),
                {},
                first_bonus=1,
                max_tokens=3,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                token_dtype=mx.int32,
                greedy_sampling=True,
            )
        )

    assert rollback_calls
    assert rollback_calls[0][1] is None


def test_mtp_next_block_size_can_prefer_requested_size():
    draft_model = SimpleNamespace(
        accept_lens=[0] * 16, prefer_requested_block_size=True
    )

    assert _mtp_next_block_size(draft_model, 6, 3, 5) == 5


def test_mtp_shared_kv_accepts_cache_state_metadata():
    keys = mx.ones((1, 1, 2, 2), dtype=mx.float32)
    values = keys + 1
    layer = SimpleNamespace(layer_type="full_attention")
    layer_cache = SimpleNamespace(state=(keys, values, "metadata"))
    lm = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    shared = _mtp_shared_kv_from_prompt_cache(lm, [layer_cache])

    assert shared["full_attention"][0] is keys
    assert shared["full_attention"][1] is values


def test_masked_embedder_argmax_matches_full_sparse_logits():
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2, vocab_size=8),
        num_centroids=2,
        centroid_intermediate_top_k=1,
    )
    embedder = MaskedEmbedder(cfg)
    embedder.centroids.weight = mx.array([[1.0, 0.0], [0.0, 1.0]])
    embedder.token_ordering = mx.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=mx.int32)
    lm_head_weight = mx.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [4.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=mx.float32,
    )
    hidden = mx.array([[[2.0, 0.5], [0.5, 2.0]]], dtype=mx.float32)

    fast = embedder.argmax(hidden, lm_head_weight)
    full = mx.argmax(embedder(hidden, lm_head_weight), axis=-1)

    assert fast.tolist() == full.tolist()


def test_format_speculative_stats_includes_variable_draft_rate():
    stats = _format_speculative_stats(
        SimpleNamespace(accept_lens=[1, 0, 2], draft_lens=[2, 1, 2])
    )

    assert (
        stats == "Speculative decoding: 1.00 accepted tokens/round "
        "(60.0% of drafted, avg draft 1.67) over 3 rounds"
    )


def test_effective_mtp_block_size_respects_requested_block_size():
    assert _effective_mtp_block_size(4, 4, [], 10) == 4
    assert _effective_mtp_block_size(3, 4, [], 10) == 3


def test_effective_mtp_block_size_treats_oversized_block_as_ceiling():
    assert _effective_mtp_block_size(8, 4, [], 10) == 4
    assert _effective_mtp_block_size(8, 4, [1, 1, 0, 1], 10) == 4


def test_effective_mtp_block_size_expands_when_prefix_is_reliable():
    accept_lens = [3, 3, 3, 2, 3, 3, 3, 3]
    assert _effective_mtp_block_size(8, 4, accept_lens, 10) == 8


def test_effective_mtp_block_size_caps_unreliable_oversized_block():
    accept_lens = [3, 0, 1, 2, 3, 0, 1, 2]
    assert _effective_mtp_block_size(8, 4, accept_lens, 10) == 4


def test_effective_mtp_block_size_caps_to_remaining_budget():
    assert _effective_mtp_block_size(8, 4, [], 3) == 3
    assert _effective_mtp_block_size(6, 4, [2, 2, 2, 2], 1) == 1


def test_mtp_draft_block_active_uses_per_row_shared_kv_for_mixed_positions():
    class FakeDraftModel:
        def __init__(self):
            self._shared_kv = None
            self._draft_round = 4
            self.calls = []
            self.rounds = []

        def set_shared_kv(
            self,
            shared_kv_states,
            kv_offset,
            position=None,
            kv_valid_len=None,
            left_padding=None,
        ):
            del left_padding
            self.calls.append((kv_offset, position, kv_valid_len))
            self._shared_kv = shared_kv_states

        def draft_block(
            self,
            last_bonus,
            hidden,
            cache,
            block_size,
            sampler,
            token_dtype,
        ):
            batch_size = hidden.shape[0]
            del cache, sampler
            self.rounds.append(self._draft_round)
            self._draft_round += 1
            base = int(next(iter(self._shared_kv.values()))[0][0, 0, 0, 0].item())
            bonus = (
                last_bonus if isinstance(last_bonus, int) else int(last_bonus[0].item())
            )
            row_count = 1 if isinstance(last_bonus, int) else batch_size
            return mx.full((row_count, block_size - 1), base + bonus, dtype=token_dtype)

    draft_model = FakeDraftModel()
    shared_kv = {
        "full_attention": (
            mx.array([[[[10.0]]], [[[20.0]]]], dtype=mx.float32),
            mx.zeros((2, 1, 1, 1), dtype=mx.float32),
        )
    }
    draft_model.set_shared_kv(shared_kv, kv_offset=11, position=mx.array([11, 12]))
    draft_model.calls = []

    drafted = _mtp_draft_block_active(
        draft_model,
        bonus_tokens=[3, 7],
        hidden=mx.zeros((2, 1, 1), dtype=mx.float32),
        block_size=2,
        sampler=lambda x: x,
        token_dtype=mx.int32,
        positions=[11, 12],
    )

    assert drafted.tolist() == [[13], [27]]
    assert draft_model.rounds == [4, 4]
    assert draft_model._draft_round == 5
    assert next(iter(draft_model._shared_kv.values()))[0].shape[0] == 2
    assert draft_model.calls[0] == (11, 10, 11)
    assert draft_model.calls[1] == (12, 11, 12)
    restored_kv_offset, restored_position, restored_valid_len = draft_model.calls[2]
    assert restored_kv_offset == 12
    assert restored_position.tolist() == [10, 11]
    assert restored_valid_len.tolist() == [11, 12]


def test_mtp_draft_block_active_uses_batched_path_for_aligned_positions():
    class FakeDraftModel:
        def __init__(self):
            self.calls = []

        def draft_block(
            self,
            last_bonus,
            hidden,
            cache,
            block_size,
            sampler,
            token_dtype,
        ):
            del cache, sampler
            self.calls.append((last_bonus.shape, hidden.shape))
            return last_bonus[:, None].astype(token_dtype) + mx.arange(block_size - 1)

    draft_model = FakeDraftModel()

    drafted = _mtp_draft_block_active(
        draft_model,
        bonus_tokens=[3, 7],
        hidden=mx.zeros((2, 1, 1), dtype=mx.float32),
        block_size=3,
        sampler=lambda x: x,
        token_dtype=mx.int32,
        positions=[11, 11],
    )

    assert drafted.tolist() == [[3, 4], [7, 8]]
    assert draft_model.calls == [((2,), (2, 1, 1))]


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


def test_kind_none_autodetects_mtp_for_qwen3_5_mtp(tmp_path):
    path = _make_drafter_dir(tmp_path, "qwen3_5_mtp")
    assert resolve_drafter_kind(path, None) == "mtp"
    assert resolve_drafter_kind(path, "dflash") == "mtp"


def test_kind_none_falls_back_to_default_for_unknown_model_type(tmp_path):
    path = _make_drafter_dir(tmp_path, "qwen3_dflash")
    assert resolve_drafter_kind(path, None) == DEFAULT_DRAFTER_KIND


def test_kind_none_falls_back_to_default_for_missing_config(tmp_path):
    d = tmp_path / "drafter"
    d.mkdir()
    assert resolve_drafter_kind(d, None) == DEFAULT_DRAFTER_KIND


def _tiny_qwen3_5_text_config():
    return qwen_language.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=16,
        intermediate_size=32,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        head_dim=8,
        full_attention_interval=1,
        rope_parameters={
            "type": "default",
            "mrope_section": [1, 0, 0],
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )


def test_qwen3_5_mtp_draft_block_smoke():
    text_config = _tiny_qwen3_5_text_config()
    text_config.mtp_num_hidden_layers = 1
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config=text_config, block_size=3)
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=nn.Embedding(32, 16))
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    hidden = mx.zeros((1, 1, 16), dtype=mx.float32)
    tokens = drafter.draft_block(
        7,
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
    )
    mx.eval(tokens)
    assert tokens.shape == (1, 2)


def test_qwen3_5_mtp_advances_draft_cache_positions():
    text_config = _tiny_qwen3_5_text_config()
    text_config.mtp_num_hidden_layers = 1
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(text_config=text_config, block_size=3)
    )
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)

    assert drafter._position_ids(0).tolist() == [[4]]
    assert drafter._position_ids(1).tolist() == [[5]]


def test_qwen3_5_mtp_sanitize_strips_prefix_and_offsets_norms():
    weights = {
        "mtp.fc.weight": mx.ones((2, 4)),
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((2,)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((2,)),
    }
    out = Qwen3_5MTPDraftModel.sanitize(None, weights)
    assert "fc.weight" in out
    assert out["pre_fc_norm_hidden.weight"].tolist() == [1.0, 1.0]
    assert out["layers.0.self_attn.q_norm.weight"].tolist() == [1.0, 1.0]


def test_split_qwen3_5_mtp_writes_sidecar_without_index_mtp_entries(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_qwen3_5_text_config()
    text_config.mtp_num_hidden_layers = 1
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": text_config.to_dict()})
    )
    mx.save_safetensors(
        str(source / "mtp.safetensors"),
        {
            "mtp.fc.weight": mx.ones((16, 32)),
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((16,)),
        },
        metadata={},
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.foo": "model.safetensors"}})
    )

    split_qwen3_5_mtp(str(source), str(output))

    with open(output / "config.json") as f:
        cfg = json.load(f)
    weights = mx.load(str(output / "model.safetensors"))
    assert cfg["model_type"] == "qwen3_5_mtp"
    assert cfg["block_size"] == 3
    assert "fc.weight" in weights
    assert weights["pre_fc_norm_hidden.weight"][0].item() == 1.0
