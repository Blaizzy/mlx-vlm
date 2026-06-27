"""Speculative decoding regressions.

This file groups drafter-kind resolution, Gemma 4 assistant mask handling,
and Qwen3.5 DFlash cache rollback coverage in one place.
"""

import importlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten, tree_map

import mlx_vlm.models.deepseek_v4.language as deepseek_language
import mlx_vlm.models.gemma4.language as gemma4_language
import mlx_vlm.models.qwen3_5.language as qwen_language
import mlx_vlm.speculative.mtp as mtp_utils
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BufferedRotatingKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    RotatingKVCache,
)
from mlx_vlm.speculative.common import _SpeculativeSamplerRNG
from mlx_vlm.speculative.drafters import (
    DEFAULT_DRAFTER_KIND,
    DRAFTER_KIND_BY_MODEL_TYPE,
    KNOWN_DRAFTER_KINDS,
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.deepseek_v4_mtp import DeepseekV4MTPDraftModel
from mlx_vlm.speculative.drafters.deepseek_v4_mtp.config import DeepseekV4MTPConfig
from mlx_vlm.speculative.drafters.deepseek_v4_mtp.split import split_deepseek_v4_mtp
from mlx_vlm.speculative.drafters.eagle3 import Eagle3DraftModel
from mlx_vlm.speculative.drafters.eagle3 import ModelConfig as Eagle3Config
from mlx_vlm.speculative.drafters.eagle3 import TextConfig as Eagle3TextConfig
from mlx_vlm.speculative.drafters.gemma4_assistant import Gemma4AssistantDraftModel
from mlx_vlm.speculative.drafters.gemma4_assistant.masked_embedder import MaskedEmbedder
from mlx_vlm.speculative.drafters.gemma4_assistant.masks import (
    make_drafter_masks,
    normalize_batched_shared_kv_states,
)
from mlx_vlm.speculative.drafters.gemma4_dflash import ModelConfig as Gemma4DFlashConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp
from mlx_vlm.speculative.drafters.qwen3_dflash import DFlashDraftModel, ModelConfig
from mlx_vlm.speculative.eagle3 import (
    _eagle3_block_settings,
    _eagle3_next_block_size,
    _eagle3_verify_target,
    _eagle3_verify_target_hot,
)
from mlx_vlm.speculative.utils import (
    _dflash_next_block_size,
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
    _speculative_walk_batch_uniform_acceptance,
    _speculative_walk_deferred_greedy,
    speculative_prefill_kwargs,
)
from mlx_vlm.turboquant import BatchTurboQuantKVCache
from mlx_vlm.utils import get_model_and_args

speculative_utils = importlib.import_module("mlx_vlm.speculative.utils")


def test_speculative_sampler_rng_keeps_draft_sampling_off_target_stream():
    logits = mx.zeros((1, 8), dtype=mx.float32)

    def sampler(values):
        return mx.random.categorical(values)

    mx.random.seed(123)
    expected_first = sampler(logits)
    mx.eval(expected_first)
    expected_second = sampler(logits)
    mx.eval(expected_second)

    draft_model = SimpleNamespace(_seed_token=None)

    def draft_prefill():
        draft_model._seed_token = sampler(logits)

    mx.random.seed(123)
    sampler_rng = _SpeculativeSamplerRNG(draft_model, enabled=True)
    first = sampler(logits)
    mx.eval(first)
    sampler_rng.target_sampled()

    sampler_rng.draft_call(draft_prefill)

    second = sampler(logits)
    mx.eval(second)
    sampler_rng.target_sampled()

    assert first.tolist() == expected_first.tolist()
    assert second.tolist() == expected_second.tolist()
    assert draft_model._seed_token is not None


def test_speculative_sampler_rng_can_resync_draft_after_rejected_draws():
    logits = mx.zeros((1, 8), dtype=mx.float32)

    def sampler(values):
        token = mx.random.categorical(values)
        mx.eval(token)
        return token

    mx.random.seed(123)
    expected_first = sampler(logits)
    expected_second = sampler(logits)
    expected_third = sampler(logits)

    draft_model = SimpleNamespace(_seed_token=None)

    def draft_block():
        return mx.stack([mx.random.categorical(logits) for _ in range(3)])

    def draft_seed():
        draft_model._seed_token = mx.random.categorical(logits)

    mx.random.seed(123)
    sampler_rng = _SpeculativeSamplerRNG(draft_model, enabled=True)
    first = sampler(logits)
    sampler_rng.target_sampled(sync_draft=True)

    # The draft stream may spend random draws on tokens later rejected by the
    # target verifier.
    sampler_rng.draft_tokens(draft_block)

    # Only one target token is actually emitted, so the next draft seed should
    # restart from the target stream after that emitted token.
    second = sampler(logits)
    sampler_rng.target_sampled(sync_draft=True)
    sampler_rng.draft_call(draft_seed)
    mx.eval(draft_model._seed_token)

    assert first.tolist() == expected_first.tolist()
    assert second.tolist() == expected_second.tolist()
    assert draft_model._seed_token.tolist() == expected_third.tolist()


def test_speculative_sampler_rng_async_evals_greedy_draft_call_state(monkeypatch):
    result_array = mx.array([1], dtype=mx.int32)
    state_array = mx.array([2], dtype=mx.int32)
    calls = []

    def fake_async_eval(*arrays):
        calls.append(arrays)

    draft_model = SimpleNamespace(
        draft_eval_state=lambda: {"cache": [state_array]},
    )
    sampler_rng = _SpeculativeSamplerRNG(draft_model, enabled=False)

    monkeypatch.setattr(mx, "async_eval", fake_async_eval)
    result = sampler_rng.draft_call(lambda: result_array)

    assert result is result_array
    assert len(calls) == 1
    assert calls[0][0] is result_array
    assert calls[0][1] is state_array


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


def _make_target_config(hidden_size: int):
    return SimpleNamespace(
        model_type="target_text",
        hidden_size=hidden_size,
    )


def _make_target_model(hidden_size: int):
    return SimpleNamespace(
        language_model=SimpleNamespace(config=_make_target_config(hidden_size))
    )


def _make_drafter_config(model_type: str, hidden_size: int, *, field: str):
    kwargs = {"model_type": model_type}
    if field == "backbone_hidden_size":
        kwargs["backbone_hidden_size"] = hidden_size
    elif field == "target_hidden_size":
        kwargs["target_hidden_size"] = hidden_size
    elif field == "text_config.hidden_size":
        kwargs["text_config"] = SimpleNamespace(hidden_size=hidden_size)
    else:
        raise ValueError(f"Unknown hidden-size field: {field}")
    return SimpleNamespace(**kwargs)


MTP_DRAFTER_COMPAT_CASES = [
    pytest.param(
        "gemma4_assistant",
        "backbone_hidden_size",
        id="gemma4-backbone-hidden-size",
    ),
    pytest.param(
        "gemma4_unified_assistant",
        "backbone_hidden_size",
        id="gemma4-unified-backbone-hidden-size",
    ),
    pytest.param(
        "qwen3_5_mtp",
        "text_config.hidden_size",
        id="text-config-hidden-size",
    ),
    pytest.param(
        "custom_mtp",
        "target_hidden_size",
        id="target-hidden-size",
    ),
]


def test_gemma4_rollback_speculative_cache_accepts_python_list():
    class DummyCache:
        keys = None

        def __init__(self):
            self.trims = []

        def trim(self, n):
            self.trims.append(n)

    cache = DummyCache()

    max_a = gemma4_language.LanguageModel.rollback_speculative_cache(
        SimpleNamespace(), [cache], None, [0, 2], block_size=4
    )

    assert max_a == 2
    assert cache.trims == [1]


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


def test_qwen_gated_delta_accept_states_matches_python_gather():
    accepted = mx.array([0, 2, 1, 3], dtype=mx.int32)
    intermediate_states = mx.arange(4 * 4 * 2 * 3 * 5, dtype=mx.float32).reshape(
        4, 4, 2, 3, 5
    )
    conv_input = mx.arange(4 * 7 * 6, dtype=mx.float32).reshape(4, 7, 6)
    live_state = mx.full((4, 2, 3, 5), -1.0, dtype=mx.float32)
    live_conv = mx.full((4, 3, 6), -2.0, dtype=mx.float32)

    ref_state, ref_conv = qwen_language.gated_delta_accept_states(
        intermediate_states,
        conv_input,
        live_state,
        live_conv,
        accepted,
        kernel_size=4,
        use_kernel=False,
    )
    out_state, out_conv = qwen_language.gated_delta_accept_states(
        intermediate_states,
        conv_input,
        live_state,
        live_conv,
        accepted,
        kernel_size=4,
        use_kernel=True,
    )
    mx.eval(ref_state, ref_conv, out_state, out_conv)

    assert bool(mx.array_equal(ref_state, out_state).item())
    assert bool(mx.array_equal(ref_conv, out_conv).item())


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


def test_qwen_gdn_sink_captures_intermediate_states_for_batched_verify():
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    layer = qwen_language.Qwen3_5GatedDeltaNet(config)
    sink = []

    def fake_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True):
        del k, v, a, b, A_log, dt_bias, state, mask, use_kernel
        B, S = q.shape[:2]
        out = mx.zeros((B, S, 2, 4), dtype=mx.float32)
        next_state = mx.zeros((B, 2, 4, 4), dtype=mx.float32)
        states = mx.ones((B, S, 2, 4, 4), dtype=mx.float32)
        return out, next_state, states

    with patch.object(
        qwen_language, "gated_delta_update_with_states", side_effect=fake_update
    ):
        out = layer(
            mx.zeros((2, 3, 16), dtype=mx.float32),
            cache=ArraysCache(size=2),
            gdn_sink=sink,
        )

    mx.eval(out)
    assert out.shape == (2, 3, 16)
    assert sink[0][11].shape == (2, 3, 2, 4, 4)


def test_qwen_gdn_sink_captures_intermediate_states_for_singleton_verify():
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    layer = qwen_language.Qwen3_5GatedDeltaNet(config)
    sink = []

    def fake_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True):
        del k, v, a, b, A_log, dt_bias, state, mask, use_kernel
        B, S = q.shape[:2]
        out = mx.zeros((B, S, 2, 4), dtype=mx.float32)
        next_state = mx.zeros((B, 2, 4, 4), dtype=mx.float32)
        states = mx.ones((B, S, 2, 4, 4), dtype=mx.float32)
        return out, next_state, states

    with patch.object(
        qwen_language, "gated_delta_update_with_states", side_effect=fake_update
    ):
        out = layer(
            mx.zeros((1, 3, 16), dtype=mx.float32),
            cache=ArraysCache(size=2),
            gdn_sink=sink,
        )

    mx.eval(out)
    assert out.shape == (1, 3, 16)
    assert sink[0][11].shape == (1, 3, 2, 4, 4)


def test_qwen_gdn_verify_update_matches_stepwise_path():
    mx.random.seed(16)
    B, S, Hk, D, Hv, Dv = 1, 3, 16, 128, 32, 128
    q = mx.random.normal((B, S, Hk, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, S, Hk, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, S, Hv, Dv)).astype(mx.bfloat16)
    a = mx.random.normal((B, S, Hv)).astype(mx.bfloat16)
    b = mx.random.normal((B, S, Hv)).astype(mx.bfloat16)
    A_log = mx.random.normal((Hv,)).astype(mx.bfloat16)
    dt_bias = mx.ones((Hv,), dtype=mx.bfloat16)
    state = mx.zeros((B, Hv, Dv, D), dtype=mx.float32)

    outputs = []
    states = []
    current_state = state
    for i in range(S):
        out, current_state = qwen_language.gated_delta_update(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            a[:, i : i + 1],
            b[:, i : i + 1],
            A_log,
            dt_bias,
            current_state,
            None,
            use_kernel=False,
        )
        outputs.append(out)
        states.append(current_state)

    ref = (mx.concatenate(outputs, axis=1), current_state, mx.stack(states, axis=1))
    out = qwen_language._gated_delta_update_verify_decode(
        q, k, v, a, b, A_log, dt_bias, state, None, use_kernel=False
    )
    mx.eval(*ref, *out)

    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(ref, out))


def test_qwen_target_verify_linear_matches_singleton_dense_gemv():
    mx.random.seed(7)
    linear = nn.Linear(16, 32, bias=True)
    linear.weight = mx.random.normal((32, 16)).astype(mx.bfloat16)
    linear.bias = mx.random.normal((32,)).astype(mx.bfloat16)
    x = mx.random.normal((3, 4, 16)).astype(mx.bfloat16)

    ref = mx.concatenate(
        [
            mx.concatenate(
                [linear(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])],
                axis=1,
            )
            for row in range(x.shape[0])
        ],
        axis=0,
    )
    out = qwen_language._target_verify_linear(linear, x, target_verify=True)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_gemv_kernel_matches_singleton_dense_gemv():
    mx.random.seed(9)
    linear = nn.Linear(256, 512, bias=False)
    linear.weight = mx.random.normal((512, 256)).astype(mx.bfloat16)
    x = mx.random.normal((1, 4, 256)).astype(mx.bfloat16)

    ref = mx.concatenate([linear(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)
    out = qwen_language._target_verify_linear(linear, x, target_verify=True)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_quantized_linear_matches_singleton_path():
    mx.random.seed(15)
    linear = nn.QuantizedLinear(512, 16, bias=False, group_size=32, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)

    ref = qwen_language._target_verify_timewise(linear, x)
    out = qwen_language._target_verify_linear(linear, x, target_verify=True)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_quantized_linear_matches_singleton_batch_path():
    mx.random.seed(17)
    linear = nn.QuantizedLinear(512, 16, bias=False, group_size=32, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)

    ref = linear(x)
    out = qwen_language._target_verify_quantized_linear(linear, x)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen3_5_decode_quantized_linears_fused_matches_separate():
    for bits in (4, 5):
        mx.random.seed(170 + bits)
        linears = [
            nn.QuantizedLinear(512, out_dim, bias=False, group_size=64, bits=bits)
            for out_dim in (64, 64, 16, 16)
        ]
        for linear in linears:
            linear.scales = linear.scales.astype(mx.bfloat16)
            linear.biases = linear.biases.astype(mx.bfloat16)
        x = mx.random.normal((4, 1, 512), dtype=mx.bfloat16)

        ref = tuple(linear(x) for linear in linears)
        out = qwen_language._decode_quantized_linears_fused(tuple(linears), x)
        mx.eval(*ref, *out)

        assert out is not None
        assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(ref, out))


def test_qwen_target_verify_quantized_argmax_matches_singleton_path():
    mx.random.seed(16)
    linear = nn.QuantizedLinear(512, 16, bias=False, group_size=32, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)

    x = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    ref = mx.argmax(qwen_language._target_verify_timewise(linear, x), axis=-1)
    out = qwen_language._target_verify_quantized_argmax(linear, x)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen3_5_quantized_argmax_batch_as_time_matches_rowwise():
    mx.random.seed(18)
    linear = nn.QuantizedLinear(512, 32, bias=False, group_size=64, bits=4)
    linear.scales = linear.scales.astype(mx.bfloat16)
    linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((4, 1, 512), dtype=mx.bfloat16)

    out = qwen_language._target_verify_quantized_argmax(linear, x)
    ref = mx.concatenate(
        [
            qwen_language._target_verify_quantized_argmax(linear, x[row : row + 1])
            for row in range(x.shape[0])
        ],
        axis=0,
    )
    mx.eval(out, ref)

    assert bool(mx.array_equal(out, ref).item())


def _qwen3_5_ragged_attention_reference(queries, keys, values, pads, scale):
    return mx.concatenate(
        [
            qwen_language.scaled_dot_product_attention(
                queries[row : row + 1],
                keys[row : row + 1, :, pad:, :],
                values[row : row + 1, :, pad:, :],
                cache=None,
                scale=scale,
                mask=None,
            )
            for row, pad in enumerate(pads)
        ],
        axis=0,
    )


def test_qwen3_5_ragged_decode_attention_matches_one_pass_singleton():
    mx.random.seed(19)
    pads = [17, 0]
    scale = 64**-0.5
    queries = mx.random.normal((2, 4, 1, 64), dtype=mx.bfloat16)
    keys = mx.random.normal((2, 2, 256, 64), dtype=mx.bfloat16)
    values = mx.random.normal((2, 2, 256, 64), dtype=mx.bfloat16)

    out = qwen_language._qwen3_5_ragged_decode_attention(
        queries, keys, values, pads, scale
    )
    ref = _qwen3_5_ragged_attention_reference(queries, keys, values, pads, scale)
    mx.eval(out, ref)

    assert out is not None
    assert bool(mx.array_equal(out, ref).item())


def test_qwen3_5_ragged_decode_attention_matches_two_pass_singleton():
    mx.random.seed(20)
    pads = [7, 0]
    scale = 64**-0.5
    key_length = (
        1100 if qwen_language._qwen3_5_device_arch_suffix() in {"d", "s"} else 4112
    )
    queries = mx.random.normal((2, 4, 1, 64), dtype=mx.bfloat16)
    keys = mx.random.normal((2, 2, key_length, 64), dtype=mx.bfloat16)
    values = mx.random.normal((2, 2, key_length, 64), dtype=mx.bfloat16)

    out = qwen_language._qwen3_5_ragged_decode_attention(
        queries, keys, values, pads, scale
    )
    ref = _qwen3_5_ragged_attention_reference(queries, keys, values, pads, scale)
    mx.eval(out, ref)

    assert out is not None
    assert bool(mx.array_equal(out, ref).item())


def test_qwen3_5_ragged_decode_attention_rejects_mixed_plan():
    mx.random.seed(21)
    scale = 64**-0.5
    if qwen_language._qwen3_5_device_arch_suffix() in {"d", "s"}:
        key_length = 1100
        pads = [101, 0]
    else:
        key_length = 4112
        pads = [33, 0]
    queries = mx.random.normal((2, 4, 1, 64), dtype=mx.bfloat16)
    keys = mx.random.normal((2, 2, key_length, 64), dtype=mx.bfloat16)
    values = mx.random.normal((2, 2, key_length, 64), dtype=mx.bfloat16)

    plans = [
        qwen_language._qwen3_5_sdpa_vector_plan(
            key_length - pad, queries.shape[1], keys.shape[1]
        )
        for pad in pads
    ]
    out = qwen_language._qwen3_5_ragged_decode_attention(
        queries, keys, values, pads, scale
    )

    assert len(set(plans)) == 2
    assert out is None


def test_qwen_target_verify_small_projection_matches_singleton_dense_gemv():
    mx.random.seed(10)
    linear = nn.Linear(256, 8, bias=False)
    linear.weight = mx.random.normal((8, 256)).astype(mx.bfloat16)
    x = mx.random.normal((3, 3, 256)).astype(mx.bfloat16)

    ref = mx.concatenate(
        [
            mx.concatenate(
                [linear(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])],
                axis=1,
            )
            for row in range(x.shape[0])
        ],
        axis=0,
    )
    out = qwen_language._target_verify_linear(linear, x, target_verify=True)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_mlp_matches_singleton_dense_path():
    mx.random.seed(11)
    mlp = qwen_language.Qwen3_5MLP(16, 32)
    mlp.gate_proj.weight = mx.random.normal((32, 16)).astype(mx.bfloat16)
    mlp.up_proj.weight = mx.random.normal((32, 16)).astype(mx.bfloat16)
    mlp.down_proj.weight = mx.random.normal((16, 32)).astype(mx.bfloat16)
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)

    ref = mx.concatenate([mlp(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)
    out = mlp(x, target_verify=True)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_norms_match_singleton_path():
    mx.random.seed(12)
    norm = nn.RMSNorm(16, eps=1e-6)
    x = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)

    ref = mx.concatenate([norm(x[:, i : i + 1]) for i in range(x.shape[1])], axis=1)
    out = norm(x)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_target_verify_gated_norm_matches_singleton_path():
    mx.random.seed(13)
    norm = qwen_language.Qwen3_5RMSNormGated(16, eps=1e-6)
    x = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)
    gate = mx.random.normal((2, 4, 3, 16)).astype(mx.bfloat16)

    ref = mx.concatenate(
        [norm(x[:, i : i + 1], gate[:, i : i + 1]) for i in range(x.shape[1])],
        axis=1,
    )
    out = norm(x, gate)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_gdn_verify_conv_matches_singleton_windows():
    mx.random.seed(14)
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    layer = qwen_language.Qwen3_5GatedDeltaNet(config)
    steps = 4
    conv_input = mx.random.normal(
        (2, layer.conv_kernel_size + steps - 1, layer.conv_dim)
    ).astype(mx.bfloat16)

    ref = mx.concatenate(
        [
            layer.conv1d(conv_input[:, offset : offset + layer.conv_kernel_size, :])
            for offset in range(steps)
        ],
        axis=1,
    )
    out = layer._causal_conv1d_verify(conv_input, steps)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen_gdn_decode_conv_matches_conv1d():
    mx.random.seed(141)
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    layer = qwen_language.Qwen3_5GatedDeltaNet(config)
    layer.conv1d.weight = layer.conv1d.weight.astype(mx.bfloat16)
    conv_input = mx.random.normal(
        (3, layer.conv_kernel_size, layer.conv_dim), dtype=mx.bfloat16
    )

    ref = layer.conv1d(conv_input)
    out = layer._causal_conv1d_decode(conv_input)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


def test_qwen3_5_rope_index_ignores_left_padding_for_vision_rows():
    model_config = qwen_language.ModelConfig(
        text_config=_tiny_qwen3_5_text_config(),
        vision_config=SimpleNamespace(spatial_merge_size=2),
        model_type="qwen3_5",
        image_token_id=101,
        video_token_id=102,
        image_token_index=101,
        video_token_index=102,
        vision_start_token_id=100,
        vision_end_token_id=103,
        vocab_size=128,
    )
    lm = qwen_language.LanguageModel.__new__(qwen_language.LanguageModel)
    lm.config = model_config

    singleton_ids = mx.array([[10, 100, 101, 11, 12]], dtype=mx.int32)
    padded_ids = mx.array(
        [[0, 10, 100, 101, 11, 12], [20, 21, 22, 23, 24, 25]],
        dtype=mx.int32,
    )
    attention_mask = mx.array([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    singleton_pos, singleton_delta = lm.get_rope_index(singleton_ids, image_grid_thw)
    padded_pos, padded_delta = lm.get_rope_index(
        padded_ids, image_grid_thw, attention_mask=attention_mask
    )
    mx.eval(singleton_pos, padded_pos, singleton_delta, padded_delta)

    assert padded_pos[:, 0, 1:].tolist() == singleton_pos[:, 0, :].tolist()
    assert padded_delta[0, 0].item() == singleton_delta[0, 0].item()
    assert padded_delta[1, 0].item() == 0


def test_qwen3_5_rope_index_handles_fully_padded_vision_rows():
    lm = qwen_language.LanguageModel.__new__(qwen_language.LanguageModel)
    lm.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
        vision_start_token_id=100,
    )

    input_ids = mx.array(
        [
            [0, 0, 0, 0],
            [10, 100, 101, 11],
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=mx.int32,
    )
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (3, 2, 4)
    assert rope_deltas.tolist() == [[0], [0]]


def test_qwen3_5_single_row_batch_cache_matches_singleton_cache():
    text_config = _tiny_qwen3_5_text_config()
    text_config.num_hidden_layers = 2
    text_config.full_attention_interval = 2
    model = qwen_language.Qwen3_5Model(text_config)

    singleton_cache = [ArraysCache(size=2), KVCache()]
    batch_arrays = ArraysCache(size=2)
    batch_arrays.left_padding = mx.array([0], dtype=mx.int32)
    batch_cache = [batch_arrays, BatchKVCache([0])]

    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    singleton_prompt = model(prompt, cache=singleton_cache)
    batch_prompt = model(prompt, cache=batch_cache)
    mx.eval(singleton_prompt, batch_prompt)

    assert bool(mx.array_equal(singleton_prompt, batch_prompt).item())
    assert isinstance(batch_cache[1], BatchKVCache)

    decode = mx.array([[4]], dtype=mx.int32)
    singleton_decode = model(decode, cache=singleton_cache)
    batch_decode = model(decode, cache=batch_cache)
    mx.eval(singleton_decode, batch_decode)

    assert bool(mx.array_equal(singleton_decode, batch_decode).item())
    assert isinstance(batch_cache[1], BatchKVCache)

    padded_arrays = ArraysCache(size=2)
    padded_arrays.left_padding = mx.array([5, 0], dtype=mx.int32)
    padded_cache = [padded_arrays, BatchKVCache([5, 0])]

    first_chunk = mx.array([[0, 0, 0], [1, 2, 3]], dtype=mx.int32)
    first_out = model(first_chunk, cache=padded_cache)
    mx.eval(first_out, padded_cache[1].offset, padded_cache[1].left_padding)

    assert first_out.shape == (2, 3, text_config.hidden_size)
    assert padded_cache[1].offset.tolist() == [-2, 3]
    assert padded_cache[1].left_padding.tolist() == [5, 0]

    second_chunk = mx.array([[0, 0, 4], [4, 5, 6]], dtype=mx.int32)
    second_out = model(second_chunk, cache=padded_cache)
    mx.eval(second_out, padded_cache[1].offset, padded_cache[1].left_padding)

    assert second_out.shape == (2, 3, text_config.hidden_size)
    assert padded_cache[1].offset.tolist() == [1, 6]
    assert padded_cache[1].left_padding.tolist() == [5, 0]


def test_qwen3_5_single_row_quantized_batch_cache_keeps_prompt_state():
    text_config = _tiny_qwen3_5_text_config()
    text_config.hidden_size = 64
    text_config.intermediate_size = 128
    text_config.num_hidden_layers = 2
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 32
    text_config.full_attention_interval = 2
    model = qwen_language.Qwen3_5Model(text_config)

    batch_arrays = ArraysCache(size=2)
    batch_arrays.left_padding = mx.array([0], dtype=mx.int32)
    quantized_cache = BatchQuantizedKVCache([0], group_size=32, bits=8)
    prompt_cache = [batch_arrays, quantized_cache]

    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    model(prompt, cache=prompt_cache)
    mx.eval(prompt_cache[1].state)

    assert prompt_cache[1] is quantized_cache
    assert quantized_cache.keys is not None
    assert quantized_cache._idx == 3
    assert quantized_cache.offset.tolist() == [3]

    decode = mx.array([[4]], dtype=mx.int32)
    model(decode, cache=prompt_cache)
    mx.eval(prompt_cache[1].state)

    assert prompt_cache[1] is quantized_cache
    assert quantized_cache._idx == 4
    assert quantized_cache.offset.tolist() == [4]


def test_qwen3_5_single_row_shortcut_skips_quantized_batch_caches():
    assert qwen_language._is_single_row_batch_cache(BatchKVCache([0]))
    assert not qwen_language._is_single_row_batch_cache(
        BatchQuantizedKVCache([0], group_size=32, bits=8)
    )
    assert not qwen_language._is_single_row_batch_cache(
        BatchTurboQuantKVCache([0], bits=3.5)
    )


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


def test_mtp_target_cache_buffers_cache_list_local_rotating_cache():
    base = RotatingKVCache(max_size=4, keep=0)
    keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    base.update_and_fetch(keys, keys)
    prompt_cache = [CacheList(base, PoolingCache(4))]
    draft_model = SimpleNamespace(config=SimpleNamespace(block_size=3))

    mtp_utils._buffer_mtp_target_cache(prompt_cache, draft_model, None)

    assert isinstance(prompt_cache[0][0], BufferedRotatingKVCache)
    assert isinstance(prompt_cache[0][1], PoolingCache)
    assert prompt_cache[0][0].state[0].reshape(-1).tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


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


def test_mtp_acceptance_walk_samples_positioned_block_once():
    class FakeEmbed:
        def __init__(self):
            self.calls = 0

        def as_linear(self, hidden):
            self.calls += 1
            return hidden

    class PositionedSampler:
        def __init__(self):
            self.calls = []

        def __call__(self, logprobs):
            raise AssertionError("positioned target sampler was not used")

        def sample_target(self, logprobs, *, row_ids, positions):
            self.calls.append((list(row_ids), list(positions)))
            return mx.argmax(logprobs, axis=-1)

    fake_head = FakeEmbed()
    sampler = PositionedSampler()
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
    verify = mtp_utils._MTPVerifyResult(hidden=target_hidden, shared_kv_states={})

    accepted, new_tokens = mtp_utils._mtp_acceptance_walk(
        lm,
        verify,
        draft_tokens,
        sampler,
        budget=4,
        row_id=5,
        base_position=7,
    )

    assert accepted == 1
    assert new_tokens == [2, 3]
    assert fake_head.calls == 1
    assert sampler.calls == [([5, 5, 5, 5], [7, 8, 9, 10])]


def test_mtp_draft_kwargs_uses_greedy_for_positioned_sampler():
    class PositionedSampler:
        def sample_target(self, logprobs, *, row_ids, positions):
            return mx.argmax(logprobs, axis=-1)

    draft_model = SimpleNamespace(supports_greedy_draft_argmax=True)

    assert mtp_utils._mtp_draft_kwargs(draft_model, False) == {}
    assert mtp_utils._mtp_draft_kwargs(draft_model, False, PositionedSampler()) == {
        "greedy": True
    }


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


def test_speculative_walk_batch_deferred_greedy_uses_positioned_sampler():
    class FakeEmbed:
        def __init__(self):
            self.calls = 0

        def as_linear(self, hidden):
            self.calls += 1
            return hidden

    class PositionedSampler:
        def __init__(self):
            self.calls = []

        def __call__(self, logprobs):
            raise AssertionError("positioned target sampler was not used")

        def sample_target(self, logprobs, *, row_ids, positions):
            self.calls.append((list(row_ids), list(positions)))
            return mx.argmax(logprobs, axis=-1)

    fake_head = FakeEmbed()
    sampler = PositionedSampler()
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
        sampler,
        budgets=[3, 2],
        row_ids=[10, 11],
        base_positions=[7, 12],
    )

    assert accepted == [1, 2]
    assert new_tokens == [[2, 1], [0, 2]]
    assert fake_head.calls == 3
    assert sampler.calls == [
        ([10, 11], [7, 12]),
        ([10, 11], [8, 13]),
        ([10, 11], [9, 14]),
    ]


def test_speculative_walk_batch_deferred_uniform_stops_at_batch_rejection():
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
            ],
            [
                [0, 9, 0, 0],
                [9, 0, 0, 0],
                [0, 0, 9, 0],
            ],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[2, 3], [0, 2]], dtype=mx.int32)

    accepted, new_tokens = mtp_utils._speculative_walk_batch_deferred_uniform(
        lm,
        target_hidden,
        draft_tokens,
        lambda logits: mx.argmax(logits, axis=-1),
        budgets=[3, 3],
    )

    assert accepted == [0, 0]
    assert new_tokens == [[2], [1]]
    assert fake_head.calls == 1


def test_mtp_server_singleton_dispatches_batch_rounds(monkeypatch):
    calls = []

    def fake_batch(*args, **kwargs):
        calls.append(("batch", args, kwargs))
        yield [3], None

    def fake_single(*args, **kwargs):
        raise AssertionError("server MTP singleton should use batch round path")

    monkeypatch.setattr(speculative_utils, "_mtp_rounds_batch", fake_batch)
    monkeypatch.setattr(speculative_utils, "_mtp_rounds", fake_single)

    result = list(
        speculative_utils.run_speculative_server_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            SimpleNamespace(),
            prompt_cache=[],
            hidden=mx.zeros((1, 1, 1), dtype=mx.float32),
            shared_kv_states={},
            draft_kind="mtp",
            first_bonus=mx.array([2], dtype=mx.int32),
            max_tokens=4,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            token_dtype=mx.int32,
            greedy_sampling=False,
            row_ids=[0],
        )
    )

    assert result == [([3], None)]
    assert calls
    assert calls[0][2]["first_bonus"].tolist() == [2]
    assert calls[0][2]["row_ids"] == [0]


def test_mtp_uses_uniform_deferred_walk_for_batched_sampling():
    ragged_drafter = SimpleNamespace(requires_uniform_batch_acceptance=False)
    uniform_drafter = SimpleNamespace(requires_uniform_batch_acceptance=True)
    normal_sampler = lambda logits: mx.argmax(logits, axis=-1)
    positioned_sampler = SimpleNamespace(sample_target=lambda *args, **kwargs: None)

    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter,
        n_active=1,
        greedy_sampling=False,
        sampler=normal_sampler,
    )
    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter,
        n_active=2,
        greedy_sampling=True,
        sampler=normal_sampler,
    )
    assert mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter,
        n_active=2,
        greedy_sampling=False,
        sampler=normal_sampler,
    )
    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter,
        n_active=2,
        greedy_sampling=False,
        sampler=positioned_sampler,
    )
    assert mtp_utils._mtp_use_uniform_deferred_walk(
        uniform_drafter,
        n_active=2,
        greedy_sampling=True,
        sampler=positioned_sampler,
    )


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


def test_mtp_verify_target_prefers_argmax_hidden_hook_for_greedy_tokens():
    verify_input = mx.array([[7, 8]], dtype=mx.int32)
    hidden = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
    target_tokens = mx.array([[3, 4]], dtype=mx.int32)
    calls = []

    def verify_hidden(inputs, cache):
        calls.append((inputs, cache))
        return hidden, {"full": ("k", "v")}, ["gdn"]

    lm = SimpleNamespace(
        speculative_verify_hidden=verify_hidden,
        speculative_argmax_from_hidden=lambda h: target_tokens,
        speculative_verify_logits=lambda *args: (_ for _ in ()).throw(
            AssertionError("full logits should not be used for greedy argmax")
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
    verify = speculative_utils._MTPVerifyResult(
        hidden=mx.zeros((1, 3, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=None,
    )

    with (
        patch.object(mtp_utils, "_mtp_verify_target", return_value=verify),
        patch.object(
            mtp_utils,
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


def test_mtp_rounds_skips_rollback_after_full_accept_with_gdn_states():
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

        def speculative_draft_hidden(self, hidden):
            return hidden

    gdn_states = [tuple([None] * 11 + [mx.zeros((1, 3, 1, 1, 1))])]
    verify = speculative_utils._MTPVerifyResult(
        hidden=mx.zeros((1, 3, 2), dtype=mx.float32),
        shared_kv_states={},
        gdn_states=gdn_states,
    )

    with (
        patch.object(mtp_utils, "_mtp_verify_target", return_value=verify),
        patch.object(
            mtp_utils,
            "_mtp_acceptance_walk",
            return_value=(2, [7, 8]),
        ),
    ):
        list(
            _mtp_rounds(
                SimpleNamespace(language_model=LM()),
                Draft(),
                [SimpleNamespace(offset=0)],
                mx.zeros((1, 1, 2), dtype=mx.float32),
                {},
                first_bonus=1,
                max_tokens=5,
                sampler=lambda logits: mx.argmax(logits, axis=-1),
                draft_block_size=3,
                token_dtype=mx.int32,
                greedy_sampling=True,
            )
        )

    assert rollback_calls == []


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


def test_masked_embedder_token_ordering_is_buffer_not_parameter():
    """token_ordering is a static cluster->vocab-id index, not a learnable param.

    Regression for a silent fine-tuning bug: nn.Module treats every mx.array
    attribute as a trainable parameter, so AdamW's tree_map walked
    token_ordering and applied `param - lr * m / (sqrt(v) + eps)`. Type
    promotion converted int32 -> float32, and the next gather in
    `_selected_logits` raised `indices must be integral`.

    This test confirms (a) token_ordering is absent from trainable_parameters,
    (b) it remains absent after a broad unfreeze, and (c) an AdamW step does
    not change its dtype or contents.
    """
    cfg = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=2, vocab_size=8),
        num_centroids=2,
        centroid_intermediate_top_k=1,
    )
    embedder = MaskedEmbedder(cfg)
    embedder.token_ordering = mx.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=mx.int32)
    original = mx.array(embedder.token_ordering.tolist(), dtype=mx.int32)

    trainable_paths = {p for p, _ in tree_flatten(embedder.trainable_parameters())}
    assert "token_ordering" not in trainable_paths

    embedder.unfreeze()
    trainable_paths = {p for p, _ in tree_flatten(embedder.trainable_parameters())}
    assert "token_ordering" not in trainable_paths

    optimizer = optim.AdamW(learning_rate=1e-3)
    grads = tree_map(lambda p: mx.zeros_like(p), embedder.trainable_parameters())
    optimizer.update(embedder, grads)

    assert embedder.token_ordering.dtype == mx.int32
    assert (embedder.token_ordering == original).all().item()


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
        stats == "Speculative decoding: 2.00 accepted tokens/round "
        "(1.00 accepted drafts/round, 60.0% of drafted, "
        "avg draft 1.67) over 3 rounds"
    )


def test_dflash_block_total_uses_runtime_default_but_honors_override():
    draft_model = SimpleNamespace(
        config=SimpleNamespace(block_size=16, runtime_block_size=14)
    )

    assert speculative_utils._dflash_block_total(draft_model, None) == 14
    assert speculative_utils._dflash_block_total(draft_model, 16) == 16


def test_dflash_block_total_falls_back_to_configured_block_size():
    draft_model = SimpleNamespace(
        config=SimpleNamespace(block_size=16, runtime_block_size=None)
    )

    assert speculative_utils._dflash_block_total(draft_model, None) == 16


def test_dflash_config_defaults_to_checkpoint_block_size():
    config = ModelConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=8,
        target_layer_ids=[0],
    )
    draft_model = SimpleNamespace(config=config)

    assert config.runtime_block_size is None
    assert speculative_utils._dflash_block_total(draft_model, None) == 16


def test_dflash_next_block_size_starts_at_requested_ceiling():
    draft_model = SimpleNamespace(accept_lens=[], draft_lens=[])

    assert _dflash_next_block_size(draft_model, 16, 20) == 16


def test_dflash_next_block_size_backs_off_on_low_acceptance():
    draft_model = SimpleNamespace(accept_lens=[1, 2], draft_lens=[15, 7])

    assert _dflash_next_block_size(draft_model, 16, 20) == 4


def test_dflash_next_block_size_grows_after_full_prefix_hits():
    draft_model = SimpleNamespace(accept_lens=[3, 3, 3, 3], draft_lens=[3, 3, 3, 3])

    assert _dflash_next_block_size(draft_model, 16, 20) == 6


def test_dflash_next_block_size_does_not_grow_on_middling_acceptance():
    draft_model = SimpleNamespace(accept_lens=[3, 2, 1, 3], draft_lens=[3, 3, 3, 3])

    assert _dflash_next_block_size(draft_model, 16, 20) == 4


def test_dflash_next_block_size_can_prefer_requested_size():
    draft_model = SimpleNamespace(
        accept_lens=[0] * 4,
        draft_lens=[15] * 4,
        prefer_requested_block_size=True,
    )

    assert _dflash_next_block_size(draft_model, 16, 8) == 8


def test_dflash_committed_hidden_segments_keep_per_row_lengths():
    hidden = mx.arange(12, dtype=mx.float32).reshape(2, 3, 2)

    segments = speculative_utils._dflash_committed_hidden_segments(
        hidden, [[1, 2], [3]]
    )

    assert segments[0].shape == (1, 2, 2)
    assert segments[0].tolist() == [[[0.0, 1.0], [2.0, 3.0]]]
    assert segments[1].shape == (1, 1, 2)
    assert segments[1].tolist() == [[[6.0, 7.0]]]


def test_gemma4_26b_dflash_config_preserves_capture_layers():
    config = Gemma4DFlashConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 262144,
            "num_target_layers": 30,
            "dflash_config": {"target_layer_ids": [1, 6, 11]},
        }
    )

    assert config.target_layer_ids == [1, 6, 11]
    assert config.runtime_block_size is None


def test_gemma4_31b_dflash_config_preserves_capture_layers():
    config = Gemma4DFlashConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 262144,
            "num_target_layers": 60,
            "dflash_config": {"target_layer_ids": [1, 12, 23]},
        }
    )

    assert config.target_layer_ids == [1, 12, 23]
    assert config.runtime_block_size is None


def test_gemma4_dflash_config_honors_runtime_block_override():
    config = Gemma4DFlashConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 262144,
            "num_target_layers": 30,
            "dflash_config": {
                "target_layer_ids": [1, 6, 11],
                "runtime_block_size": 12,
            },
        }
    )

    assert config.target_layer_ids == [1, 6, 11]
    assert config.runtime_block_size == 12


def test_generic_dflash_config_parses_gemma4_metadata_without_runtime_cap():
    config = ModelConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 262144,
            "num_target_layers": 30,
            "dflash_config": {
                "mask_token_id": 4,
                "target_layer_ids": [1, 6, 11],
            },
        }
    )

    assert config.target_layer_ids == [1, 6, 11]
    assert config.runtime_block_size is None


def test_dflash_drafter_uses_bound_target_embedding_scale():
    class Embed:
        def __call__(self, inputs):
            return mx.ones((*inputs.shape, 4), dtype=mx.float32)

        def as_linear(self, hidden):
            return hidden

    config = ModelConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=8,
        target_layer_ids=[0],
    )
    drafter = DFlashDraftModel(config)
    target = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=Embed(), embed_scale=2.0)
    )

    drafter.bind(target)

    embedded = drafter._embed_input_tokens(mx.array([[1, 2]], dtype=mx.int32))
    assert embedded.tolist() == [[[2.0] * 4, [2.0] * 4]]


def test_dflash_config_parses_sliding_attention_metadata():
    config = ModelConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 16,
            "final_logit_softcapping": 30.0,
            "dflash_config": {
                "target_layer_ids": [0],
                "mask_token_id": 4,
            },
        }
    )

    assert config.layer_types == ["sliding_attention", "full_attention"]
    assert config.sliding_window == 16
    assert config.final_logit_softcapping == 30.0
    assert config.mask_token_id == 4


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


def test_mtp_draft_block_active_uses_batched_path_for_mixed_positions_without_shared_kv():
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
        positions=[11, 12],
    )

    assert drafted.tolist() == [[3, 4], [7, 8]]
    assert draft_model.calls == [((2,), (2, 1, 1))]


def test_speculative_walk_batch_uniform_acceptance_keeps_exact_tokens():
    draft_tokens = mx.array([[10, 11, 12], [20, 21, 22]], dtype=mx.int32)
    target_tokens = mx.array([[10, 99, 98, 97], [20, 21, 77, 76]], dtype=mx.int32)
    accepted, new_tokens = _speculative_walk_batch_uniform_acceptance(
        draft_tokens,
        target_tokens,
        accepted_list=[1, 2],
        budgets=[4, 4],
    )

    assert accepted == [1, 1]
    assert new_tokens == [[10, 99], [20, 21]]


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


def test_kind_none_autodetects_mtp_for_gemma4_unified_assistant(tmp_path):
    path = _make_drafter_dir(tmp_path, "gemma4_unified_assistant")
    assert resolve_drafter_kind(path, None) == "mtp"
    assert resolve_drafter_kind(path) == "mtp"


def test_kind_none_autodetects_mtp_for_qwen3_5_mtp(tmp_path):
    path = _make_drafter_dir(tmp_path, "qwen3_5_mtp")
    assert resolve_drafter_kind(path, None) == "mtp"
    assert resolve_drafter_kind(path, "dflash") == "mtp"


def test_kind_none_autodetects_mtp_for_deepseek_v4_mtp(tmp_path):
    path = _make_drafter_dir(tmp_path, "deepseek_v4_mtp")
    assert resolve_drafter_kind(path, None) == "mtp"
    assert resolve_drafter_kind(path, "dflash") == "mtp"


def test_kind_none_autodetects_eagle3_speculators_config(tmp_path):
    path = tmp_path / "drafter"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"speculators_model_type": "eagle3"}))
    assert resolve_drafter_kind(path, None) == "eagle3"
    assert resolve_drafter_kind(path, "dflash") == "eagle3"


@pytest.mark.parametrize("model_type,hidden_size_field", MTP_DRAFTER_COMPAT_CASES)
def test_mtp_drafter_compatibility_accepts_matching_target(
    model_type, hidden_size_field
):
    target = _make_target_model(hidden_size=4096)
    drafter = SimpleNamespace(
        config=_make_drafter_config(model_type, 4096, field=hidden_size_field)
    )

    validate_drafter_compatibility(target, drafter, "mtp")


@pytest.mark.parametrize("model_type,hidden_size_field", MTP_DRAFTER_COMPAT_CASES)
def test_mtp_drafter_compatibility_rejects_mismatched_target(
    model_type, hidden_size_field
):
    target = _make_target_model(hidden_size=5376)
    drafter = SimpleNamespace(
        config=_make_drafter_config(model_type, 1536, field=hidden_size_field)
    )

    with pytest.raises(ValueError, match="incompatible with the target model") as exc:
        validate_drafter_compatibility(target, drafter, "mtp")

    message = str(exc.value)
    assert "Drafter target hidden_size=1536" in message
    assert "target hidden_size=5376" in message


@pytest.mark.parametrize(
    "model_type",
    [
        "gemma4_assistant",
        "gemma4_unified_assistant",
        "qwen3_5_mtp",
        "deepseek_v4_mtp",
        "custom_mtp",
    ],
)
def test_mtp_drafter_compatibility_requires_mtp_kind(model_type):
    target = _make_target_model(hidden_size=4096)
    drafter = SimpleNamespace(
        config=_make_drafter_config(model_type, 4096, field="text_config.hidden_size")
    )

    with pytest.raises(ValueError, match="requires draft_kind='mtp'"):
        validate_drafter_compatibility(target, drafter, "dflash")


def test_drafter_compatibility_ignores_unknown_backbone_free_drafters():
    target = _make_target_model(hidden_size=5376)
    drafter = SimpleNamespace(
        config=SimpleNamespace(model_type="custom_drafter", hidden_size=1536)
    )

    validate_drafter_compatibility(target, drafter, "dflash")


def test_model_loader_uses_speculators_model_type_for_eagle3_config():
    arch, model_type = get_model_and_args({"speculators_model_type": "eagle3"})

    assert model_type == "eagle3"
    assert arch.Model is Eagle3DraftModel


def test_model_loader_uses_gemma4_unified_assistant_drafter():
    arch, model_type = get_model_and_args({"model_type": "gemma4_unified_assistant"})

    assert model_type == "gemma4_unified_assistant"
    assert arch.Model is Gemma4AssistantDraftModel

    config = arch.ModelConfig.from_dict(
        {
            "model_type": "gemma4_unified_assistant",
            "backbone_hidden_size": 3840,
            "text_config": {
                "model_type": "gemma4_unified_text",
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "num_kv_shared_layers": 0,
            },
        }
    )
    assert config.model_type == "gemma4_unified_assistant"
    assert config.backbone_hidden_size == 3840
    assert config.text_config.model_type == "gemma4_unified_text"
    assert config.text_config.num_kv_shared_layers == 4


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


def _tiny_deepseek_v4_config():
    return deepseek_language.ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        head_dim=8,
        o_groups=1,
        o_lora_rank=8,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=1,
        num_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        compress_ratios=[0],
        sliding_window=16,
        max_position_embeddings=128,
    )


def test_eagle3_config_uses_speculators_fields():
    cfg = Eagle3Config.from_dict(
        {
            "speculators_model_type": "eagle3",
            "eagle_aux_hidden_state_layer_ids": [2, 30, 57],
            "speculators_config": {
                "proposal_methods": [{"speculative_tokens": 3}],
            },
            "transformer_layer_config": {
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
            },
        }
    )

    assert cfg.model_type == "eagle3"
    assert cfg.block_size == 5
    assert cfg.target_layer_ids == [2, 30, 57]
    assert cfg.capture_layer_ids == [1, 29, 56]
    assert cfg.transformer_layer_config.hidden_size == 8


def test_eagle3_prefill_uses_mlx_capture_layer_indexes():
    cfg = Eagle3Config(eagle_aux_hidden_state_layer_ids=[2, 30, 57])
    drafter = SimpleNamespace(config=cfg)

    assert speculative_prefill_kwargs("eagle3", drafter) == {
        "capture_layer_ids": [1, 29, 56]
    }


def test_eagle3_adaptive_block_size_grows_and_backs_off():
    cfg = Eagle3Config(
        block_size=5,
        adaptive_max_block_size=12,
        transformer_layer_config=Eagle3TextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=32,
        ),
    )
    drafter = SimpleNamespace(config=cfg, accept_lens=[], draft_lens=[])
    block_total, configured, adaptive = _eagle3_block_settings(drafter, None)

    assert (block_total, configured, adaptive) == (12, 5, True)
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 5
    )

    drafter.accept_lens = [0, 0, 1, 1, 0, 1]
    drafter.draft_lens = [4, 4, 4, 4, 4, 4]
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 12
    )

    drafter.accept_lens = [4, 0, 4, 0, 4, 0, 4]
    drafter.draft_lens = [4, 4, 4, 4, 4, 4, 4]
    drafter._adaptive_block_size = 5
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 8
    )

    drafter.accept_lens = [0, 0, 0, 1, 1, 1]
    drafter.draft_lens = [11, 11, 11, 11, 11, 11]
    drafter._adaptive_block_size = 12
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 8
    )

    drafter.accept_lens.extend([0, 0, 0, 0, 0, 0])
    drafter.draft_lens.extend([7, 7, 7, 7, 7, 7])
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 5
    )


def test_eagle3_default_block_size_stays_at_checkpoint_depth():
    cfg = Eagle3Config(block_size=5)
    drafter = SimpleNamespace(config=cfg, accept_lens=[], draft_lens=[])

    assert _eagle3_block_settings(drafter, None) == (5, 5, False)


def test_eagle3_user_block_size_disables_adaptive_block_size():
    cfg = Eagle3Config(block_size=5)
    drafter = SimpleNamespace(config=cfg, accept_lens=[0] * 6, draft_lens=[15] * 6)
    block_total, configured, adaptive = _eagle3_block_settings(drafter, 16)

    assert (block_total, configured, adaptive) == (16, 5, False)
    assert (
        _eagle3_next_block_size(
            drafter, block_total, configured, 128, adaptive=adaptive
        )
        == 16
    )


def test_eagle3_gemma4_verification_seeds_then_batches_tail():
    class FakeGemma4LM:
        __module__ = "mlx_vlm.models.gemma4.language"

        def __init__(self):
            self.calls = []

        def __call__(self, inputs, cache, capture_layer_ids):
            del cache, capture_layer_ids
            self.calls.append(inputs.tolist())
            hidden = inputs.astype(mx.float32)[..., None]
            return SimpleNamespace(
                hidden_states=[hidden],
                logits=mx.zeros((*inputs.shape, 4), dtype=mx.float32),
                gdn_states=None,
            )

    lm = FakeGemma4LM()
    next_token = 0

    def sampler(logits):
        nonlocal next_token
        width = int(logits.shape[1])
        out = mx.arange(next_token + 1, next_token + width + 1, dtype=mx.int32)
        next_token += width
        return out[None, :]

    hidden, target_tokens, gdn_states = _eagle3_verify_target(
        lm,
        mx.array([[10, 11, 12]], dtype=mx.int32),
        prompt_cache=[],
        sampler=sampler,
        target_layer_ids=[1],
    )

    assert lm.calls == [[[10]], [[11, 12]]]
    assert hidden.squeeze(-1).tolist() == [[10.0, 11.0, 12.0]]
    assert target_tokens.tolist() == [[1, 2, 3]]
    assert gdn_states is None


def test_eagle3_hot_verifier_uses_draft_vocab_and_eos():
    class FakeEmbedding:
        def __init__(self):
            self.weight = mx.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [3.5, -0.5],
                ],
                dtype=mx.float32,
            )

    class FakeModel:
        def __init__(self):
            self.embed_tokens = FakeEmbedding()

        def __call__(self, inputs, cache, capture_layer_ids, hidden_sink):
            del cache, capture_layer_ids
            hidden = inputs.astype(mx.float32)
            hidden = mx.stack([hidden, hidden + 1], axis=-1)
            hidden_sink.append(hidden)
            return hidden

    class FakeGemma4LM:
        __module__ = "mlx_vlm.models.gemma4.language"

        def __init__(self):
            self.config = SimpleNamespace(eos_token_id=4)
            self.model = FakeModel()
            self.final_logit_softcapping = None

        def logits_from_hidden(self, hidden):
            return self.model.embed_tokens.weight[None, : hidden.shape[1], :]

    drafter = SimpleNamespace(d2t=mx.array([1, 1], dtype=mx.int32))

    hidden, target_tokens, gdn_states = _eagle3_verify_target_hot(
        FakeGemma4LM(),
        drafter,
        mx.array([[0, 1]], dtype=mx.int32),
        prompt_cache=[],
        sampler=lambda logits: mx.array([[4]], dtype=mx.int32),
        target_layer_ids=[1],
    )

    assert hidden.tolist() == [[[0.0, 1.0], [1.0, 2.0]]]
    assert target_tokens.tolist() == [[1, 4]]
    assert gdn_states is None


def test_eagle3_accept_replays_committed_tokens_with_verifier_hidden():
    cfg = Eagle3Config(
        draft_vocab_size=8,
        transformer_layer_config=Eagle3TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
        ),
    )
    drafter = Eagle3DraftModel(cfg)

    class FakeCache:
        def __init__(self):
            self.trimmed = []

        def trim(self, n):
            self.trimmed.append(n)

    fake_cache = FakeCache()
    drafter._cache = [fake_cache]
    drafter._round_appended = 2
    drafter._next_position = 7
    calls = {}

    def fake_forward(self, tokens, hiddens, token_dtype):
        calls["tokens"] = tokens
        calls["hiddens"] = hiddens
        calls["token_dtype"] = token_dtype
        return mx.zeros((1, tokens.shape[1], self.hidden_size), dtype=mx.float32)

    def fake_seed(self, hidden, sampler, token_dtype, greedy):
        calls["seed_shape"] = hidden.shape
        calls["greedy"] = greedy

    drafter._forward_tokens = MethodType(fake_forward, drafter)
    drafter._set_seed_from_hidden = MethodType(fake_seed, drafter)

    verify_hidden = mx.arange(4 * 12, dtype=mx.float32).reshape(1, 4, 12)
    draft_tokens = mx.array([[10, 11, 12]], dtype=mx.int32)

    drafter.accept_verified_tokens(
        verify_hidden,
        draft_tokens,
        accepted=2,
        new_tokens=[10, 11, 99],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        token_dtype=mx.int32,
        greedy=True,
    )

    assert fake_cache.trimmed == [2]
    assert drafter._next_position == 5
    assert calls["tokens"].tolist() == [[10, 11, 99]]
    assert calls["hiddens"].tolist() == verify_hidden[:, :3, :].tolist()
    assert calls["greedy"] is True


def test_eagle3_draft_vocab_mapping_uses_d2t_offsets():
    cfg = Eagle3Config(
        draft_vocab_size=4,
        transformer_layer_config=Eagle3TextConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
        ),
    )
    model = Eagle3DraftModel(cfg)
    model.d2t = mx.array([0, 4, 8, 12], dtype=mx.int32)

    mapped = model._draft_to_target(mx.array([[0, 1, 3]], dtype=mx.int32), mx.int32)

    assert mapped.tolist() == [[0, 5, 15]]


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


def test_qwen3_5_mtp_batch_accept_updates_uniform_cache():
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
    hidden = mx.zeros((2, 1, 16), dtype=mx.float32)
    draft_tokens = drafter.draft_block(
        mx.array([7, 8], dtype=mx.int32),
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[0, 0],
        new_tokens=[[3], [4]],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        token_dtype=mx.int32,
        greedy=True,
    )

    mx.eval(drafter._seed_token)
    assert drafter._seed_token.shape == (2, 1)
    assert drafter._round_appended == 0
    assert drafter._cache[0].offset == 1
    assert drafter._next_position == 5


def test_qwen3_5_mtp_batch_accept_updates_ragged_cache():
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
    drafter.reset(target, left_padding=[0, 0])
    drafter.set_shared_kv(
        {},
        kv_offset=4,
        position=mx.array([4, 4], dtype=mx.int32),
        kv_valid_len=mx.array([4, 4], dtype=mx.int32),
    )
    hidden = mx.zeros((2, 1, 16), dtype=mx.float32)
    draft_tokens = drafter.draft_block(
        mx.array([7, 8], dtype=mx.int32),
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[1, 0],
        new_tokens=[[int(draft_tokens[0, 0].item()), 5], [6]],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        token_dtype=mx.int32,
        greedy=True,
    )

    mx.eval(drafter._seed_token, drafter._cache[0].offset)
    assert drafter._seed_token.shape == (2, 1)
    assert drafter._round_appended == 0
    assert drafter._cache[0]._idx == 3
    assert drafter._cache[0].offset.tolist() == [2, 1]
    assert drafter._cache[0].left_padding.tolist() == [1, 2]
    assert drafter._next_position.tolist() == [6, 5]


def test_qwen3_5_rollback_speculative_cache_trims_batch_rows_ragged():
    text_config = _tiny_qwen3_5_text_config()
    language = qwen_language.LanguageModel(args=text_config)
    cache = qwen_language.KVCache.merge(
        [qwen_language.KVCache(), qwen_language.KVCache()]
    )
    keys = mx.arange(2 * 1 * 5 * 4, dtype=mx.float32).reshape(2, 1, 5, 4)
    values = keys + 100
    cache.update_and_fetch(keys, values)

    language.rollback_speculative_cache([cache], [], mx.array([2, 0]), block_size=3)

    mx.eval(cache.keys, cache.values, cache.offset, cache.left_padding)
    assert cache._idx == 5
    assert cache.offset.tolist() == [5, 3]
    assert cache.left_padding.tolist() == [0, 2]


def test_qwen3_5_rollback_speculative_cache_handles_turboquant_batch_kv():
    cache = BatchTurboQuantKVCache([0, 0], bits=3.5)
    keys = mx.arange(2 * 1 * 7 * 8, dtype=mx.float32).reshape(2, 1, 7, 8)
    values = keys + 100
    cache.update_and_fetch(keys, values)

    qwen_language.LanguageModel.rollback_speculative_cache(
        None, [cache], [], mx.array([0, 2]), block_size=5
    )

    mx.eval(cache.keys.norms, cache.keys.indices, cache.values.norms, cache.offset)
    assert cache._idx == 5
    assert cache.offset.tolist() == [5, 5]
    assert mx.all(cache.keys.norms[0, :, 3:5] == 0).item()
    assert mx.all(cache.keys.indices[0, :, 3:5, :] == 0).item()
    assert mx.all(cache.values.norms[0, :, 3:5] == 0).item()
    assert mx.any(cache.keys.norms[1, :, 3:5] != 0).item()


def test_gemma4_rollback_speculative_cache_handles_turboquant_batch_tail_zero():
    cache = BatchTurboQuantKVCache([0, 0], bits=3.5)
    keys = mx.arange(2 * 1 * 5 * 8, dtype=mx.float32).reshape(2, 1, 5, 8)
    values = keys + 100
    cache.update_and_fetch(keys, values)

    gemma4_language.LanguageModel.rollback_speculative_cache(
        None, [cache], [], mx.array([0, 2]), block_size=3
    )

    mx.eval(cache.keys.norms, cache.keys.indices, cache.values.norms)
    assert mx.all(cache.keys.norms[0, :, 3:5] == 0).item()
    assert mx.all(cache.keys.indices[0, :, 3:5, :] == 0).item()
    assert mx.all(cache.values.norms[0, :, 3:5] == 0).item()
    assert mx.any(cache.keys.norms[1, :, 3:5] != 0).item()


def test_deepseek_v4_rollback_speculative_cache_handles_turboquant_batch_tail_zero():
    cache = BatchTurboQuantKVCache([0, 0], bits=3.5)
    keys = mx.arange(2 * 1 * 5 * 8, dtype=mx.float32).reshape(2, 1, 5, 8)
    values = keys + 100
    cache.update_and_fetch(keys, values)

    deepseek_language.LanguageModel.rollback_speculative_cache(
        None, [cache], [], mx.array([0, 2]), block_size=3
    )

    mx.eval(cache.keys.norms, cache.keys.indices, cache.values.norms)
    assert mx.all(cache.keys.norms[0, :, 3:5] == 0).item()
    assert mx.all(cache.keys.indices[0, :, 3:5, :] == 0).item()
    assert mx.all(cache.values.norms[0, :, 3:5] == 0).item()
    assert mx.any(cache.keys.norms[1, :, 3:5] != 0).item()


def test_qwen3_5_mtp_filter_batch_keeps_drafter_state_aligned():
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
    drafter.set_shared_kv(
        {},
        kv_offset=4,
        position=mx.array([3, 3], dtype=mx.int32),
        kv_valid_len=mx.array([4, 4], dtype=mx.int32),
    )
    hidden = mx.zeros((2, 1, 16), dtype=mx.float32)
    draft_tokens = drafter.draft_block(
        mx.array([7, 8], dtype=mx.int32),
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[1, 1],
        new_tokens=[[3, 5], [4, 6]],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        token_dtype=mx.int32,
        greedy=True,
    )

    drafter.filter_batch(mx.array([1], dtype=mx.int32))
    mx.eval(drafter._cache[0].keys, drafter._seed_token)
    assert drafter._cache[0].keys.shape[0] == 1
    assert drafter._seed_token.shape == (1, 1)
    assert drafter._next_position.tolist() == [6]


def test_qwen3_5_mtp_filter_batch_keeps_batch_cache_padding_aligned():
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
    drafter.reset(target, left_padding=[0, 1, 2])
    drafter._cache[0].update_and_fetch(
        mx.zeros((3, 1, 2, 8), dtype=mx.float32),
        mx.zeros((3, 1, 2, 8), dtype=mx.float32),
    )
    drafter._next_position = mx.array([4, 5, 6], dtype=mx.int32)

    drafter.filter_batch(mx.array([0, 2], dtype=mx.int32))

    mx.eval(drafter._cache[0].left_padding, drafter._cache[0].offset)
    assert drafter._cache[0].keys.shape[0] == 2
    assert drafter._cache[0].left_padding.tolist() == [0, 2]
    assert drafter._cache[0].offset.tolist() == [2, 0]
    assert drafter._next_position.tolist() == [4, 6]


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


def test_deepseek_v4_returns_mtp_hidden_and_trims_without_snapshot():
    cfg = _tiny_deepseek_v4_config()
    lm = deepseek_language.LanguageModel(cfg)
    cache = lm.make_cache()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

    hidden, shared_kv, rollback_state = lm.speculative_verify_hidden(inputs, cache)
    mx.eval(hidden)

    assert hidden.shape == (1, 3, cfg.hc_mult, cfg.hidden_size)
    assert shared_kv == {}
    assert rollback_state is None
    assert cache[0].offset == 3

    lm.rollback_speculative_cache(cache, rollback_state, accepted=0, block_size=3)
    assert cache[0].offset == 1

    logits = lm.speculative_logits_from_hidden(hidden[:, :1])
    mx.eval(logits)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_deepseek_v4_replay_snapshot_required_only_when_pooling_can_cross_window():
    pool = PoolingCache(4)
    pool.accumulate_windows(mx.array([[[10.0]]]), mx.ones((1, 1, 1)), offset=0)

    assert not deepseek_language._needs_replay_snapshot_for_cache([pool], 2)
    assert deepseek_language._needs_replay_snapshot_for_cache([pool], 3)
    assert not deepseek_language._needs_replay_snapshot_for_cache(
        [RotatingKVCache(max_size=8)], 3
    )


def test_deepseek_v4_pooling_snapshot_skips_clone_when_verify_does_not_overwrite_remainder():
    pool = PoolingCache(4)
    old_kv = mx.array([[[10.0]]])
    old_gate = mx.array([[[1.0]]])
    pool.accumulate_windows(old_kv, old_gate, offset=0)

    snapshot = deepseek_language._snapshot_cache_state([pool], incoming_tokens=3)
    assert snapshot[0][2] is None

    new_kv = mx.array([[[20.0], [21.0], [22.0]]])
    new_gate = mx.ones_like(new_kv)
    pooled, _, _ = pool.accumulate_windows(new_kv, new_gate, offset=1)
    pool.update_and_fetch(pooled)

    deepseek_language._restore_cache_state([pool], snapshot)

    assert pool.remainder == 1
    assert pool.pooled is None
    assert pool.buf_kv[:, :1].reshape(-1).tolist() == [10.0]


def test_deepseek_v4_pooling_snapshot_restores_only_overwritten_prefix():
    pool = PoolingCache(4)
    old_kv = mx.array([[[10.0], [11.0], [12.0]]])
    old_gate = mx.ones_like(old_kv)
    pool.accumulate_windows(old_kv, old_gate, offset=0)

    snapshot = deepseek_language._snapshot_cache_state([pool], incoming_tokens=3)
    assert snapshot[0][2].shape == (1, 2, 1)

    new_kv = mx.array([[[20.0], [21.0], [22.0]]])
    new_gate = mx.ones_like(new_kv)
    pooled, _, _ = pool.accumulate_windows(new_kv, new_gate, offset=3)
    pool.update_and_fetch(pooled)

    deepseek_language._restore_cache_state([pool], snapshot)

    assert pool.remainder == 3
    assert pool.pooled is None
    assert pool.buf_kv[:, :3].reshape(-1).tolist() == [10.0, 11.0, 12.0]


def test_deepseek_v4_language_ignores_generation_metadata_kwargs():
    cfg = _tiny_deepseek_v4_config()
    lm = deepseek_language.LanguageModel(cfg)
    out = lm(mx.array([[1]], dtype=mx.int32), fps=2.0, image=None, video=None)
    mx.eval(out.logits)

    assert out.logits.shape == (1, 1, cfg.vocab_size)


def test_deepseek_v4_local_mask_aligns_to_layer_cache_width():
    mask = mx.array([[[[False, True, True, False, True]]]], dtype=mx.bool_)

    trimmed = deepseek_language._align_local_mask(mask, 3)
    padded = deepseek_language._align_local_mask(trimmed, 5)

    assert trimmed.tolist() == [[[[True, False, True]]]]
    assert padded.tolist() == [[[[True, True, True, False, True]]]]


def test_deepseek_v4_mtp_draft_block_smoke():
    text_config = _tiny_deepseek_v4_config()
    drafter = DeepseekV4MTPDraftModel(
        DeepseekV4MTPConfig(text_config=text_config, block_size=3)
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=nn.Embedding(32, 16))
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    hidden = mx.zeros((1, 1, text_config.hc_mult, 16), dtype=mx.float32)
    tokens = drafter.draft_block(
        7,
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    mx.eval(tokens)
    assert tokens.shape == (1, 2)


def test_deepseek_v4_mtp_runtime_block_size_defaults_to_native_nextn_depth():
    text_config = _tiny_deepseek_v4_config()
    cfg = DeepseekV4MTPConfig.from_dict(
        {
            "model_type": "deepseek_v4_mtp",
            "text_config": text_config.to_dict(),
            "block_size": 3,
        }
    )

    assert cfg.block_size == 3
    assert cfg.runtime_block_size == 2


def test_deepseek_v4_mtp_batch_accept_updates_uniform_cache():
    text_config = _tiny_deepseek_v4_config()
    drafter = DeepseekV4MTPDraftModel(
        DeepseekV4MTPConfig(text_config=text_config, block_size=3)
    )
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=nn.Embedding(32, 16))
        )
    )
    drafter.reset(target)
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    hidden = mx.zeros((2, 1, text_config.hc_mult, 16), dtype=mx.float32)
    draft_tokens = drafter.draft_block(
        mx.array([7, 8], dtype=mx.int32),
        hidden,
        None,
        3,
        lambda logits: mx.argmax(logits, axis=-1),
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, text_config.hc_mult, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[0, 0],
        new_tokens=[[3], [4]],
        sampler=lambda logits: mx.argmax(logits, axis=-1),
        token_dtype=mx.int32,
        greedy=True,
    )

    mx.eval(drafter._seed_token)
    assert drafter._seed_token.shape == (2, 1)
    assert drafter._seed_hidden.shape == (2, 1, text_config.hc_mult, 16)
    assert drafter._round_appended == 0
    assert drafter._cache[0].offset == 1
    assert drafter._next_position == 5


def test_deepseek_v4_mtp_sanitize_maps_embedded_weights():
    cfg = _tiny_deepseek_v4_config()
    context = SimpleNamespace(args=cfg)
    weights = {
        "mtp.0.e_proj.weight": mx.zeros((4, 4), dtype=mx.uint8),
        "mtp.0.e_proj.scale": mx.ones((1, 1), dtype=mx.float32),
        "mtp.0.ffn.gate.bias": mx.zeros((cfg.n_routed_experts,)),
        "mtp.0.hc_attn_fn": mx.ones((2, 2)),
        "mtp.0.hc_head_scale": mx.ones((1,)),
        "mtp.0.attn.wo_a.weight": mx.ones((cfg.o_lora_rank, 4)),
    }
    for expert in range(cfg.n_routed_experts):
        for name in ("w1", "w2", "w3"):
            weights[f"mtp.0.ffn.experts.{expert}.{name}.weight"] = mx.zeros(
                (4, 16), dtype=mx.int8
            )
            weights[f"mtp.0.ffn.experts.{expert}.{name}.scale"] = mx.ones(
                (4, 1), dtype=mx.uint8
            )

    out = DeepseekV4MTPDraftModel.sanitize(context, weights)

    assert "e_proj.scales" in out
    assert "decoder.ffn.gate.e_score_correction_bias" in out
    assert "decoder.attn_hc.fn" in out
    assert "hc_head.scale" in out
    assert out["decoder.ffn.switch_mlp.gate_proj.weight"].shape[0] == (
        cfg.n_routed_experts
    )
    assert out["decoder.ffn.switch_mlp.gate_proj.scales"].shape[0] == (
        cfg.n_routed_experts
    )
    assert out["decoder.attn.wo_a.weight"].shape == (1, cfg.o_lora_rank, 4)


def test_split_deepseek_v4_mtp_writes_sidecar_without_index_mtp_entries(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    source.mkdir()
    text_config = _tiny_deepseek_v4_config()
    (source / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", **text_config.to_dict()})
    )
    mx.save_safetensors(
        str(source / "mtp.safetensors"),
        {
            "mtp.0.e_proj.weight": mx.zeros((4, 4), dtype=mx.uint8),
            "mtp.0.e_proj.scale": mx.ones((1, 1), dtype=mx.float32),
            "mtp.0.attn.wq_a.weight": mx.zeros((4, 4), dtype=mx.uint8),
            "mtp.0.attn.wq_a.scale": mx.ones((1, 1), dtype=mx.float32),
            "mtp.0.enorm.weight": mx.zeros((text_config.hidden_size,)),
        },
        metadata={},
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.foo": "model.safetensors"}})
    )

    split_deepseek_v4_mtp(str(source), str(output))

    with open(output / "config.json") as f:
        cfg = json.load(f)
    weights = mx.load(str(output / "model.safetensors"))
    assert cfg["model_type"] == "deepseek_v4_mtp"
    assert cfg["block_size"] == 2
    assert cfg["quantization"]["e_proj"]["mode"] == "mxfp8"
    assert cfg["quantization"]["decoder.attn.wq_a"]["mode"] == "mxfp8"
    assert "e_proj.weight" in weights
    assert "e_proj.scales" in weights
    assert "enorm.weight" in weights
