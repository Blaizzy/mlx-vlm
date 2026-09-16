"""Shared drafter contracts, speculative decoding, verification, and cache lifetimes."""

import importlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map, tree_map_with_path

import mlx_vlm.models.deepseek_v4.language as deepseek_language
import mlx_vlm.models.fast_ops as fast_ops
import mlx_vlm.models.gemma4.language as gemma4_language
import mlx_vlm.models.laguna.language as laguna_language
import mlx_vlm.models.minimax_m3_vl.language as minimax_language
import mlx_vlm.models.qwen3_5.gated_delta as qwen_gated_delta
import mlx_vlm.models.qwen3_5.language as qwen_language
import mlx_vlm.models.qwen3_5.speculative_verifier as qwen_verifier
import mlx_vlm.models.qwen3_5_moe.language as qwen_moe_language
import mlx_vlm.speculative.cache_state as speculative_cache_state
import mlx_vlm.speculative.mtp as mtp_utils
import mlx_vlm.speculative.ops.linear as verifier_linear
from mlx_vlm.generate.ar import PromptProcessingBatch, _make_cache, generate_step
from mlx_vlm.models.base import InputEmbeddingsFeatures, LanguageModelOutput
from mlx_vlm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchQuantizedKVCache,
    BatchRotatingKVCache,
    BufferedRotatingKVCache,
    CacheList,
    KVCache,
    PoolingCache,
    RotatingKVCache,
)
from mlx_vlm.models.deepseek_v4.language import LanguageModel as DeepseekLanguageModel
from mlx_vlm.models.glm5_next.language import LanguageModel as GlmLanguageModel
from mlx_vlm.models.lfm2 import Model as Lfm2Model
from mlx_vlm.models.lfm2 import ModelConfig as Lfm2Config
from mlx_vlm.models.lfm2.language import Lfm2MoeSparseMoeBlock
from mlx_vlm.models.lfm2.speculative_verifier import Lfm2ExactSpeculativeVerifier
from mlx_vlm.models.lfm2_moe import Model as Lfm2MoeModel
from mlx_vlm.models.lfm2_moe import ModelConfig as Lfm2MoeConfig
from mlx_vlm.models.linear import native_batch_linear
from mlx_vlm.models.minimax_m3_vl.language import (
    MiniMaxM3BatchKVCache,
    MiniMaxM3KVCache,
)
from mlx_vlm.models.muse_glimmer import Model as MuseGlimmerModel
from mlx_vlm.models.muse_glimmer import ModelConfig as MuseGlimmerConfig
from mlx_vlm.models.muse_glimmer import TextConfig, VisionConfig
from mlx_vlm.models.quantized_verifier import (
    decode_quantized_argmax,
    decode_quantized_linear,
    exact_quantized_linear,
    exact_quantized_moe_hc_expand,
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
)
from mlx_vlm.models.qwen4_exp.config import TextConfig as Qwen4TextConfig
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_vlm.quantization.one_bit import OneBitLinear
from mlx_vlm.server.generation import _PositionedTargetSampler
from mlx_vlm.speculative.cache_state import start_speculative_cache
from mlx_vlm.speculative.common import _SpeculativeSamplerRNG, verify_forward
from mlx_vlm.speculative.dflash import (
    _dflash_rounds,
    _dflash_rounds_batch,
    _dflash_verify,
    _dflash_verify_greedy,
)
from mlx_vlm.speculative.drafters import (
    DRAFTER_KIND_BY_MODEL_TYPE,
    KNOWN_DRAFTER_KINDS,
    resolve_drafter_kind,
    validate_drafter_compatibility,
)
from mlx_vlm.speculative.drafters.deepseek_v4_dspark import DeepseekV4DsparkDraftModel
from mlx_vlm.speculative.drafters.deepseek_v4_dspark.config import (
    DeepseekV4DsparkConfig,
)
from mlx_vlm.speculative.drafters.deepseek_v4_dspark.split import (
    split_deepseek_v4_dspark,
)
from mlx_vlm.speculative.drafters.deepseek_v4_mtp import DeepseekV4MTPDraftModel
from mlx_vlm.speculative.drafters.deepseek_v4_mtp.config import DeepseekV4MTPConfig
from mlx_vlm.speculative.drafters.deepseek_v4_mtp.split import split_deepseek_v4_mtp
from mlx_vlm.speculative.drafters.dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.dflash2 import ModelConfig as DFlash2Config
from mlx_vlm.speculative.drafters.dspark import DSparkDraftModel
from mlx_vlm.speculative.drafters.dspark import ModelConfig as DSparkConfig
from mlx_vlm.speculative.drafters.dspark import validate_dspark_target
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
from mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split import split_glm4_moe_lite_mtp
from mlx_vlm.speculative.drafters.glm5_next_mtp import Glm5NextMTPDraftModel
from mlx_vlm.speculative.drafters.glm5_next_mtp import ModelConfig as Glm5NextMTPConfig
from mlx_vlm.speculative.drafters.glm5_next_mtp.split import split_glm5_next_mtp
from mlx_vlm.speculative.drafters.laguna_dflash import ModelConfig as LagunaDFlashConfig
from mlx_vlm.speculative.drafters.laguna_dflash.config import (
    expected_laguna_dflash_weight_shapes,
    validate_laguna_dflash_target,
    validate_laguna_dflash_weights,
)
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    Model as MuseGlimmerAssistantModel,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    ModelConfig as MuseGlimmerAssistantConfig,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    expected_muse_glimmer_assistant_weight_shapes,
    validate_muse_glimmer_assistant_weights,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp import ModelConfig as Qwen3_5MTPConfig
from mlx_vlm.speculative.drafters.qwen3_5_mtp import Qwen3_5MTPDraftModel
from mlx_vlm.speculative.drafters.qwen3_5_mtp.split import split_qwen3_5_mtp
from mlx_vlm.speculative.drafters.qwen3_dflash import DFlashDraftModel, ModelConfig
from mlx_vlm.speculative.eagle3 import (
    _eagle3_block_settings,
    _eagle3_next_block_size,
    _eagle3_rounds,
    _eagle3_rounds_batch,
    _eagle3_verify_target,
    _eagle3_verify_target_hot,
)
from mlx_vlm.speculative.mtp import _mtp_rounds_batch
from mlx_vlm.speculative.utils import (
    _dflash_next_block_size,
    _format_speculative_stats,
    _mtp_draft_block_active,
    _mtp_rounds,
    _mtp_shared_kv_from_prompt_cache,
    _mtp_verify_target,
    _speculative_walk,
    _speculative_walk_batch,
    _speculative_walk_batch_deferred_greedy,
    _speculative_walk_batch_uniform_acceptance,
    _speculative_walk_deferred_greedy,
    make_speculative_prompt_cache,
    speculative_prefill_kwargs,
)
from mlx_vlm.split_mtp import split_mtp
from mlx_vlm.tests.speculative_fixtures import (
    tiny_deepseek_config as _tiny_deepseek_v4_config,
)
from mlx_vlm.tests.speculative_fixtures import (
    tiny_glm_text_config as _tiny_glm5_next_text_config,
)
from mlx_vlm.tests.speculative_fixtures import (
    tiny_qwen_moe_text_config as _tiny_qwen3_5_moe_text_config,
)
from mlx_vlm.tests.speculative_fixtures import (
    tiny_qwen_text_config as _tiny_qwen3_5_text_config,
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


def test_speculative_sampler_rng_async_evals_greedy_draft_call_state(monkeypatch):
    result_array = mx.array([1], dtype=mx.int32)
    state_array = mx.array([2], dtype=mx.int32)
    calls = []

    def fake_async_eval(*arrays):
        calls.append(arrays)

    draft_model = SimpleNamespace(draft_eval_state=lambda: {"cache": [state_array]})
    sampler_rng = _SpeculativeSamplerRNG(draft_model, enabled=False)

    monkeypatch.setattr(mx, "async_eval", fake_async_eval)
    result = sampler_rng.draft_call(lambda: result_array)

    assert result is result_array
    assert len(calls) == 1
    assert calls[0][0] is result_array
    assert calls[0][1] is state_array


def _write_checkpoint(
    path, config, weights, *, shard="model.safetensors", indexed=False
):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(path / shard), weights, metadata={})
    if indexed:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.foo": "model.safetensors"}})
        )
    return path


def _read_checkpoint(path):
    weights = {}
    for shard in sorted(path.glob("model*.safetensors")):
        weights.update(mx.load(str(shard)))
    return json.loads((path / "config.json").read_text()), weights


def _greedy(logits):
    return mx.argmax(logits, axis=-1)


def _make_mtp_drafter(family, *, left_padding=None):
    config_factory, config_class, model_class, block_size = {
        "qwen": (_tiny_qwen3_5_text_config, Qwen3_5MTPConfig, Qwen3_5MTPDraftModel, 3),
        "glm": (
            _tiny_glm5_next_text_config,
            Glm5NextMTPConfig,
            Glm5NextMTPDraftModel,
            2,
        ),
        "deepseek": (
            _tiny_deepseek_v4_config,
            DeepseekV4MTPConfig,
            DeepseekV4MTPDraftModel,
            3,
        ),
    }[family]
    text = config_factory()
    if family == "qwen":
        text.mtp_num_hidden_layers = 1
    drafter = model_class(config_class(text_config=text, block_size=block_size))
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                embed_tokens=nn.Embedding(text.vocab_size, text.hidden_size)
            )
        )
    )
    drafter.reset(
        target, **({"left_padding": left_padding} if left_padding is not None else {})
    )
    return drafter


def _make_drafter_dir(
    tmp_path: Path, model_type: str | None, extra: dict | None = None
) -> Path:
    d = tmp_path / "drafter"
    d.mkdir()
    cfg = {} if model_type is None else {"model_type": model_type}
    if extra:
        cfg.update(extra)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _make_target_config(hidden_size: int):
    return SimpleNamespace(model_type="target_text", hidden_size=hidden_size)


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
        "gemma4_assistant", "backbone_hidden_size", id="gemma4-backbone-hidden-size"
    ),
    pytest.param(
        "gemma4_unified_assistant",
        "backbone_hidden_size",
        id="gemma4-unified-backbone-hidden-size",
    ),
    pytest.param(
        "qwen3_5_mtp", "text_config.hidden_size", id="text-config-hidden-size"
    ),
    pytest.param("custom_mtp", "target_hidden_size", id="target-hidden-size"),
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


def test_arrays_cache_transaction_handles_boundaries_abort_and_stale_commit():
    cache = ArraysCache(size=1)
    initial = mx.arange(6, dtype=mx.float32).reshape(2, 3)
    cache[0] = initial

    transaction = speculative_cache_state.start_speculative_cache([cache], 3)
    intermediate = mx.stack([initial + 1, initial + 2], axis=1)
    final = initial + 3
    cache[0] = final
    cache.record_speculative_states(0, intermediate, final)
    transaction.commit([0, 3])
    assert cache[0].tolist() == [initial[0].tolist(), final[1].tolist()]

    stale = speculative_cache_state.start_speculative_cache([cache], 2)
    current = speculative_cache_state.start_speculative_cache([cache], 2)
    with pytest.raises(RuntimeError, match="stale"):
        stale.commit([1, 1])
    current.abort()
    assert not cache.is_speculating

    missing = speculative_cache_state.start_speculative_cache([cache], 2)
    before = cache[0]
    cache[0] = before + 1
    cache._qwen3_5_lengths_info = (mx.array([3, 3]), 3)
    with pytest.raises(RuntimeError, match="without temporal records"):
        missing.commit([1, 1])
    missing.abort()
    assert cache[0] is before
    qwen_language._qwen3_5_lengths_info(cache)
    assert not hasattr(cache, "_qwen3_5_lengths_info")


@pytest.mark.parametrize("batch", [1, 2])
def test_qwen_gdn_cache_captures_intermediate_states(batch):
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

    def fake_update(
        q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True, state_steps=None
    ):
        del k, v, a, b, A_log, dt_bias, state, mask, use_kernel
        B, S = q.shape[:2]
        state_steps = S if state_steps is None else state_steps
        out = mx.zeros((B, S, 2, 4), dtype=mx.float32)
        next_state = mx.zeros((B, 2, 4, 4), dtype=mx.float32)
        states = mx.ones((B, state_steps, 2, 4, 4), dtype=mx.float32)
        return out, next_state, states

    verifier = qwen_verifier.Qwen3_5ExactSpeculativeVerifier()
    cache = ArraysCache(size=2)
    cache.start_speculation(3)
    with patch.object(
        qwen_gated_delta, "gated_delta_update_with_states", side_effect=fake_update
    ):
        out = verifier._gated_delta(
            layer, mx.zeros((batch, 3, 16), dtype=mx.float32), None, cache
        )

    mx.eval(out)
    assert out.shape == (batch, 3, 16)
    assert cache._speculation["records"][1][1].shape == (batch, 2, 2, 4, 4)
    cache.commit_speculation([2] if batch == 1 else [1, 2])
    assert cache[1].shape == (batch, 2, 4, 4)


def test_qwen_target_verify_linear_matches_singleton_dense_gemv():
    mx.random.seed(7)
    linear = nn.Linear(16, 32, bias=True)
    linear.weight = mx.random.normal((32, 16)).astype(mx.bfloat16)
    linear.bias = mx.random.normal((32,)).astype(mx.bfloat16)
    x = mx.random.normal((3, 4, 16)).astype(mx.bfloat16)

    ref = mx.concatenate(
        [
            mx.concatenate(
                [linear(x[row : row + 1, i : i + 1]) for i in range(x.shape[1])], axis=1
            )
            for row in range(x.shape[0])
        ],
        axis=0,
    )
    out = verifier_linear._target_verify_linear(linear, x)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


@pytest.mark.parametrize("input_dims", [512, 6144])
@pytest.mark.parametrize("verify_length", [3, 4, 6, 7, 8])
def test_qwen_target_verify_4bit_linear_matches_singleton_path_exactly(
    input_dims, verify_length
):
    mx.random.seed(31 + input_dims + verify_length)
    linear = nn.QuantizedLinear(input_dims, 16, bias=False, group_size=64, bits=4)
    _bf16_quantization_parameters(linear)
    x = mx.random.normal((1, verify_length, input_dims)).astype(mx.bfloat16)

    ref = verifier_linear._target_verify_timewise(linear, x)
    out = verifier_linear._target_verify_linear(linear, x)
    mx.eval(ref, out)

    assert bool(mx.array_equal(ref, out).item())


@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("output_dims", [(16, 24), (16, 24, 32), (8, 16, 24, 32)])
@pytest.mark.parametrize("verify_length", [2, 3, 6, 8])
def test_qwen_target_verify_affine_linears_fuse_exactly(
    bits, output_dims, verify_length
):
    mx.random.seed(51 + bits + len(output_dims) + verify_length)
    linears = tuple(
        nn.QuantizedLinear(512, output_dim, bias=False, group_size=64, bits=bits)
        for output_dim in output_dims
    )
    for linear in linears:
        linear.scales = linear.scales.astype(mx.bfloat16)
        linear.biases = linear.biases.astype(mx.bfloat16)
    x = mx.random.normal((1, verify_length, 512)).astype(mx.bfloat16)

    ref = tuple(
        verifier_linear._target_verify_timewise(linear, x) for linear in linears
    )
    out = verifier_linear._target_verify_linears(linears, x)
    mx.eval(*ref, *out)

    assert all(bool(mx.array_equal(a, b).item()) for a, b in zip(ref, out))


def test_qwen_fused_greedy_decode_support_matches_lm_head():
    linear = nn.QuantizedLinear(512, 16, bias=False, group_size=32, bits=4)
    _bf16_quantization_parameters(linear)
    model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=False), lm_head=linear
    )
    assert qwen_verifier._can_target_verify_quantized_head(model.lm_head)

    model.lm_head = OneBitLinear(512, 16, bias=False, group_size=32)
    assert not qwen_verifier._can_target_verify_quantized_head(model.lm_head)
    assert (
        qwen_language.LanguageModel.fused_greedy_decode(
            model, mx.array([[1]], dtype=mx.int32), cache=[]
        )
        is None
    )

    model.lm_head = linear
    model.args.tie_word_embeddings = True
    assert (
        qwen_language.LanguageModel.fused_greedy_decode(
            model, mx.array([[1]], dtype=mx.int32), cache=[]
        )
        is None
    )


def test_qwen_fused_greedy_decode_uses_quantized_argmax():
    mx.random.seed(19)
    hidden = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)

    class Model:
        args = SimpleNamespace(tie_word_embeddings=False)

        def __init__(self):
            self.lm_head = nn.QuantizedLinear(
                512, 16, bias=False, group_size=32, bits=4
            )
            self.lm_head.scales = self.lm_head.scales.astype(mx.bfloat16)
            self.lm_head.biases = self.lm_head.biases.astype(mx.bfloat16)
            self.calls = []

        def __call__(self, inputs, cache=None, **kwargs):
            self.calls.append((inputs.tolist(), cache, kwargs))
            return SimpleNamespace(hidden_states=[hidden])

        def speculative_logits_from_hidden(self, value):
            return self.lm_head(value)

    model = Model()
    inputs = mx.array([[1]], dtype=mx.int32)
    out = qwen_language.LanguageModel.fused_greedy_decode(
        model, inputs, cache=["cache"]
    )
    ref = verifier_linear._target_verify_quantized_argmax(model.lm_head, hidden)
    mx.eval(out, ref)

    assert bool(mx.array_equal(out, ref).item())
    assert model.calls == [
        ([[1]], ["cache"], {"return_hidden": True, "skip_logits": True})
    ]


@pytest.mark.parametrize("verify_length", [6, 7, 8])
def test_qwen3_5_4bit_quantized_argmax_wide_blocks_match_singletons(verify_length):
    mx.random.seed(32 + verify_length)
    linear = nn.QuantizedLinear(512, 32, bias=False, group_size=64, bits=4)
    _bf16_quantization_parameters(linear)
    x = mx.random.normal((1, verify_length, 512), dtype=mx.bfloat16)

    out = verifier_linear._target_verify_quantized_argmax(linear, x)
    mask = mx.full((verify_length, 1), -1, dtype=mx.int32)
    masked = verifier_linear._target_verify_quantized_argmax(linear, x, token_mask=mask)
    ref = mx.argmax(verifier_linear._target_verify_timewise(linear, x), axis=-1)
    mx.eval(out, masked, ref)

    assert bool(mx.array_equal(out, ref).item())
    assert bool(mx.array_equal(masked, ref).item())


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


def test_qwen_exact_verifier_moe_matches_singleton_path():
    mx.random.seed(111)
    config = SimpleNamespace(
        hidden_size=16,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    moe = qwen_moe_language.Qwen3_5MoeSparseMoeBlock(config)
    moe.set_dtype(mx.bfloat16)
    x = mx.random.normal((2, 3, 16)).astype(mx.bfloat16)

    expected = mx.concatenate(
        [moe(x[:, index : index + 1]) for index in range(x.shape[1])], axis=1
    )
    actual = qwen_verifier.Qwen3_5ExactSpeculativeVerifier()._feed_forward(moe, x)
    mx.eval(expected, actual)

    assert bool(mx.array_equal(expected, actual).item())


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


def _qwen3_5_hybrid_batch_model():
    text_config = _tiny_qwen3_5_text_config()
    text_config.num_hidden_layers = 2
    text_config.full_attention_interval = 2
    return qwen_language.Qwen3_5Model(text_config), text_config


def _qwen3_5_batch_cache(left_padding):
    arrays = ArraysCache(size=2)
    arrays.left_padding = mx.array(left_padding, dtype=mx.int32)
    return [arrays, BatchKVCache(list(left_padding))]


def test_qwen3_5_fully_padded_prefill_row_survives_chunks():
    model, text_config = _qwen3_5_hybrid_batch_model()
    cache = _qwen3_5_batch_cache([5, 0])

    first = model(mx.array([[0, 0, 0], [1, 2, 3]], dtype=mx.int32), cache=cache)
    mx.eval(first, cache[1].offset, cache[1].left_padding)
    assert first.shape == (2, 3, text_config.hidden_size)
    assert cache[1].offset.tolist() == [-2, 3]
    assert cache[1].left_padding.tolist() == [5, 0]

    second = model(mx.array([[0, 0, 4], [4, 5, 6]], dtype=mx.int32), cache=cache)
    mx.eval(second, cache[1].offset, cache[1].left_padding)
    assert second.shape == (2, 3, text_config.hidden_size)
    assert cache[1].offset.tolist() == [1, 6]
    assert cache[1].left_padding.tolist() == [5, 0]


def test_qwen3_5_all_rows_fully_padded_prefill():
    model, text_config = _qwen3_5_hybrid_batch_model()
    cache = _qwen3_5_batch_cache([5, 5])
    out = model(mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32), cache=cache)
    mx.eval(out)
    assert out.shape == (2, 3, text_config.hidden_size)


def test_qwen3_5_mtp_verify_supports_native_quantized_batch_cache():
    text_config = _tiny_qwen3_5_text_config()
    text_config.hidden_size = 64
    text_config.intermediate_size = 128
    text_config.num_hidden_layers = 2
    text_config.num_attention_heads = 2
    text_config.num_key_value_heads = 1
    text_config.head_dim = 32
    text_config.full_attention_interval = 2
    model_config = qwen_language.ModelConfig(
        text_config=text_config,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        model_type="qwen3_5",
        image_token_id=101,
        video_token_id=102,
        image_token_index=101,
        video_token_index=102,
        vision_start_token_id=100,
        vision_end_token_id=103,
        vocab_size=text_config.vocab_size,
    )
    model = qwen_language.LanguageModel(text_config, config=model_config)

    arrays = ArraysCache(size=2)
    arrays.left_padding = mx.array([0, 0], dtype=mx.int32)
    quantized = BatchQuantizedKVCache([0, 0], group_size=32, bits=4)
    prompt_cache = [arrays, quantized]

    prompt = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    model(prompt, cache=prompt_cache)
    mx.eval(prompt_cache[1].state)

    verify = mx.array([[7, 8, 9], [10, 11, 12]], dtype=mx.int32)
    hidden, shared_kv, rollback_state = model.speculative_verify_hidden(
        verify, prompt_cache
    )
    mx.eval(hidden, prompt_cache[1].state)

    assert hidden.shape == (2, 3, text_config.hidden_size)
    assert shared_kv == {}
    assert rollback_state is not None
    assert quantized._idx == 6
    assert quantized.offset.tolist() == [6, 6]

    accepted = model.rollback_speculative_cache(
        prompt_cache, rollback_state, mx.array([0, 1]), block_size=3
    )
    mx.eval(prompt_cache[1].state)

    assert accepted == 1
    assert quantized._idx == 5
    assert quantized.offset.tolist() == [4, 5]
    assert quantized.left_padding.tolist() == [1, 0]


def test_speculative_walk_batch_handles_empty_active_batch():
    accepted, new_tokens = _speculative_walk_batch(
        mx.zeros((0, 2), dtype=mx.int32), mx.zeros((0, 3), dtype=mx.int32), budgets=[]
    )

    assert accepted == []
    assert new_tokens == []


@pytest.mark.parametrize("budgets", ([1], [1, 1, 1], [1, -1]))
def test_speculative_walk_batch_rejects_invalid_budgets(budgets):
    drafts = mx.array([[1], [2]], dtype=mx.int32)
    targets = mx.array([[1, 3], [2, 4]], dtype=mx.int32)

    with pytest.raises(ValueError):
        _speculative_walk_batch(drafts, targets, budgets)


def test_speculative_walk_batch_async_stress_matches_python_reference():
    rng = np.random.default_rng(20260904)
    stream = mx.new_stream(mx.default_device())

    for iteration in range(100):
        batch = iteration % 4 + 1
        draft_count = iteration % 7
        drafts_np = rng.integers(0, 64, size=(batch, draft_count), dtype=np.int32)
        targets_np = rng.integers(0, 64, size=(batch, draft_count + 1), dtype=np.int32)
        accepted_expected = []
        tokens_expected = []
        budgets = []
        for row in range(batch):
            accepted = (iteration + row) % (draft_count + 1)
            targets_np[row, :accepted] = drafts_np[row, :accepted]
            if accepted < draft_count:
                targets_np[row, accepted] = (drafts_np[row, accepted] + 1) % 64
            budget = (2 * iteration + row) % (draft_count + 2)
            accepted_expected.append(accepted)
            budgets.append(budget)
            walked = drafts_np[row, :accepted].tolist()
            walked.append(int(targets_np[row, accepted]))
            tokens_expected.append(walked[:budget])

        with mx.stream(stream):
            drafts = mx.array(drafts_np) + mx.array(0, dtype=mx.int32)
            targets = mx.array(targets_np) + mx.array(0, dtype=mx.int32)
            mx.async_eval(drafts, targets)
            accepted_actual, tokens_actual = _speculative_walk_batch(
                drafts, targets, budgets
            )

        assert accepted_actual == accepted_expected
        assert tokens_actual == tokens_expected


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


def test_mtp_drafter_sliding_mask_uses_local_rotating_cache_offset():
    kv = (mx.zeros((1, 1, 8, 4)), mx.zeros((1, 1, 8, 4)))

    masks = make_drafter_masks(
        {"sliding_attention": kv}, query_len=1, query_offset=128, sliding_window=4
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


def test_mtp_target_cache_buffers_cache_list_local_rotating_cache():
    base = RotatingKVCache(max_size=4, keep=0)
    keys = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    base.update_and_fetch(keys, keys)
    prompt_cache = [CacheList(base, PoolingCache(4))]
    draft_model = SimpleNamespace(config=SimpleNamespace(block_size=3))

    mtp_utils._buffer_mtp_target_cache(prompt_cache, draft_model, None)

    assert isinstance(prompt_cache[0][0], BufferedRotatingKVCache)
    assert isinstance(prompt_cache[0][1], PoolingCache)
    assert prompt_cache[0][0].state[0].reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0]


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
        [[[0, 0, 9, 0], [0, 0, 0, 9], [0, 9, 0, 0], [9, 0, 0, 0]]], dtype=mx.float32
    )
    draft_tokens = mx.array([[2, 1, 3]], dtype=mx.int32)
    accepted, new_tokens = _speculative_walk_deferred_greedy(
        lm,
        target_hidden,
        draft_tokens,
        _greedy,
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
        [[[0, 0, 9, 0], [0, 0, 0, 9], [0, 9, 0, 0], [9, 0, 0, 0]]], dtype=mx.float32
    )
    draft_tokens = mx.array([[2, 1, 3]], dtype=mx.int32)
    verify = mtp_utils._MTPVerifyResult(hidden=target_hidden, shared_kv_states={})

    accepted, new_tokens = mtp_utils._mtp_acceptance_walk(
        lm, verify, draft_tokens, sampler, budget=4, row_id=5, base_position=7
    )

    assert accepted == 1
    assert new_tokens == [2, 3]
    assert fake_head.calls == 1
    assert sampler.calls == [([5, 5, 5, 5], [7, 8, 9, 10])]


@pytest.mark.parametrize("uniform", [False, True])
def test_speculative_walk_batch_deferred_uses_positioned_sampler(uniform):
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
            [[0, 0, 9, 0], [0, 9, 0, 0], [0, 0, 0, 9]],
            [[9, 0, 0, 0], [0, 0, 9, 0], [0, 9, 0, 0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[2, 3], [0, 2]], dtype=mx.int32)

    walk = (
        mtp_utils._speculative_walk_batch_deferred_uniform
        if uniform
        else _speculative_walk_batch_deferred_greedy
    )
    accepted, new_tokens = walk(
        lm,
        target_hidden,
        draft_tokens,
        sampler,
        budgets=[3, 2],
        row_ids=[10, 11],
        base_positions=[7, 12],
    )

    assert accepted == ([1, 1] if uniform else [1, 2])
    assert new_tokens == [[2, 1], [0, 2]]
    assert fake_head.calls == (2 if uniform else 3)
    assert (
        sampler.calls
        == [([10, 11], [7, 12]), ([10, 11], [8, 13]), ([10, 11], [9, 14])][
            : fake_head.calls
        ]
    )


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
            [[0, 0, 9, 0], [0, 0, 0, 9], [0, 9, 0, 0]],
            [[0, 9, 0, 0], [9, 0, 0, 0], [0, 0, 9, 0]],
        ],
        dtype=mx.float32,
    )
    draft_tokens = mx.array([[2, 3], [0, 2]], dtype=mx.int32)

    accepted, new_tokens = mtp_utils._speculative_walk_batch_deferred_uniform(
        lm,
        target_hidden,
        draft_tokens,
        _greedy,
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
            prompt_tokens=mx.array([[7, 8]], dtype=mx.int32),
            row_ids=[0],
        )
    )

    assert result == [([3], None)]
    assert calls
    assert calls[0][2]["first_bonus"].tolist() == [2]
    assert calls[0][2]["prompt_tokens"].tolist() == [[7, 8]]
    assert calls[0][2]["row_ids"] == [0]


def test_dflash_server_singleton_dispatches_single_rounds(monkeypatch):
    calls = []

    def fake_single(*args, **kwargs):
        calls.append((args, kwargs))
        yield 3, None
        yield 4, None
        yield 5, None

    def fake_batch(*args, **kwargs):
        raise AssertionError("server DFlash singleton should use single round path")

    monkeypatch.setattr(speculative_utils, "_dflash_rounds", fake_single)
    monkeypatch.setattr(speculative_utils, "_dflash_rounds_batch", fake_batch)

    result = list(
        speculative_utils.run_speculative_server_rounds(
            SimpleNamespace(language_model=SimpleNamespace()),
            SimpleNamespace(),
            prompt_cache=[],
            hidden=mx.zeros((1, 1, 1), dtype=mx.float32),
            draft_kind="dflash",
            first_bonus=mx.array([2], dtype=mx.int32),
            max_tokens=4,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            token_dtype=mx.int32,
            greedy_sampling=True,
            stop_check=lambda _seq_idx, token_id: token_id == 4,
        )
    )

    assert result == [([3], None), ([4], None)]
    assert calls
    assert calls[0][1]["first_bonus"] == 2
    assert "use_model_initial_block_size" not in calls[0][1]


def test_mtp_uses_uniform_deferred_walk_for_batched_sampling():
    ragged_drafter = SimpleNamespace(requires_uniform_batch_acceptance=False)
    uniform_drafter = SimpleNamespace(requires_uniform_batch_acceptance=True)
    normal_sampler = _greedy
    positioned_sampler = SimpleNamespace(sample_target=lambda *args, **kwargs: None)

    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter, n_active=1, greedy_sampling=False, sampler=normal_sampler
    )
    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter, n_active=2, greedy_sampling=True, sampler=normal_sampler
    )
    assert mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter, n_active=2, greedy_sampling=False, sampler=normal_sampler
    )
    assert not mtp_utils._mtp_use_uniform_deferred_walk(
        ragged_drafter, n_active=2, greedy_sampling=False, sampler=positioned_sampler
    )
    assert mtp_utils._mtp_use_uniform_deferred_walk(
        uniform_drafter, n_active=2, greedy_sampling=True, sampler=positioned_sampler
    )


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
        sampler=_greedy,
    )

    assert calls[0][0] is verify_input
    assert calls[0][1] == ["cache"]
    assert result.hidden is hidden
    assert result.shared_kv_states == {"full": ("k", "v")}
    assert result.rollback_state == ["gdn"]
    assert result.target_tokens is target_tokens


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
        rollback_state=gdn_states,
    )

    with (
        patch.object(mtp_utils, "_mtp_verify_target", return_value=verify),
        patch.object(mtp_utils, "_mtp_acceptance_walk", return_value=(2, [7, 8])),
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
                sampler=_greedy,
                draft_block_size=3,
                token_dtype=mx.int32,
                greedy_sampling=True,
            )
        )

    assert rollback_calls == []


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


def test_dflash_next_block_size_starts_at_requested_ceiling():
    draft_model = SimpleNamespace(accept_lens=[], draft_lens=[])

    assert _dflash_next_block_size(draft_model, 16, 20) == 16


def test_dflash_next_block_size_backs_off_on_low_acceptance():
    draft_model = SimpleNamespace(accept_lens=[1, 2], draft_lens=[15, 7])

    assert _dflash_next_block_size(draft_model, 16, 20) == 4


def test_dflash_next_block_size_does_not_grow_on_middling_acceptance():
    draft_model = SimpleNamespace(accept_lens=[3, 2, 1, 3], draft_lens=[3, 3, 3, 3])

    assert _dflash_next_block_size(draft_model, 16, 20) == 4


def test_dflash_greedy_verify_prefers_hidden_argmax_hook():
    captured = [mx.ones((1, 3, 2))]
    final_hidden = mx.zeros((1, 3, 2))
    target_tokens = mx.array([[4, 5, 6]])

    class LM:
        def speculative_verify_dflash_hidden(self, inputs, cache, layer_ids):
            assert inputs.shape == (1, 3)
            assert cache == ["cache"]
            assert layer_ids == [1]
            return captured, final_hidden, ["gdn"]

        def speculative_argmax_from_hidden(self, hidden):
            assert hidden is final_hidden
            return target_tokens

        def __call__(self, *args, **kwargs):
            raise AssertionError("full target logits should not be materialized")

    actual_captured, gdn_states, actual_tokens = _dflash_verify_greedy(
        LM(),
        mx.array([[1, 2, 3]]),
        ["cache"],
        [1],
        _greedy,
    )

    assert actual_captured is captured
    assert gdn_states == ["gdn"]
    assert actual_tokens is target_tokens


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
            "dflash_config": {"target_layer_ids": [1, 6, 11], "runtime_block_size": 12},
        }
    )

    assert config.target_layer_ids == [1, 6, 11]
    assert config.runtime_block_size == 12


def test_generic_dflash_config_infers_target_depth_and_parses_rope_parameters():
    config = ModelConfig.from_dict(
        {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 8,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "rope_type": "yarn",
                "factor": 8.0,
            },
            "dflash_config": {"target_layer_ids": [1, 5], "mask_token_id": 4},
        }
    )

    assert config.num_target_layers == 6
    assert config.rope_theta == 10000.0
    assert config.rope_scaling == {"rope_type": "yarn", "factor": 8.0}


def test_dflash_sanitize_installs_checkpoint_embedding():
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
    weight = mx.zeros((8, 4))

    sanitized = drafter.sanitize(
        {"model.embed_tokens.weight": weight, "model.norm.weight": mx.ones((4,))}
    )

    assert sanitized["embed_tokens.weight"] is weight
    assert drafter.embed_tokens is not None
    assert sanitized["norm.weight"] is not None


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


def test_dflash_drafter_binds_backbone_embeddings():
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
    embeddings = nn.Embedding(8, 4)
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            backbone=SimpleNamespace(embeddings=embeddings),
            lm_head=nn.Linear(4, 8, bias=False),
        )
    )

    drafter.bind(target)

    assert drafter.embed_tokens is embeddings


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
            self, last_bonus, hidden, cache, block_size, sampler, token_dtype
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


def test_speculative_walk_batch_uniform_acceptance_keeps_exact_tokens():
    draft_tokens = mx.array([[10, 11, 12], [20, 21, 22]], dtype=mx.int32)
    target_tokens = mx.array([[10, 99, 98, 97], [20, 21, 77, 76]], dtype=mx.int32)
    accepted, new_tokens = _speculative_walk_batch_uniform_acceptance(
        draft_tokens, target_tokens, accepted_list=[1, 2], budgets=[4, 4]
    )

    assert accepted == [1, 1]
    assert new_tokens == [[10, 99], [20, 21]]


def test_missing_config_keeps_caller_kind(tmp_path):
    d = tmp_path / "drafter"
    d.mkdir()
    assert resolve_drafter_kind(d, "dflash") == "dflash"


def test_kind_table_only_uses_known_kinds():
    for mt, kind in DRAFTER_KIND_BY_MODEL_TYPE.items():
        assert kind in KNOWN_DRAFTER_KINDS, f"{mt} maps to unknown kind {kind}"


def test_native_nextn_overrides_dflash_to_mtp(tmp_path, caplog):
    path = _make_drafter_dir(tmp_path, "deepseek_v4", {"num_nextn_predict_layers": 1})
    with caplog.at_level("WARNING"):
        assert resolve_drafter_kind(path, "dflash") == "mtp"
    assert any("requires --draft-kind='mtp'" in r.getMessage() for r in caplog.records)


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


def test_eagle3_config_uses_speculators_fields():
    cfg = Eagle3Config.from_dict(
        {
            "speculators_model_type": "eagle3",
            "eagle_aux_hidden_state_layer_ids": [2, 30, 57],
            "speculators_config": {"proposal_methods": [{"speculative_tokens": 3}]},
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
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [3.5, -0.5]],
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
        sampler=_greedy,
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


def test_qwen3_5_moe_mtp_builds_moe_layer_and_sanitizes_both_expert_layouts():
    num_experts, moe_inter, hidden = 4, 8, 16
    drafter = Qwen3_5MTPDraftModel(
        Qwen3_5MTPConfig(
            text_config=_tiny_qwen3_5_moe_text_config(
                num_experts=num_experts, moe_intermediate_size=moe_inter
            ),
            block_size=3,
        )
    )
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    self_mlp = drafter.layers[0].mlp
    assert isinstance(self_mlp, Qwen3_5MoeSparseMoeBlock)
    assert hasattr(self_mlp, "switch_mlp")

    p = "mtp.layers.0.mlp"
    split = {}
    for e in range(num_experts):
        split[f"{p}.experts.{e}.gate_proj.weight"] = mx.ones((moe_inter, hidden))
        split[f"{p}.experts.{e}.up_proj.weight"] = mx.ones((moe_inter, hidden))
        split[f"{p}.experts.{e}.down_proj.weight"] = mx.ones((hidden, moe_inter))
    fused = {
        f"{p}.experts.gate_up_proj": mx.ones((num_experts, 2 * moe_inter, hidden)),
        f"{p}.experts.down_proj": mx.ones((num_experts, hidden, moe_inter)),
    }

    for label, weights in (("split", split), ("fused", fused)):
        out = drafter.sanitize(dict(weights))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"layers.0.mlp.switch_mlp.{proj}.weight"
            assert key in out, f"[{label}] missing {key}"
            assert out[key].shape[0] == num_experts, f"[{label}] {key} not stacked"
        assert not any(
            ".experts." in k and "switch_mlp" not in k for k in out
        ), f"[{label}] raw expert keys leaked"


@pytest.mark.parametrize(
    "family,position,block_size", [("qwen", 3, 3), ("glm", 4, 2)], ids=["qwen", "glm"]
)
def test_mtp_draft_block_smoke(family, position, block_size):
    drafter = _make_mtp_drafter(family)
    drafter.set_shared_kv({}, kv_offset=4, position=position, kv_valid_len=4)
    tokens = drafter.draft_block(
        7,
        mx.zeros((1, 1, 16)),
        None,
        block_size,
        _greedy,
        mx.int32,
        **({"greedy": True} if family == "glm" else {}),
    )
    mx.eval(tokens)
    assert tokens.shape == (1, block_size - 1)
    if family == "glm":
        assert drafter.config.runtime_block_size is None
        assert mtp_utils._dflash_block_total(drafter, None) == 3
        assert not drafter.prefer_requested_block_size


def test_qwen3_5_mtp_batch_accept_updates_ragged_cache():
    drafter = _make_mtp_drafter("qwen", left_padding=[0, 0])
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
        _greedy,
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[1, 0],
        new_tokens=[[int(draft_tokens[0, 0].item()), 5], [6]],
        sampler=_greedy,
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


@pytest.mark.parametrize("family", ["gemma4", "deepseek_v4"])
def test_ragged_turboquant_rollback_requires_uniform_acceptance(family):
    cache = BatchTurboQuantKVCache([0, 0], bits=3.5)
    keys = mx.arange(2 * 1 * 5 * 8, dtype=mx.float32).reshape(2, 1, 5, 8)
    cache.update_and_fetch(keys, keys + 100)
    with pytest.raises(RuntimeError, match="uniform"):
        if family == "gemma4":
            gemma4_language.LanguageModel.rollback_speculative_cache(
                None, [cache], [], mx.array([0, 2]), block_size=3
            )
        else:
            speculative_cache_state.rollback_speculative_cache(
                [cache], None, mx.array([0, 2]), block_size=3
            )


def test_uniform_turboquant_batch_rollback_trims_without_raising():
    # With uniform per-row acceptance there is no ragged tail: rollback is a
    # plain uniform trim, no phantom keys, no raise (issue #1962 fix).
    for lm_cls, block in (
        (qwen_language.LanguageModel, 5),
        (gemma4_language.LanguageModel, 5),
        (None, 5),
    ):
        cache = BatchTurboQuantKVCache([0, 0], bits=3.5)
        keys = mx.arange(2 * 1 * 7 * 8, dtype=mx.float32).reshape(2, 1, 7, 8)
        cache.update_and_fetch(keys, keys + 100)

        if lm_cls is None:
            max_a = speculative_cache_state.rollback_speculative_cache(
                [cache], None, mx.array([2, 2]), block_size=block
            )
        else:
            max_a = lm_cls.rollback_speculative_cache(
                None, [cache], [], mx.array([2, 2]), block_size=block
            )
        mx.eval(cache.offset)
        assert max_a == 2
        assert cache._idx == 5


def test_uniform_batch_acceptance_flag_advertised_on_affected_models():
    import mlx_vlm.models.minimax_m3_vl.language as minimax_language
    import mlx_vlm.models.muse_glimmer.language as muse_glimmer_language

    for module in (
        gemma4_language,
        deepseek_language,
        qwen_language,
        muse_glimmer_language,
        minimax_language,
    ):
        assert module.LanguageModel.requires_uniform_batch_acceptance is True


def test_qwen3_5_mtp_filter_batch_keeps_drafter_state_aligned():
    drafter = _make_mtp_drafter("qwen")
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
        _greedy,
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[1, 1],
        new_tokens=[[3, 5], [4, 6]],
        sampler=_greedy,
        token_dtype=mx.int32,
        greedy=True,
    )

    drafter.filter_batch(mx.array([1], dtype=mx.int32))
    mx.eval(drafter._cache[0].keys, drafter._seed_token)
    assert drafter._cache[0].keys.shape[0] == 1
    assert drafter._seed_token.shape == (1, 1)
    assert drafter._next_position.tolist() == [6]


def test_qwen3_5_mtp_filter_batch_keeps_batch_cache_padding_aligned():
    drafter = _make_mtp_drafter("qwen", left_padding=[0, 1, 2])
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


def test_split_qwen3_5_mtp_converts_fine_grained_fp8(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    text_config = _tiny_qwen3_5_text_config()
    text_config.mtp_num_hidden_layers = 1
    _write_checkpoint(
        source,
        {
            "model_type": "qwen3_5",
            "text_config": text_config.to_dict(),
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        },
        {
            "mtp.layers.0.mlp.down_proj.weight": mx.to_fp8(
                mx.ones((128, 128), dtype=mx.bfloat16)
            ),
            "mtp.layers.0.mlp.down_proj.weight_scale_inv": mx.full(
                (1, 1), 0.125, dtype=mx.bfloat16
            ),
            "mtp.pre_fc_norm_hidden.weight": mx.zeros((16,)),
        },
        shard="mtp.safetensors",
    )

    split_qwen3_5_mtp(str(source), str(output))

    cfg, weights = _read_checkpoint(output)
    expected = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    assert cfg["quantization"] == expected
    assert cfg["quantization_config"] == expected
    assert weights["layers.0.mlp.down_proj.weight"].dtype == mx.uint32
    assert weights["layers.0.mlp.down_proj.scales"].dtype == mx.uint8
    assert not any(key.endswith("weight_scale_inv") for key in weights)


def test_deepseek_v4_returns_mtp_hidden_and_trims_without_snapshot():
    cfg = _tiny_deepseek_v4_config()
    lm = deepseek_language.LanguageModel(cfg)
    cache = lm.make_cache()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

    result = _mtp_verify_target(lm, inputs, cache, None, sample_target_tokens=False)
    hidden, shared_kv, rollback_state = (
        result.hidden,
        result.shared_kv_states,
        result.rollback_state,
    )
    mx.eval(hidden)

    assert hidden.shape == (1, 3, cfg.hc_mult, cfg.hidden_size)
    assert shared_kv == {}
    assert rollback_state.active
    assert cache[0].offset == 3

    speculative_cache_state.commit_speculative_round(
        lm, cache, rollback_state, accepted=0, block_size=3
    )
    assert cache[0].offset == 1

    logits = lm.logits_from_hidden(hidden[:, :1])
    mx.eval(logits)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_deepseek_v4_local_mask_aligns_to_layer_cache_width():
    mask = mx.array([[[[False, True, True, False, True]]]], dtype=mx.bool_)

    trimmed = deepseek_language._align_local_mask(mask, 3)
    padded = deepseek_language._align_local_mask(trimmed, 5)

    assert trimmed.tolist() == [[[[True, False, True]]]]
    assert padded.tolist() == [[[[True, True, True, False, True]]]]


def test_glm5_next_mtp_owns_left_padded_prefill(monkeypatch):
    text_config = _tiny_glm5_next_text_config()
    drafter = Glm5NextMTPDraftModel(
        Glm5NextMTPConfig(text_config=text_config, block_size=2)
    )
    captured = {}

    def forward_tokens(self, tokens, hidden, token_dtype):
        del token_dtype
        captured["start"] = self._next_position
        self._next_position = self._next_position + tokens.shape[1]
        return hidden, hidden

    monkeypatch.setattr(Glm5NextMTPDraftModel, "_forward_tokens", forward_tokens)
    monkeypatch.setattr(
        Glm5NextMTPDraftModel,
        "_set_seed_from_hidden",
        lambda self, hidden, sampler, greedy: None,
    )
    drafter.prefill_from_target_hidden(
        mx.array([[0, 0, 1, 2], [1, 2, 3, 4]], dtype=mx.int32),
        mx.zeros((2, 4, text_config.hidden_size)),
        mx.array([3, 5], dtype=mx.int32),
        _greedy,
        left_padding=[2, 0],
    )
    mx.eval(captured["start"], drafter._next_position)

    assert (
        "left_padding"
        not in inspect.signature(
            DeepseekV4MTPDraftModel.prefill_from_target_hidden
        ).parameters
    )
    assert captured["start"].tolist() == [-2, 0]
    assert drafter._next_position.tolist() == [2, 4]


def test_glm5_next_mtp_last_only_commit_preserves_cache_and_final_output():
    text_config = _tiny_glm5_next_text_config()
    drafter = Glm5NextMTPDraftModel(
        Glm5NextMTPConfig(text_config=text_config, block_size=2)
    )
    drafter.apply(
        lambda value: (
            value.astype(mx.bfloat16)
            if isinstance(value, mx.array) and value.dtype == mx.float32
            else value
        )
    )

    full_cache = drafter.make_cache()[0]
    last_cache = drafter.make_cache()[0]
    inputs = mx.arange(2 * text_config.hidden_size, dtype=mx.bfloat16).reshape(
        1, 2, text_config.hidden_size
    )

    full_output = drafter.mtp_block(inputs, cache=full_cache)
    last_output = drafter.mtp_block(inputs, cache=last_cache, last_only=True)
    mx.eval(full_output, last_output, full_cache.state, last_cache.state)

    assert full_output.shape == (1, 2, text_config.hidden_size)
    assert last_output.shape == (1, 1, text_config.hidden_size)
    assert mx.array_equal(full_output[:, -1:], last_output).item()

    for full_subcache, last_subcache in zip(
        full_cache.caches, last_cache.caches, strict=True
    ):
        assert full_subcache.meta_state == last_subcache.meta_state
        for (_, full_value), (_, last_value) in zip(
            tree_flatten(full_subcache.state),
            tree_flatten(last_subcache.state),
            strict=True,
        ):
            if full_value is None or last_value is None:
                assert full_value is last_value
            else:
                assert mx.array_equal(full_value, last_value).item()


@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 9, 16, 32, 64])
@pytest.mark.parametrize("length", [2, 4, 6])
def test_glm5_next_dense_verifier_matches_batched_decode(batch, length):
    mx.random.seed(90 + batch)
    linear = nn.Linear(512, 32, bias=False)
    linear.weight = linear.weight.astype(mx.bfloat16)
    inputs = mx.random.normal((batch, length, 512)).astype(mx.bfloat16)
    expected = _decode_reference(linear, inputs)

    actual = native_batch_linear(linear, inputs)
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


def _quantized_formats(group_sizes=(64,)):
    return [
        ("affine", bits, size) for size in group_sizes for bits in (2, 3, 4, 5, 6, 8)
    ] + [("mxfp4", 4, 32), ("mxfp8", 8, 32), ("nvfp4", 4, 16)]


def _quantize_bf16_linear(mode, bits, group_size):
    dense = nn.Linear(512, 16, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(
        dense, group_size=group_size, bits=bits, mode=mode
    )


def _decode_reference(linear, inputs):
    return mx.concatenate(
        [
            linear(mx.contiguous(inputs[:, position : position + 1]))
            for position in range(inputs.shape[1])
        ],
        axis=1,
    )


def _bf16_quantization_parameters(linear):
    linear.scales = linear.scales.astype(mx.bfloat16)
    if linear.biases is not None:
        linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
def test_glm5_next_affine_gate_up_fusion_matches_batched_decode(bits, batch):
    mx.random.seed(400 + bits + batch)
    switch = SimpleNamespace(
        up_proj=_bf16_quantization_parameters(
            QuantizedSwitchLinear(512, 16, 4, False, 64, bits)
        ),
        gate_proj=_bf16_quantization_parameters(
            QuantizedSwitchLinear(512, 16, 4, False, 64, bits)
        ),
    )
    inputs = mx.random.normal((batch, 2, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 4, dtype=mx.int32).reshape(batch, 2, 2) % 4
    expected_up = exact_quantized_switch_linear(switch.up_proj, inputs, indices)
    expected_gate = exact_quantized_switch_linear(switch.gate_proj, inputs, indices)

    actual = fast_ops.exact_affine_switch_gate_up(switch, inputs, indices)
    mx.eval(expected_up, expected_gate, *actual)

    assert actual is not None
    assert mx.array_equal(actual[0], expected_up).item()
    assert mx.array_equal(actual[1], expected_gate).item()


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64])
def test_glm5_next_affine_moe_fusion_matches_batched_decode(bits, batch):
    mx.random.seed(500 + bits + batch)
    routed_linear = _bf16_quantization_parameters(
        QuantizedSwitchLinear(512, 16, 4, False, 64, bits)
    )
    dense = nn.Linear(512, 16, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    shared_linear = _bf16_quantization_parameters(
        nn.QuantizedLinear.from_linear(dense, group_size=64, bits=bits, mode="affine")
    )
    routed_inputs = mx.random.normal((batch, 2, 2, 512)).astype(mx.bfloat16)
    shared_inputs = mx.random.normal((batch, 2, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 4, dtype=mx.int32).reshape(batch, 2, 2) % 4
    weights = mx.softmax(mx.random.normal((batch, 2, 2)), axis=-1)
    routed = exact_quantized_selected_linear(routed_linear, routed_inputs, indices)
    shared = exact_quantized_linear(shared_linear, shared_inputs)
    expected = SwitchGLU._combine(routed, weights, shared)

    actual = fast_ops.exact_affine_moe_down(
        routed_linear, routed_inputs, indices, weights, shared
    )
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    _quantized_formats((32, 64, 128)),
)
@pytest.mark.parametrize("batch", [1, 4, 8, 64, 127])
def test_general_quantized_moe_hc_matches_separate_kernels(
    mode, bits, group_size, batch
):
    mx.random.seed(600 + bits + batch)
    routed_linear = QuantizedSwitchLinear(
        512, 16, 4, False, group_size, bits, mode=mode
    )
    if mode == "affine":
        routed_linear = _bf16_quantization_parameters(routed_linear)
    routed_inputs = mx.random.normal((batch, 2, 2, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 4, dtype=mx.int32).reshape(batch, 2, 2) % 4
    weights = mx.softmax(mx.random.normal((batch, 2, 2)), axis=-1)
    shared = mx.random.normal((batch, 2, 16)).astype(mx.bfloat16)
    residual = mx.random.normal((batch, 2, 4, 16)).astype(mx.bfloat16)
    post = mx.random.normal((batch, 2, 4))
    comb = mx.random.normal((batch, 2, 4, 4))

    routed = exact_quantized_selected_linear(routed_linear, routed_inputs, indices)
    collapsed = SwitchGLU._combine(routed, weights, shared)
    expected = fast_ops.exact_hc_expand(collapsed, residual, post, comb)
    with patch(
        "mlx_vlm.models.quantized_verifier.exact_quantized_selected_linear",
        side_effect=AssertionError("supported formats must use the fused backend"),
    ):
        actual = exact_quantized_moe_hc_expand(
            routed_linear, routed_inputs, indices, weights, shared, residual, post, comb
        )
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    _quantized_formats(),
)
@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 9, 16, 32, 64, 127])
def test_general_quantized_verifier_matches_decode(mode, bits, group_size, batch):
    mx.random.seed(100 + bits + batch)
    linear = _quantize_bf16_linear(mode, bits, group_size)
    inputs = mx.random.normal((batch, 3, 512)).astype(mx.bfloat16)
    if mode == "nvfp4":
        expected = verifier_linear._target_verify_singletons(linear, inputs)
    else:
        expected = _decode_reference(linear, inputs)

    native_reference = _decode_reference(linear, inputs)
    assert mx.array_equal(native_batch_linear(linear, inputs), native_reference).item()
    actual = decode_quantized_linear(linear, inputs)
    tokens = decode_quantized_argmax(linear, inputs)
    mx.eval(expected, actual, tokens)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(tokens, mx.argmax(expected, axis=-1)).item()


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    _quantized_formats(),
)
def test_general_quantized_argmax_supports_packed_mask(mode, bits, group_size):
    mx.random.seed(400 + bits)
    linear = _quantize_bf16_linear(mode, bits, group_size)
    inputs = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    allowed = mx.array([[1, 3, 5], [7, 9, 11]], dtype=mx.int32)
    token_mask = (mx.array(1, dtype=mx.int32) << allowed).reshape(-1, 1)

    actual = decode_quantized_argmax(linear, inputs, token_mask=token_mask)
    mx.eval(actual)

    assert actual is not None
    assert mx.array_equal(actual, allowed).item()


def test_glm5_next_mtp_sampler_state_tolerates_uninitialized_kv_cache():
    drafter = _make_mtp_drafter("glm")

    sampler_rng = _SpeculativeSamplerRNG(drafter, enabled=False)
    assert sampler_rng.draft_call(lambda: None) is None


def test_glm5_next_mtp_sanitize_fuses_native_layer_weights():
    config = _tiny_glm5_next_text_config()
    context = SimpleNamespace(args=config)
    weights = {
        "mtp_block.mlp.shared_experts.gate_proj.weight": mx.zeros((8, 16)),
        "mtp_block.mlp.shared_experts.up_proj.weight": mx.zeros((8, 16)),
        "mtp_block.self_attn.q_a_proj.weight": mx.zeros((8, 16)),
        "mtp_block.self_attn.kv_a_proj_with_mqa.weight": mx.zeros((4, 16)),
        "mtp_block.self_attn.kv_b_proj.weight": mx.zeros((16, 4)),
    }
    for expert in range(config.n_routed_experts):
        weights[f"mtp_block.mlp.experts.{expert}.gate_proj.weight"] = mx.zeros((8, 16))
        weights[f"mtp_block.mlp.experts.{expert}.up_proj.weight"] = mx.zeros((8, 16))
        weights[f"mtp_block.mlp.experts.{expert}.down_proj.weight"] = mx.zeros((16, 8))

    out = Glm5NextMTPDraftModel.sanitize(context, weights)

    assert out["mtp_block.mlp.shared_experts.gate_up_proj.weight"].shape == (16, 16)
    assert out["mtp_block.mlp.switch_mlp.gate_proj.weight"].shape == (2, 8, 16)
    assert out["mtp_block.self_attn.qkv_a_proj.weight"].shape == (12, 16)
    assert out["mtp_block.self_attn.embed_q.weight"].shape == (2, 4, 4)
    assert out["mtp_block.self_attn.unembed_out.weight"].shape == (2, 4, 4)


def test_split_glm5_next_mtp_extracts_layer_after_target_stack(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    text_config = _tiny_glm5_next_text_config()
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    _write_checkpoint(
        source,
        {"model_type": "glm5_next", "text_config": text_config.to_dict()},
        {
            f"{prefix}.enorm.weight": mx.ones((16,)),
            f"{prefix}.hnorm.weight": mx.ones((16,)),
            f"{prefix}.eh_proj.weight": mx.ones((16, 32)),
            f"{prefix}.shared_head.norm.weight": mx.ones((16,)),
        },
    )

    split_glm5_next_mtp(str(source), str(output))

    config, weights = _read_checkpoint(output)
    assert config["model_type"] == "glm5_next_mtp"
    assert config["block_size"] == 2
    assert set(weights) == {
        "eh_proj.weight",
        "enorm.weight",
        "hnorm.weight",
        "shared_head_norm.weight",
    }


def test_split_glm5_next_mtp_honors_requested_quantization_for_fp8(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    text_config = _tiny_glm5_next_text_config()
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    _write_checkpoint(
        source,
        {
            "model_type": "glm5_next",
            "text_config": text_config.to_dict(),
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        },
        {
            f"{prefix}.eh_proj.weight": mx.to_fp8(
                mx.ones((128, 128), dtype=mx.bfloat16)
            ),
            f"{prefix}.eh_proj.weight_scale_inv": mx.full(
                (1, 1), 0.125, dtype=mx.bfloat16
            ),
        },
    )

    split_mtp(str(source), str(output), q_bits=4, q_group_size=64)

    config, weights = _read_checkpoint(output)
    expected = {"group_size": 64, "bits": 4, "mode": "affine"}
    assert config["quantization"] == expected
    assert config["quantization_config"] == expected
    assert weights["eh_proj.weight"].dtype == mx.uint32
    assert weights["eh_proj.scales"].dtype == mx.bfloat16
    assert weights["eh_proj.biases"].dtype == mx.bfloat16
    assert not any(key.endswith("weight_scale_inv") for key in weights)


def test_split_glm5_next_mtp_supports_independent_mxfp8_quantization(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    text_config = _tiny_glm5_next_text_config()
    prefix = f"model.language_model.layers.{text_config.num_hidden_layers}"
    _write_checkpoint(
        source,
        {"model_type": "glm5_next", "text_config": text_config.to_dict()},
        {f"{prefix}.eh_proj.weight": mx.ones((128, 128), dtype=mx.bfloat16)},
    )

    split_mtp(str(source), str(output), q_mode="mxfp8")

    config, weights = _read_checkpoint(output)
    expected = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    assert config["quantization"] == expected
    assert config["quantization_config"] == expected
    assert weights["eh_proj.weight"].dtype == mx.uint32
    assert weights["eh_proj.scales"].dtype == mx.uint8
    assert "eh_proj.biases" not in weights


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
    drafter = _make_mtp_drafter("deepseek")
    drafter.set_shared_kv({}, kv_offset=4, position=3, kv_valid_len=4)
    text_config = drafter.config.text_config
    hidden = mx.zeros((2, 1, text_config.hc_mult, 16), dtype=mx.float32)
    draft_tokens = drafter.draft_block(
        mx.array([7, 8], dtype=mx.int32),
        hidden,
        None,
        3,
        _greedy,
        mx.int32,
        greedy=True,
    )
    verify_hidden = mx.zeros((2, 3, text_config.hc_mult, 16), dtype=mx.float32)
    drafter.accept_verified_tokens_batch(
        verify_hidden,
        draft_tokens,
        accepted=[0, 0],
        new_tokens=[[3], [4]],
        sampler=_greedy,
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


def _tiny_deepseek_v4_dspark_config():
    return DeepseekV4DsparkConfig(
        text_config=_tiny_deepseek_v4_config(),
        n_mtp_layers=3,
        target_layer_ids=[0, 1, 2],
        mask_token_id=1,
        markov_rank=8,
        block_size=5,
    )


def _forge_dspark_source(model, cfg):
    """Reconstruct the ``mtp.<stage>.*`` checkpoint tensors from the drafter's
    own params, so the split->load round-trip can be exercised without the real
    (multi-GB) DeepSeek-V4-Flash-0731 checkpoint."""
    params = dict(tree_flatten(model.parameters()))
    text = cfg.text_config
    n_experts, o_groups, o_lora_rank = (
        text.n_routed_experts,
        text.o_groups,
        text.o_lora_rank,
    )
    last_stage = cfg.n_mtp_layers - 1
    proj_to_w = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
    hc = {"attn_hc": "hc_attn", "ffn_hc": "hc_ffn"}
    src = {}
    for key, value in params.items():
        if key.startswith("markov_head."):
            # the model-level markov head lives under the last stage on disk
            src[f"mtp.{last_stage}.{key}"] = value
            continue
        _, stage, body = key.split(".", 2)
        prefix = f"mtp.{stage}."
        if body.startswith("ffn.switch_mlp."):
            w = proj_to_w[body.split(".")[-2]]
            for expert in range(n_experts):
                src[f"{prefix}ffn.experts.{expert}.{w}.weight"] = value[expert]
        elif body.startswith("ffn.shared_experts."):
            w = proj_to_w[body.split(".")[-2]]
            src[f"{prefix}ffn.shared_experts.{w}.weight"] = value
        elif body == "ffn.gate.e_score_correction_bias":
            src[f"{prefix}ffn.gate.bias"] = value
        elif body == "attn.wo_a.weight":
            src[f"{prefix}attn.wo_a.weight"] = (
                value.reshape(o_groups * o_lora_rank, -1) if value.ndim == 3 else value
            )
        elif body.startswith("attn_hc.") or body.startswith("ffn_hc."):
            module, param = body.split(".")
            src[f"{prefix}{hc[module]}_{param}"] = value
        elif body.startswith("hc_head."):
            src[f"{prefix}hc_head_{body.split('.')[-1]}"] = value
        else:
            src[f"{prefix}{body}"] = value
    return src


def test_deepseek_v4_dspark_sanitize_round_trips_three_stage_layout():
    cfg = _tiny_deepseek_v4_dspark_config()
    model = DeepseekV4DsparkDraftModel(cfg)
    mx.eval(model.parameters())
    src = _forge_dspark_source(model, cfg)
    # a trained checkpoint also ships an (unused) confidence head -> dropped
    src["mtp.2.confidence_head.proj.weight"] = mx.zeros(
        (1, cfg.hidden_size + cfg.markov_rank)
    )
    src["mtp.0.ffn.gate.bias_vl"] = mx.zeros((cfg.text_config.n_routed_experts,))

    written = DeepseekV4DsparkDraftModel.sanitize(
        SimpleNamespace(args=cfg.text_config), dict(src)
    )
    fresh = DeepseekV4DsparkDraftModel(cfg)
    mx.eval(fresh.parameters())
    sanitized = fresh.sanitize(dict(written))
    fresh.load_weights(list(sanitized.items()), strict=True)

    assert not any(key.startswith("mtp.") for key in sanitized)
    assert not any("confidence_head" in key for key in sanitized)
    assert not any("bias_vl" in key for key in sanitized)
    assert any(key.startswith("stages.0.main_proj") for key in sanitized)
    assert any(key.startswith("markov_head.") for key in sanitized)


def test_deepseek_v4_dspark_draft_block_emits_proposal_tokens():
    cfg = _tiny_deepseek_v4_dspark_config()
    model = DeepseekV4DsparkDraftModel(cfg)
    mx.eval(model.parameters())
    text = cfg.text_config
    target = SimpleNamespace(
        embed_tokens=nn.Embedding(text.vocab_size, text.hidden_size),
        lm_head=nn.Linear(text.hidden_size, text.vocab_size, bias=False),
    )
    mx.eval(target.embed_tokens.parameters(), target.lm_head.parameters())
    draft_cache = model.reset(target)

    ctx = 5
    n_targets = len(cfg.target_layer_ids)
    hidden = mx.random.normal((1, ctx, text.hidden_size * n_targets))
    drafts = model.draft_block(
        3,
        hidden,
        draft_cache,
        cfg.block_size,
        _greedy,
    )
    assert drafts.shape == (1, cfg.block_size - 1)


def test_split_deepseek_v4_dspark_writes_dspark_config(tmp_path):
    cfg = _tiny_deepseek_v4_dspark_config()
    model = DeepseekV4DsparkDraftModel(cfg)
    mx.eval(model.parameters())
    src_weights = _forge_dspark_source(model, cfg)

    source = tmp_path / "source"
    text = cfg.text_config
    _write_checkpoint(
        source,
        {
            "model_type": "deepseek_v4",
            **text.to_dict(),
            "dspark_block_size": cfg.block_size - 1,
            "dspark_noise_token_id": cfg.mask_token_id,
            "dspark_target_layer_ids": cfg.target_layer_ids,
            "dspark_markov_rank": cfg.markov_rank,
        },
        src_weights,
        shard="mtp.safetensors",
        indexed=True,
    )

    output = tmp_path / "dspark"
    from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter

    assert type(detect_mtp_splitter(source)).__name__ == "DeepseekV4DsparkSplitter"
    split_deepseek_v4_dspark(str(source), str(output))

    written_cfg, weights = _read_checkpoint(output)
    assert written_cfg["model_type"] == "deepseek_v4_dspark"
    assert written_cfg["n_mtp_layers"] == 3
    assert written_cfg["target_layer_ids"] == cfg.target_layer_ids
    assert written_cfg["mask_token_id"] == cfg.mask_token_id
    assert len(list(output.glob("model-*.safetensors"))) == 3
    assert (output / "model.safetensors.index.json").exists()
    assert any(key.startswith("stages.0.main_proj") for key in weights)
    assert any(key.startswith("markov_head.") for key in weights)


def test_deepseek_v4_dspark_uses_dflash_rounds_losslessly():
    # deepseek_v4_dspark maps to the dflash kind. Losslessness comes from
    # verification, not drafter quality: with a random (untrained) drafter every
    # proposal is rejected, so speculative decoding must reproduce the target's
    # plain-greedy trajectory exactly.
    from mlx_vlm.speculative.dflash import _dflash_rounds

    base = _tiny_deepseek_v4_config()
    tgt_cfg = deepseek_language.ModelConfig(
        **{**base.to_dict(), "num_hidden_layers": 4, "compress_ratios": [0, 0, 0, 0]}
    )
    target = deepseek_language.LanguageModel(tgt_cfg)
    mx.eval(target.parameters())

    dcfg = DeepseekV4DsparkConfig(
        text_config=tgt_cfg,
        n_mtp_layers=3,
        target_layer_ids=[1, 2, 3],
        mask_token_id=1,
        markov_rank=8,
        block_size=5,
    )
    drafter = DeepseekV4DsparkDraftModel(dcfg)
    mx.eval(drafter.parameters())

    prompt = mx.array([[5, 6, 7, 8, 9]])
    n_new = 12

    def argmax_last(logits):
        return int(mx.argmax(logits[:, -1, :], axis=-1).item())

    cache_ref = target.make_cache()
    tok = argmax_last(target(prompt, cache=cache_ref).logits)
    ref = [tok]
    for _ in range(n_new - 1):
        tok = argmax_last(target(mx.array([[tok]]), cache=cache_ref).logits)
        ref.append(tok)

    cache_spec = target.make_cache()
    out = target(prompt, cache=cache_spec, capture_layer_ids=[1, 2, 3])
    hidden = mx.concatenate([h[:, -1:, :] for h in out.hidden_states], axis=-1)
    first_bonus = argmax_last(out.logits)
    spec = [first_bonus]
    for tok, _ in _dflash_rounds(
        target,
        drafter,
        cache_spec,
        hidden,
        first_bonus=first_bonus,
        max_tokens=n_new,
        sampler=_greedy,
        greedy_sampling=True,
    ):
        spec.append(tok)

    n = min(len(ref), len(spec))
    assert spec[:n] == ref[:n]


def test_split_deepseek_v4_mtp_writes_sidecar_without_index_mtp_entries(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    text_config = _tiny_deepseek_v4_config()
    _write_checkpoint(
        source,
        {"model_type": "deepseek_v4", **text_config.to_dict()},
        {
            "mtp.0.e_proj.weight": mx.zeros((4, 4), dtype=mx.uint8),
            "mtp.0.e_proj.scale": mx.ones((1, 1), dtype=mx.float32),
            "mtp.0.attn.wq_a.weight": mx.zeros((4, 4), dtype=mx.uint8),
            "mtp.0.attn.wq_a.scale": mx.ones((1, 1), dtype=mx.float32),
            "mtp.0.enorm.weight": mx.zeros((text_config.hidden_size,)),
        },
        shard="mtp.safetensors",
        indexed=True,
    )

    split_deepseek_v4_mtp(str(source), str(output))

    cfg, weights = _read_checkpoint(output)
    assert cfg["model_type"] == "deepseek_v4_mtp"
    assert cfg["block_size"] == 2
    assert cfg["quantization"]["e_proj"]["mode"] == "mxfp8"
    assert cfg["quantization"]["decoder.attn.wq_a"]["mode"] == "mxfp8"
    assert "e_proj.weight" in weights
    assert "e_proj.scales" in weights
    assert "enorm.weight" in weights


def test_split_glm4_moe_lite_mtp_flattens_nextn_layer(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "mtp"
    cfg = {
        "model_type": "glm4_moe_lite",
        "hidden_size": 8,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 4,
        "v_head_dim": 6,
        "kv_lora_rank": 4,
        "moe_intermediate_size": 4,
        "n_routed_experts": 2,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False,
    }
    p = "model.layers.2."
    weights = {
        f"{p}embed_tokens.weight": mx.zeros((16, 8)),
        f"{p}enorm.weight": mx.ones((8,)),
        f"{p}hnorm.weight": mx.ones((8,)),
        f"{p}eh_proj.weight": mx.zeros((8, 16)),
        f"{p}input_layernorm.weight": mx.ones((8,)),
        f"{p}post_attention_layernorm.weight": mx.ones((8,)),
        f"{p}self_attn.kv_b_proj.weight": mx.arange(20 * 4)
        .reshape(20, 4)
        .astype(mx.bfloat16),
        f"{p}self_attn.o_proj.weight": mx.zeros((8, 12)),
        f"{p}self_attn.rotary_emb.inv_freq": mx.ones((2,)),
        f"{p}mlp.gate.weight": mx.zeros((2, 8)),
        f"{p}mlp.gate.e_score_correction_bias": mx.ones((2,), dtype=mx.float32),
        f"{p}mlp.shared_experts.gate_proj.weight": mx.zeros((4, 8)),
        f"{p}mlp.shared_experts.up_proj.weight": mx.zeros((4, 8)),
        f"{p}mlp.shared_experts.down_proj.weight": mx.zeros((8, 4)),
        f"{p}shared_head.norm.weight": mx.ones((8,)),
        f"{p}shared_head.head.weight": mx.zeros((16, 8)),
    }
    for e in range(2):
        weights[f"{p}mlp.experts.{e}.gate_proj.weight"] = mx.zeros((4, 8))
        weights[f"{p}mlp.experts.{e}.up_proj.weight"] = mx.zeros((4, 8))
        weights[f"{p}mlp.experts.{e}.down_proj.weight"] = mx.zeros((8, 4))
    _write_checkpoint(source, cfg, weights, shard="model.safetensors", indexed=True)

    split_glm4_moe_lite_mtp(str(source), str(output))

    out_cfg, out = _read_checkpoint(output)
    assert out_cfg["model_type"] == "glm4_moe_lite_mtp"
    assert out_cfg["block_size"] == 2
    assert out_cfg["text_config"]["model_type"] == "glm4_moe_lite"
    # dedicated nextn embedding and untied head
    assert "model.embed_tokens.weight" in out
    assert "lm_head.weight" in out
    # absorbed-MLA split replaces the fused kv_b_proj
    assert "model.mtp_block.self_attn.kv_b_proj.weight" not in out
    assert out["model.mtp_block.self_attn.embed_q.weight"].shape == (2, 4, 4)
    assert out["model.mtp_block.self_attn.unembed_out.weight"].shape == (2, 6, 4)
    # experts stacked into switch_mlp
    assert out["model.mtp_block.mlp.switch_mlp.gate_proj.weight"].shape == (2, 4, 8)
    assert not any(".experts.0." in k for k in out)
    # router correction bias stays fp32
    assert out["model.mtp_block.mlp.gate.e_score_correction_bias"].dtype == mx.float32
    # non-parameter buffers are dropped
    assert not any(k.endswith("rotary_emb.inv_freq") for k in out)


def _laguna_language_model(num_hidden_layers=4):
    from mlx_vlm.models.laguna.config import ModelConfig

    config = ModelConfig(
        model_type="laguna",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"] * (num_hidden_layers // 2),
    )
    model = laguna_language.LanguageModel(config)
    mx.eval(model.parameters())
    return model


def test_laguna_rollback_skips_trim_when_block_fully_accepted():
    class DummyCache:
        def __init__(self):
            self.trims = []

        def trim(self, n):
            self.trims.append(n)

    cache = DummyCache()
    laguna_language.LanguageModel.rollback_speculative_cache(
        SimpleNamespace(), [cache], None, 3, block_size=4
    )

    assert cache.trims == []


def test_laguna_rollback_rejects_ragged_batch_acceptance():
    with pytest.raises(RuntimeError, match="uniform per-row"):
        laguna_language.LanguageModel.rollback_speculative_cache(
            SimpleNamespace(), [None], None, [0, 2], block_size=4
        )


def test_laguna_rollback_advances_real_cache_offsets():
    model = _laguna_language_model()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]]), cache=cache)
    assert [c.offset for c in cache] == [4, 4, 4, 4]

    model.rollback_speculative_cache(cache, None, 1, block_size=4)

    assert [c.offset for c in cache] == [2, 2, 2, 2]


def test_laguna_captures_target_hidden_states_for_dflash():
    model = _laguna_language_model()
    out = model(
        mx.array([[1, 2, 3, 4]]),
        cache=model.make_cache(),
        capture_layer_ids=[1, 2],
        speculative_verify=True,
    )

    assert out.hidden_states is not None
    assert [h.shape for h in out.hidden_states] == [(1, 4, 64), (1, 4, 64)]
    assert mx.concatenate(out.hidden_states, axis=-1).shape == (1, 4, 128)


def test_laguna_dflash_config_derives_sliding_windows_when_absent():
    from mlx_vlm.speculative.drafters.laguna_dflash.config import DFlashConfig

    params = {
        "model_type": "laguna",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "draft_vocab_size": 256,
        "max_position_embeddings": 512,
        "rope_theta": 500000.0,
        "layer_types": ["sliding_attention"] * 2,
        "sliding_window": 512,
        "gating": "per-head",
        "eagle_aux_hidden_state_layer_ids": [2, 4],
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 3,
            "target_layer_ids": [1, 3],
            "num_target_layers": 4,
            "causal": True,
        },
    }

    config = DFlashConfig.from_dict(params)

    assert config.sliding_windows == [512, 512]


def _published_drafter_config(**overrides):
    return {
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        **overrides,
    }


def _tiny_draft_dimensions(**overrides):
    return {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        **overrides,
    }


def _dflash2_published_config():
    return _published_drafter_config(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        is_causal=False,
        hidden_size=5120,
        intermediate_size=17408,
        hidden_act="silu",
        vocab_size=248320,
        max_position_embeddings=262144,
        num_target_layers=64,
        layer_types=["sliding_attention"] * 5,
        sliding_window=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000},
        dflash_config={
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61],
        },
    )


def _dflash2_config():
    config = _dflash2_published_config()
    config.update(
        _tiny_draft_dimensions(
            num_target_layers=2,
            layer_types=["full_attention"],
            sliding_window=None,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
        )
    )
    config["dflash_config"] = {
        "block_size": 3,
        "runtime_block_size": 3,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 4,
        "selector_top_k": 4,
        "target_layer_ids": [0],
    }
    return DFlash2Config.from_dict(config)


def _qwen_dflash_target():
    config = _tiny_qwen3_5_text_config()
    config.num_hidden_layers = config.full_attention_interval = 2
    outer_config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=config,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=30,
        video_token_id=29,
        vision_start_token_id=28,
    )
    model = qwen_language.LanguageModel(config, outer_config)
    model.set_dtype(mx.bfloat16)
    return model


def test_published_dflash2_config_contract():
    published = _dflash2_published_config()
    config = DFlash2Config.from_dict(published)

    assert config.model_type == "dflash2"
    assert config.backbone_model_type == "qwen3"
    assert config.block_size == 8
    assert config.runtime_block_size == 5
    assert config.target_layer_ids == [5, 19, 33, 47, 61]
    assert config.conv_kernel_size == 2
    assert config.conv_group_size == 16
    assert config.selector_rank == 256
    assert config.selector_top_k == 16
    assert config.rope_theta == 10000000
    assert config.rope_scaling == {"rope_type": "default"}

    drafter = DFlash2DraftModel(config)
    assert drafter.prefer_requested_block_size is False
    assert drafter.dflash_initial_block_size == 3
    assert drafter.dflash_min_block_size == 3


def test_dflash2_sanitize_normalizes_published_codebooks():
    drafter = DFlash2DraftModel(_dflash2_config())
    predecessor = mx.zeros((32, 4))
    successor = mx.ones((32, 4))

    weights = drafter.sanitize(
        {
            "candidate_selector.predecessor_codebook": predecessor,
            "candidate_selector.successor_codebook": successor,
        }
    )

    assert weights == {
        "candidate_selector.predecessor_codebook.weight": predecessor,
        "candidate_selector.successor_codebook.weight": successor,
    }


def test_positioned_proposal_sampling_is_independent_of_target_filters():
    sampler = _PositionedTargetSampler(temperature=1.0, top_p=0.95, top_k=20, seed=7)
    scores = mx.zeros((1, 16))

    first = sampler.sample_proposal(scores, row_ids=[0], positions=[3])
    second = sampler.sample_proposal(scores, row_ids=[0], positions=[3])

    assert first.shape == (1,)
    assert bool(mx.array_equal(first, second))


def _laguna_config_dict():
    return _published_drafter_config(
        model_type="laguna",
        hidden_size=3072,
        intermediate_size=12288,
        num_hidden_layers=6,
        num_attention_heads=72,
        max_position_embeddings=1048576,
        rope_theta=500000.0,
        vocab_size=100352,
        draft_vocab_size=100352,
        layer_types=["sliding_attention"] * 6,
        sliding_windows=[512] * 6,
        sliding_window=512,
        gating="per-head",
        eagle_aux_hidden_state_layer_ids=[2, 11, 20, 30, 39, 48],
        dflash_config={
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 48,
            "target_layer_ids": [1, 10, 19, 29, 38, 47],
            "causal": True,
        },
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda params: params.pop("hidden_size"),
            "missing checkpoint fields: hidden_size",
        ),
        (
            lambda params: params["dflash_config"].pop("target_layer_ids"),
            "missing dflash_config fields: target_layer_ids",
        ),
        (
            lambda params: params.update({"layer_types": ["full_attention"] * 6}),
            "sliding_attention",
        ),
        (
            lambda params: params["dflash_config"].update(
                {"target_layer_ids": [1, 10, 19, 29, 38, 48]}
            ),
            "target_layer_ids",
        ),
    ],
)
def test_malformed_checkpoint_contract_is_rejected(mutate, message):
    params = _laguna_config_dict()
    mutate(params)

    with pytest.raises(ValueError, match=message):
        LagunaDFlashConfig.from_dict(params)


def test_target_layer_count_and_tokenizer_length_are_checked():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    target = SimpleNamespace(num_hidden_layers=48, vocab_size=100352)

    validate_laguna_dflash_target(
        config, target_model_config=target, target_tokenizer_length=100352
    )
    with pytest.raises(ValueError, match="layer count"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(num_hidden_layers=47),
            target_tokenizer_length=100352,
        )
    with pytest.raises(ValueError, match="vocabulary"):
        validate_laguna_dflash_target(
            config, target_model_config=target, target_tokenizer_length=100351
        )


def test_generic_drafter_gate_invokes_laguna_target_validation():
    from mlx_vlm.speculative.drafters.laguna_dflash import LagunaDFlashDraftModel

    draft = LagunaDFlashDraftModel(LagunaDFlashConfig.from_dict(_laguna_config_dict()))
    target = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=48, vocab_size=100352),
        model=SimpleNamespace(layers=[object()] * 48),
        rollback_speculative_cache=lambda *args: 0,
    )

    validate_drafter_compatibility(target, draft, "dflash")
    target.model.layers.pop()
    with pytest.raises(ValueError, match="layer count"):
        validate_drafter_compatibility(target, draft, "dflash")


def test_published_weight_keys_and_shapes_are_explicit():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    assert expected["layers.0.self_attn.o_proj.weight"] == (3072, 9216)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}

    validate_laguna_dflash_weights(weights, config)
    bad = dict(weights)
    bad["layers.0.self_attn.g_proj.weight"] = SimpleNamespace(shape=(71, 3072))
    with pytest.raises(ValueError, match="weight shapes"):
        validate_laguna_dflash_weights(bad, config)


def test_weight_key_drift_is_rejected():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())
    expected = expected_laguna_dflash_weight_shapes(config)
    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    weights["layers.0.self_attn.extra.weight"] = SimpleNamespace(shape=(1,))

    with pytest.raises(ValueError, match="weight keys"):
        validate_laguna_dflash_weights(weights, config)


def test_target_without_rollback_support_is_rejected():
    config = LagunaDFlashConfig.from_dict(_laguna_config_dict())

    with pytest.raises(ValueError, match="rollback"):
        validate_laguna_dflash_target(
            config,
            target_model_config=SimpleNamespace(
                num_hidden_layers=48, vocab_size=100352
            ),
            target_tokenizer_length=100352,
            target_language_model=SimpleNamespace(),
        )


def _glimmer_published_config():
    return _published_drafter_config(
        model_type="muse_glimmer_assistant",
        hidden_size=6656,
        intermediate_size=19968,
        rms_norm_eps=1e-05,
        max_position_embeddings=131072,
        rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
        layer_types=["sliding_attention"] * 5,
        sliding_window=2048,
        block_size=16,
        mask_token_id=201818,
        target_layer_ids=[1, 13, 25, 37, 49],
    )


def _glimmer_config():
    return MuseGlimmerAssistantConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        block_size=4,
        mask_token_id=63,
        target_layer_ids=[0, 1],
        num_target_layers=2,
        vocab_size=64,
    )


def _glimmer_target():
    text = TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    return MuseGlimmerModel(
        MuseGlimmerConfig(
            text_config=text,
            vision_config=vision,
            image_token_id=7,
            video_token_id=6,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    )


def test_published_config_and_weight_contract():
    config = MuseGlimmerAssistantConfig.from_dict(_glimmer_published_config())

    assert config.rope_theta == 500000.0
    assert config.target_layer_ids == [1, 13, 25, 37, 49]
    assert config.num_target_layers == 52
    assert config.vocab_size == 202048

    expected = expected_muse_glimmer_assistant_weight_shapes(config)
    assert len(expected) == 58
    assert expected["encoder.fc.weight"] == (6656, 33280)
    assert expected["layers.0.self_attn.o_proj.weight"] == (6656, 4096)
    assert expected["layers.4.mlp.down_proj.weight"] == (6656, 19968)

    weights = {key: SimpleNamespace(shape=shape) for key, shape in expected.items()}
    validate_muse_glimmer_assistant_weights(weights, config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config.update({"layer_types": ["full_attention"] * 5}),
            "sliding_attention",
        ),
        (
            lambda config: config.update({"target_layer_ids": [1, 13, 25, 37, 52]}),
            "target_layer_ids",
        ),
        (lambda config: config.update({"mask_token_id": 202048}), "mask_token_id"),
    ],
)
def test_invalid_checkpoint_contract_is_rejected(mutate, message):
    config = _glimmer_published_config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        MuseGlimmerAssistantConfig.from_dict(config)


def test_binding_uses_raw_target_embedding_and_checks_target_family():
    target = _glimmer_target()
    drafter = MuseGlimmerAssistantModel(_glimmer_config())

    validate_drafter_compatibility(target, drafter, "dflash")
    drafter.bind(target)
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    raw = target.language_model.model.embed_tokens(inputs)
    normalized = target.language_model.model.embed_norm(raw)
    actual = drafter._embed_input_tokens(inputs)
    mx.eval(raw, normalized, actual)

    assert bool(mx.array_equal(actual, raw).item())
    assert not bool(mx.array_equal(actual, normalized).item())

    target.language_model.config.model_type = "other"
    with pytest.raises(ValueError, match="Muse Glimmer text target"):
        validate_drafter_compatibility(target, drafter, "dflash")


def _dspark_published_config():
    return _published_drafter_config(
        architectures=["Lfm2DSparkDraftModel"],
        model_type="qwen3",
        hidden_size=2048,
        head_dim=64,
        intermediate_size=6144,
        hidden_act="silu",
        rms_norm_eps=1e-05,
        vocab_size=128000,
        rope_theta=10000000.0,
        max_position_embeddings=128000,
        layer_types=["full_attention"] * 5,
        block_size=9,
        dflash_config={
            "mask_token_id": 125017,
            "target_layer_ids": [2, 9, 17, 21, 27],
            "num_target_layers": 30,
        },
        markov_rank=256,
        rope_is_neox_style=False,
        enable_confidence_head=True,
        markov_head_type="vanilla",
    )


def _dspark_qwen_published_config():
    return _published_drafter_config(
        architectures=["DSparkDraftModel"],
        model_type="qwen3",
        block_size=7,
        confidence_head_with_markov=True,
        hidden_size=5120,
        intermediate_size=10240,
        num_attention_heads=40,
        hidden_act="silu",
        vocab_size=248320,
        max_position_embeddings=262144,
        num_target_layers=64,
        layer_types=["full_attention"] * 5,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        dflash_config={
            "projector_type": "dspark",
            "mask_token_id": 248077,
            "target_layer_ids": [4, 16, 28, 40, 52],
            "markov_rank": 256,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
        },
    )


def _dspark_nemotron_published_config():
    return _published_drafter_config(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        hidden_size=2688,
        intermediate_size=6144,
        num_hidden_layers=6,
        num_key_value_heads=2,
        hidden_act="silu",
        vocab_size=131072,
        max_position_embeddings=1048576,
        rope_theta=10000,
        layer_types=["sliding_attention"] * 6,
        sliding_window=1024,
        block_size=8,
        dspark_bonus_anchor=True,
        dflash_query_causal=True,
        attention_sink_bias=True,
        markov_rank=512,
        markov_head_type="vanilla",
        dflash_config={
            "attention_sink_bias": True,
            "causal": True,
            "mask_token_id": 990,
            "swa_window_size": 1024,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
            "use_swa": True,
            "sample_from_anchor": False,
        },
    )


def _dspark_config(qwen=False):
    return DSparkConfig.from_dict(
        _tiny_draft_dimensions(
            hidden_size=16 if qwen else 8,
            intermediate_size=32 if qwen else 16,
            head_dim=8 if qwen else 4,
            architectures=["DSparkDraftModel" if qwen else "Lfm2DSparkDraftModel"],
            model_type="qwen3",
            rms_norm_eps=1e-6 if qwen else 1e-5,
            rope_theta=10000.0,
            layer_types=["full_attention"],
            block_size=3,
            markov_rank=4,
            markov_head_type="vanilla",
            enable_confidence_head=True,
            **({"num_target_layers": 2} if qwen else {"rope_is_neox_style": False}),
            dflash_config={
                "mask_token_id": 31,
                **(
                    {"projector_type": "dspark", "target_layer_ids": [0]}
                    if qwen
                    else {"target_layer_ids": [0, 2], "num_target_layers": 3}
                ),
            },
        )
    )


def _dspark_qwen_config():
    return _dspark_config(qwen=True)


def _lfm2_target():
    config = Lfm2Config(
        model_type="lfm2",
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        block_dim=8,
        block_ff_dim=16,
        block_multiple_of=1,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=False,
        rope_theta=10000.0,
        layer_types=["conv", "full_attention", "conv"],
        full_attn_idxs=[1],
        tie_word_embeddings=True,
    )
    return Lfm2Model(config)


def _lfm2_moe_target():
    config = Lfm2MoeConfig(
        model_type="lfm2_moe",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        use_expert_bias=True,
        num_dense_layers=1,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        rope_theta=10000.0,
        layer_types=["conv", "full_attention", "conv"],
        tie_word_embeddings=True,
    )
    return Lfm2MoeModel(config)


def _generated_tokens(
    target, prompt, drafter=None, *, max_tokens=10, temperature=0, seed=None
):
    kwargs = {}
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="dflash")
    if hasattr(target, "language_model"):
        generation_target = target
    else:

        def get_input_embeddings(input_ids, pixel_values=None, mask=None, **kwargs):
            del pixel_values, kwargs
            position_ids, rope_deltas = target.get_rope_index(
                input_ids, attention_mask=mask
            )
            return InputEmbeddingsFeatures(
                inputs_embeds=target.model.embed_tokens(input_ids),
                position_ids=position_ids,
                rope_deltas=rope_deltas,
            )

        generation_target = SimpleNamespace(
            language_model=target, get_input_embeddings=get_input_embeddings
        )
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            prompt,
            generation_target,
            None,
            None,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            prefill_step_size=None,
            **kwargs,
        )
    ]


def test_published_nemotron_config_normalizes_dspark_contract():
    config = DSparkConfig.from_dict(_dspark_nemotron_published_config())
    drafter = DSparkDraftModel(config)

    assert config.proposal_length == 7
    assert config.block_size == 8
    assert config.runtime_block_size == 8
    assert config.target_layer_ids == [1, 5, 19, 29, 41, 51]
    assert config.num_target_layers == 52
    assert config.sliding_window == 1024
    assert config.is_causal is True
    assert config.attention_sink_bias is True
    assert config.sample_from_anchor is False
    assert config.enable_confidence_head is False
    assert config.block_size_policy == "adaptive"
    assert config.dflash_initial_block_size == 4
    assert drafter.prefer_requested_block_size is False
    assert drafter.choose_initial_block_size(512, 8) == 6
    assert drafter.choose_initial_block_size(1024, 8) == 6
    assert drafter.choose_initial_block_size(1025, 8) == 4
    assert drafter.choose_block_ceiling(512, 8) == 6
    assert drafter.choose_block_ceiling(1024, 8) == 6
    assert drafter.choose_block_ceiling(1025, 8) == 4
    assert drafter.confidence_head is None
    assert all(layer.self_attn.is_causal for layer in drafter.layers)
    assert all(
        layer.self_attn.attention_sink_bias is not None for layer in drafter.layers
    )


def _dspark_nemotron_config():
    config_dict = _dspark_nemotron_published_config()
    config_dict["hidden_size"] = 8
    config_dict["intermediate_size"] = 16
    config_dict["num_hidden_layers"] = 1
    config_dict["num_attention_heads"] = 2
    config_dict["num_key_value_heads"] = 1
    config_dict["head_dim"] = 4
    config_dict["vocab_size"] = 32
    config_dict["layer_types"] = ["sliding_attention"]
    config_dict["markov_rank"] = 4
    config_dict["dflash_config"]["mask_token_id"] = 31
    config_dict["dflash_config"]["target_layer_ids"] = [0]
    return DSparkConfig.from_dict(config_dict)


def test_nemotron_dspark_sanitize_installs_checkpoint_embedding():
    config = _dspark_nemotron_config()
    drafter = DSparkDraftModel(config)
    weight = mx.zeros((32, 8))

    sanitized = drafter.sanitize({"model.embed_tokens.weight": weight})

    assert sanitized["embed_tokens.weight"] is weight
    assert drafter.embed_tokens is not None


def test_dspark_bonus_anchor_uses_mask_position_logits():
    config = _dspark_nemotron_config()
    drafter = DSparkDraftModel(config)
    seen = {}

    def hidden(inputs, target_hidden, cache):
        seen["inputs"] = inputs
        return mx.zeros((1, inputs.shape[1], config.hidden_size))

    def logits(states):
        seen["states"] = states
        return mx.zeros((1, states.shape[1], config.vocab_size))

    drafter._hidden = hidden
    drafter._logits = logits
    drafter.draft_block(
        3,
        mx.zeros((1, 1, config.hidden_size)),
        [],
        config.block_size,
        lambda values: mx.argmax(values, axis=-1),
    )

    assert seen["inputs"].shape[1] == config.block_size
    assert seen["inputs"][0, 0].item() == 3
    assert seen["inputs"][0, 1:].tolist() == [31] * config.proposal_length
    assert seen["states"].shape[1] == config.proposal_length


def test_dspark_block_policy_can_be_declared_by_any_checkpoint():
    published = _dspark_published_config()
    published["dflash_config"]["block_size_policy"] = "adaptive"
    published["dflash_config"]["dflash_initial_block_size"] = 3

    config = DSparkConfig.from_dict(published)
    drafter = DSparkDraftModel(config)

    assert config.block_size_policy == "adaptive"
    assert config.dflash_initial_block_size == 3
    assert drafter.prefer_requested_block_size is False


def test_dspark_requires_matching_target_structure():
    drafter = DSparkDraftModel(_dspark_config())
    target = _lfm2_target()

    validate_drafter_compatibility(target, drafter, "dflash")
    target.language_model.config.hidden_size += 1
    with pytest.raises(ValueError, match="target hidden-size mismatch"):
        validate_drafter_compatibility(target, drafter, "dflash")


def test_dspark_rollback_error_names_target_model_class():
    class TargetWithoutRollback:
        def __init__(self):
            self.config = SimpleNamespace(
                hidden_size=8, num_hidden_layers=3, vocab_size=32
            )
            self.model = SimpleNamespace(layers=[object()] * 3)

    target = SimpleNamespace(language_model=TargetWithoutRollback())

    with pytest.raises(
        ValueError, match="DSpark target TargetWithoutRollback does not expose"
    ):
        validate_dspark_target(_dspark_config(), target)


def test_published_qwen38_dspark_accepts_nested_target_metadata():
    config = DSparkConfig.from_dict(_dspark_qwen_published_config())
    target = SimpleNamespace(
        language_model=SimpleNamespace(
            config=SimpleNamespace(
                model_type="qwen3_5",
                text_config=SimpleNamespace(
                    model_type="qwen3_5_text",
                    hidden_size=5120,
                    num_hidden_layers=64,
                    vocab_size=248320,
                ),
            ),
            model=SimpleNamespace(layers=[object()] * 64),
            rollback_speculative_cache=lambda *args: None,
        )
    )

    validate_dspark_target(config, target)


def test_tiny_dspark_forward_uses_markov_head_and_published_block_semantics():
    mx.random.seed(0)
    target = _lfm2_target()
    drafter = DSparkDraftModel(_dspark_config())
    mx.eval(target.parameters(), drafter.parameters())

    assert drafter.rope.traditional is True
    target_cache = target.make_cache()
    prompt = mx.array([[1, 2, 3]], dtype=mx.int32)
    output = target.language_model(
        prompt, cache=target_cache, capture_layer_ids=drafter.config.target_layer_ids
    )
    hidden = mx.concatenate(output.hidden_states, axis=-1)
    draft_cache = drafter.reset(target)
    tokens = drafter.draft_block(
        4,
        hidden,
        draft_cache,
        block_size=drafter.config.block_size,
        sampler=_greedy,
    )
    mx.eval(tokens)

    assert hidden.shape == (1, 3, 16)
    assert tokens.shape == (1, drafter.config.proposal_length)
    assert all(cache.offset == 3 for cache in draft_cache)


def test_lfm2_moe_exact_speculative_verify_narrow_router_matches_singletons():
    mx.random.seed(12)
    moe = Lfm2MoeSparseMoeBlock(
        SimpleNamespace(
            hidden_size=64,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            use_expert_bias=True,
        )
    )
    moe.set_dtype(mx.bfloat16)
    inputs = mx.random.normal((1, 5, 64)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [moe(inputs[:, position : position + 1]) for position in range(5)], axis=1
    )
    actual = Lfm2ExactSpeculativeVerifier()._feed_forward(moe, inputs)
    mx.eval(expected, actual)

    assert bool(mx.array_equal(actual, expected))


def test_lfm2_ragged_batch_rollback_matches_each_committed_prefix():
    mx.random.seed(5)
    target = _lfm2_target()
    lm = target.language_model
    mx.eval(target.parameters())

    prompt = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
    verify = mx.array([[9, 10, 11, 12], [13, 14, 15, 16]], dtype=mx.int32)
    accepted = mx.array([0, 2], dtype=mx.int32)
    rolled_cache = [
        BatchKVCache([0, 0]) if layer.is_attention_layer else ArraysCache(size=1)
        for layer in lm.model.layers
    ]

    lm(prompt, cache=rolled_cache)
    verify_out = lm(verify, cache=rolled_cache, speculative_verify=True)
    lm.rollback_speculative_cache(
        rolled_cache, verify_out.gdn_states, accepted, block_size=verify.shape[1]
    )
    assert rolled_cache[1].offset.tolist() == [5, 7]
    probes = mx.array([[17], [18]], dtype=mx.int32)
    rolled_logits = lm(probes, cache=rolled_cache).logits

    reference_logits = []
    for row, accepted_count in enumerate(accepted.tolist()):
        reference_cache = lm.make_cache()
        committed = mx.concatenate(
            [prompt[row], verify[row, : int(accepted_count) + 1]], axis=0
        )[None, :]
        lm(committed, cache=reference_cache)
        reference_logits.append(lm(probes[row : row + 1], cache=reference_cache).logits)
    reference_logits = mx.concatenate(reference_logits, axis=0)
    mx.eval(rolled_logits, reference_logits)

    assert bool(mx.allclose(rolled_logits, reference_logits, atol=1e-4))


@pytest.mark.parametrize(
    ("target_factory", "draft_config_factory"),
    [
        (_lfm2_target, _dspark_config),
        (_qwen_dflash_target, _dspark_qwen_config),
    ],
)
def test_dspark_repeated_generation_resets_request_state(
    target_factory, draft_config_factory
):
    mx.random.seed(31)
    target = target_factory()
    drafter = DSparkDraftModel(draft_config_factory())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    baseline = _generated_tokens(target, prompt)
    first = _generated_tokens(target, prompt, drafter)
    first_accept_lens = list(drafter.accept_lens)
    first_draft_lens = list(drafter.draft_lens)

    # These lists drive the adaptive controller and must describe one request,
    # not lifetime state carried over from a previous generation.
    drafter.accept_lens.append(999)
    drafter.draft_lens.append(999)
    second = _generated_tokens(target, prompt, drafter)

    assert first == second == baseline
    assert drafter.accept_lens == first_accept_lens
    assert drafter.draft_lens == first_draft_lens


def test_sampled_dspark_generation_matches_stateful_baseline():
    mx.random.seed(43)
    target = _qwen_dflash_target()
    drafter = DSparkDraftModel(_dspark_qwen_config())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    mx.random.seed(47)
    baseline = _generated_tokens(target, prompt, max_tokens=24, temperature=0.7)
    mx.random.seed(47)
    speculative = _generated_tokens(
        target, prompt, drafter, max_tokens=24, temperature=0.7
    )

    assert speculative == baseline
    assert drafter.draft_lens


def _published_gemma4_dspark_config():
    return _published_drafter_config(
        architectures=["Gemma4DSparkModel"],
        model_type="gemma4_text",
        attention_k_eq_v=True,
        block_size=7,
        confidence_head_with_markov=True,
        enable_confidence_head=True,
        final_logit_softcapping=30.0,
        global_head_dim=512,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        hidden_size=3840,
        intermediate_size=15360,
        layer_types=["full_attention"] * 5,
        markov_head_type="vanilla",
        markov_rank=256,
        mask_token_id=4,
        max_position_embeddings=262144,
        num_attention_heads=16,
        num_global_key_value_heads=1,
        num_target_layers=48,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        sliding_window=1024,
        target_layer_ids=[5, 17, 29, 41, 46],
        tie_word_embeddings=False,
        vocab_size=262144,
    )


def test_published_gemma4_dspark_config_reads_flat_contract():
    """The Gemma 4 checkpoint publishes DSpark fields flat, not under dflash_config."""
    from mlx_vlm.speculative.drafters.gemma4_dspark import ModelConfig as G4Config

    config = G4Config.from_dict(_published_gemma4_dspark_config())

    assert config.model_type == "gemma4_dspark"
    assert config.backbone_model_type == "gemma4_text"
    assert config.proposal_length == 7
    assert config.block_size == 8
    assert config.target_layer_ids == [5, 17, 29, 41, 46]
    assert config.num_target_layers == 48
    assert config.attention_k_eq_v is True
    assert config.global_head_dim == 512
    assert config.final_logit_softcapping == 30.0


def test_gemma4_dspark_attention_forward_runs():
    """A forward must not raise: the custom __init__ still has to set is_causal and
    attention_sink_bias, which the inherited DFlashAttention.__call__ reads."""
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.speculative.drafters.gemma4_dspark import Model as G4Model
    from mlx_vlm.speculative.drafters.gemma4_dspark import ModelConfig as G4Config
    from mlx_vlm.speculative.drafters.gemma4_dspark.gemma4_dspark import (
        Gemma4DSparkAttention,
    )

    config = G4Config.from_dict(_published_gemma4_dspark_config())
    rope = G4Model(config).rope
    attention = Gemma4DSparkAttention(config, 0)
    x = mx.zeros((1, 2, config.hidden_size))
    x_ctx = mx.zeros((1, 3, config.hidden_size))

    out = attention(x, x_ctx, rope, KVCache())
    mx.eval(out)

    assert out.shape == (1, 2, config.hidden_size)


def test_gemma4_exact_verifier_is_opt_in():
    """Exact block verification is off unless the checkpoint asks for it.

    Singleton-equivalent numerics roughly halve decode throughput, and the
    other DSpark targets verify with the plain path, so Gemma 4 matches them
    by default and exposes the trade as a config flag.
    """
    from mlx_vlm.models.gemma4.config import TextConfig

    assert TextConfig().exact_speculative_verify is False
    assert TextConfig(exact_speculative_verify=True).exact_speculative_verify is True

    routed = []

    class _Spy:
        def __call__(self, *a, **kw):
            routed.append(True)
            return "verified"

    import mlx_vlm.models.gemma4.language as g4

    real = g4._EXACT_SPECULATIVE_VERIFIER
    g4._EXACT_SPECULATIVE_VERIFIER = _Spy()
    try:
        model = g4.LanguageModel(TextConfig(exact_speculative_verify=True))
        assert model(mx.array([[1]]), speculative_verify=True) == "verified"
        assert routed == [True]

        routed.clear()
        off = g4.LanguageModel(TextConfig(exact_speculative_verify=False))
        try:
            off(mx.array([[1]]), speculative_verify=True)
        except Exception:
            pass
        assert routed == []
    finally:
        g4._EXACT_SPECULATIVE_VERIFIER = real


@pytest.mark.parametrize("batch", [2, 4])
@pytest.mark.parametrize("state_steps", [1, 3])
def test_qwen_recurrent_history_uses_allocated_batch_stride(batch, state_steps):
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update_with_states

    mx.random.seed(2127)
    length, heads, width = 4, 2, 32
    shape = (batch, length, heads, width)
    q, k, v = [mx.random.normal(shape).astype(mx.bfloat16) * 0.1 for _ in range(3)]
    a, b = [mx.random.normal(shape[:-1]).astype(mx.bfloat16) for _ in range(2)]
    args = (q, k, v, a, b, mx.zeros((heads,)), mx.zeros((heads,)))
    output, state, history = gated_delta_update_with_states(
        *args, state_steps=state_steps
    )
    mx.eval(output, state, history)
    # Each independently executed row is an oracle with no batch stride.
    for row in range(batch):
        expected = gated_delta_update_with_states(
            *(x[row : row + 1] for x in args[:5]), *args[5:], state_steps=state_steps
        )
        for actual, reference in zip((output, state, history), expected):
            assert mx.array_equal(actual[row : row + 1], reference).item()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("step", [1, 4])
def test_rotating_transaction_preserves_native_layout_across_rounds(batched, step):
    cache = BatchRotatingKVCache(8, [0, 0]) if batched else RotatingKVCache(8)
    prefix = mx.broadcast_to(mx.arange(13)[None, None, :, None], (2, 1, 13, 1))
    cache.update_and_fetch(prefix, prefix)
    reference = deepcopy(cache)
    for round_index, retained in enumerate((3, 1, 4, 0, None, 2)):
        incoming = mx.broadcast_to(
            (mx.arange(4) + 100 + round_index * 10)[None, None, :, None], (2, 1, 4, 1)
        )
        transaction = start_speculative_cache([cache], 4)
        for start in range(0, 4, step):
            part = incoming[:, :, start : start + step]
            cache.update_and_fetch(part, part)
        mx.eval(cache.state)
        if retained is None:
            transaction.abort()
        else:
            transaction.commit([retained] * 2)
            for start in range(0, retained, step):
                part = incoming[:, :, start : min(start + step, retained)]
                reference.update_and_fetch(part, part)
        assert "update_and_fetch" not in cache.__dict__
        for actual, expected in zip(cache.state, reference.state):
            assert mx.array_equal(actual, expected).item()
        assert cache.meta_state == reference.meta_state


def test_rotating_transaction_retains_different_row_prefixes():
    cache = BatchRotatingKVCache(8, [0, 0])
    prefix = mx.broadcast_to(mx.arange(12)[None, None, :, None], (2, 1, 12, 1))
    cache.update_and_fetch(prefix, prefix)
    transaction = start_speculative_cache([cache], 4)
    incoming = mx.broadcast_to(mx.arange(100, 104)[None, None, :, None], (2, 1, 4, 1))
    cache.update_and_fetch(incoming, incoming)
    transaction.commit([3, 1])
    assert cache.offset.tolist() == [15, 13]
    mask = cache.make_mask(1)
    next_kv = mx.full((2, 1, 1, 1), 200)
    keys, _ = cache.update_and_fetch(next_kv, next_kv)
    for row, retained in enumerate((3, 1)):
        visible = mx.where(mask[row, 0, 0], keys[row, 0, :, 0], -1)
        expected = list(range(12)) + list(range(100, 100 + retained)) + [200]
        assert sorted(value for value in visible.tolist() if value >= 0) == sorted(
            expected[-8:]
        )


@pytest.mark.parametrize("batch", [1, 2])
def test_deepseek_chunked_prefill_keeps_all_dflash_features(batch):
    mx.random.seed(2127)
    config = _tiny_deepseek_v4_config()
    config.compress_ratios = [4]
    model = DeepseekLanguageModel(config)
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7]] * batch)
    drafter = SimpleNamespace(config=SimpleNamespace(target_layer_ids=[0]))
    cache = _make_cache(model, [0] * batch)
    reference = []
    for start in range(0, 6, 2):
        output = model(tokens[:, start : start + 2], cache=cache, capture_layer_ids=[0])
        reference.append(output.hidden_states[0])
    output = model(tokens[:, -1:], cache=cache, capture_layer_ids=[0])
    reference.append(output.hidden_states[0])
    expected = mx.concatenate(reference, axis=1)
    processing = PromptProcessingBatch(
        model=model,
        uids=list(range(batch)),
        input_ids=tokens.tolist(),
        max_tokens=[8] * batch,
        inputs_embeds=model.model.embed_tokens(tokens),
        prompt_kwargs={},
        prefill_step_size=2,
        draft_model=drafter,
        draft_kind="dflash",
    )
    assert processing.prefill_step_size == 2
    while processing.needs_processing():
        assert processing.prompt_step() <= 2
    generated = processing.generate(
        _greedy,
        lambda *args: False,
        compute_logprobs=False,
    )
    assert generated.hidden.shape[1] == tokens.shape[1]
    assert mx.array_equal(generated.hidden, expected).item()


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("format", ["mxfp", "affine"])
def test_deepseek_native_quantized_verifier_matches_repeated_decode(batch, format):
    mx.random.seed(2127)
    config = _tiny_deepseek_v4_config()
    config.hidden_size = config.moe_intermediate_size = 64
    config.q_lora_rank = config.head_dim = config.o_lora_rank = 32
    config.index_head_dim = 32
    config.hc_mult = 4
    config.n_routed_experts, config.num_experts_per_tok = 8, 2
    config.num_hidden_layers, config.compress_ratios = 3, [0, 4, 128]
    model = DeepseekLanguageModel(config)
    model.update(
        tree_map_with_path(
            lambda key, value: (
                value.astype(mx.bfloat16) if model.cast_predicate(key) else value
            ),
            model.parameters(),
        )
    )
    for layer in model.model.layers:
        for connection in (layer.attn_hc, layer.ffn_hc):
            connection.fn = mx.random.normal(connection.fn.shape) * 0.01
    nn.quantize(
        model,
        class_predicate=lambda path, module: (
            (
                {
                    "group_size": 32,
                    "bits": 4,
                    "mode": "mxfp4" if format == "mxfp" else "affine",
                }
                if "switch_mlp" in path
                else {
                    "group_size": 32,
                    "bits": 8,
                    "mode": "mxfp8" if format == "mxfp" else "affine",
                }
            )
            if hasattr(module, "to_quantized") and module.weight.shape[-1] % 32 == 0
            else False
        ),
    )
    model.eval()
    reference, speculative = [_make_cache(model, [0] * batch) for _ in range(2)]
    prompt = mx.broadcast_to((mx.arange(131)[None] % 30) + 1, (batch, 131))
    for caches in (reference, speculative):
        mx.eval(model(prompt, cache=caches).logits)
    for retained in (1, 4, 3):
        tokens = mx.random.randint(1, 31, (batch, 4))
        oracle_cache = deepcopy(reference)
        expected = []
        expected_features = []
        for index in range(4):
            output = model(
                tokens[:, index : index + 1],
                cache=oracle_cache,
                capture_layer_ids=[0, 1, 2],
            )
            mx.eval(output.logits, output.hidden_states)
            expected.append(output.logits)
            expected_features.append(output.hidden_states)
        output, transaction = _dflash_verify(model, tokens, speculative, [0, 1, 2])
        assert mx.array_equal(output.logits, mx.concatenate(expected, axis=1)).item()
        for actual, parts in zip(output.hidden_states, zip(*expected_features)):
            assert mx.array_equal(actual, mx.concatenate(parts, axis=1)).item()
        transaction.commit([retained] * batch)
        for index in range(retained):
            mx.eval(model(tokens[:, index : index + 1], cache=reference).logits)


def test_hyperconnection_prefill_is_chunk_invariant():
    from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection

    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hidden_size, config.hc_mult = 4096, 4
    connection = HyperConnection(config)
    connection.fn = mx.random.normal(connection.fn.shape) * 0.01
    connection.eval()
    x = mx.random.normal((1, 4096, 4, 4096)).astype(mx.bfloat16)
    expected = connection(x)
    mx.eval(expected)
    parts = [connection(x[:, :2048]), connection(x[:, 2048:])]
    for reference, segments in zip(expected, zip(*parts)):
        assert mx.array_equal(reference, mx.concatenate(segments, axis=1)).item()


@pytest.mark.parametrize("family", ["glm", "qwen"])
@pytest.mark.parametrize("step", [1, 2, 5])
def test_ordinary_linear_attention_uses_temporal_cache_without_adapter(family, step):
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet

    mx.random.seed(2127)
    if family == "glm":
        config = _tiny_glm5_next_text_config()
        config.linear_head_dim = 32
        layer = Glm5NextLinearAttention(config)
    else:
        cases = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
            "cases"
        ]
        values = next(case for case in cases if case["id"] == "qwen4_exp")["config"][
            "text_config"
        ]
        config = Qwen4TextConfig.from_dict(values)
        config.linear_key_head_dim = config.linear_value_head_dim = 32
        layer = Qwen3_5GatedDeltaNet(config)
    layer.eval()
    cache = ArraysCache(2, left_padding=[0] * 3)
    cache.prepare(lengths=[100] * 3)
    prefix = mx.random.normal((3, 4, config.hidden_size))
    mx.eval(layer(prefix, cache=cache), cache.state)
    assert cache.history_capacity == 0

    for retained in ([0, 2, 5], [3, 1, 4]):
        inputs = mx.random.normal((3, 5, config.hidden_size))
        reference = deepcopy(cache)
        states = [list(reference.state)]
        for index in range(5):
            mx.eval(layer(inputs[:, index : index + 1], cache=reference))
            states.append(list(reference.state))
        initial_lengths = cache.lengths
        transaction = start_speculative_cache([cache], 5)
        for index in range(0, 5, step):
            mx.eval(layer(inputs[:, index : index + step], cache=cache))
        transaction.commit(retained)
        for slot in range(2):
            expected = mx.concatenate(
                [states[keep][slot][row : row + 1] for row, keep in enumerate(retained)]
            )
            assert mx.allclose(cache[slot], expected, rtol=0, atol=1e-6).item()
        assert mx.array_equal(
            cache.lengths, initial_lengths - mx.array(retained)
        ).item()
        assert cache.history_capacity == 0
        assert cache.nbytes == sum(value.nbytes for value in cache.state)


@pytest.mark.parametrize("family", ["glm", "deepseek"])
@pytest.mark.parametrize("batch", [1, 2])
def test_ordinary_verification_spans_multiple_short_blocks(family, batch):
    mx.random.seed(2127)
    if family == "glm":
        config = _tiny_glm5_next_text_config()
        config.hc_mult = 4
        model = GlmLanguageModel(config)
        capture = dict(return_hidden=True, return_shared_kv=True)
    else:
        config = _tiny_deepseek_v4_config()
        config.compress_ratios = [4]
        model = DeepseekLanguageModel(config)
        capture = dict(capture_layer_ids=[0])
    model.eval()
    caches = [_make_cache(model, [0] * batch) for _ in range(2)]
    prefix = mx.broadcast_to((mx.arange(19)[None] % 30) + 1, (batch, 19))
    for cache in caches:
        mx.eval(model(prefix, cache=cache).logits)
    tokens = mx.broadcast_to((mx.arange(13)[None] % 30) + 1, (batch, 13))
    oracle = deepcopy(caches[0])
    expected = [model(tokens[:, i : i + 1], cache=oracle, **capture) for i in range(13)]
    actual, transaction = verify_forward(model, tokens, caches[1], **capture)
    assert mx.array_equal(
        actual.logits, mx.concatenate([out.logits for out in expected], axis=1)
    ).item()
    for value, parts in zip(
        actual.hidden_states, zip(*(out.hidden_states for out in expected))
    ):
        assert mx.array_equal(value, mx.concatenate(parts, axis=1)).item()
    transaction.commit([9] * batch)
    for i in range(9):
        mx.eval(model(tokens[:, i : i + 1], cache=caches[0]).logits)
    probe = mx.full((batch, 1), 20)
    assert mx.array_equal(
        model(probe, cache=caches[0]).logits, model(probe, cache=caches[1]).logits
    ).item()


def test_ordinary_forward_failure_aborts_all_verification_parts():
    cache = ArraysCache(1)
    initial = mx.zeros((1, 1), dtype=mx.int32)
    cache[0] = initial
    calls = []

    def model(tokens, cache):
        calls.append(tokens.shape[1])
        cache[0].update_window(0, mx.concatenate([cache[0][0], tokens], axis=1), 1)
        if len(calls) == 2:
            raise RuntimeError("second forward failed")
        return LanguageModelOutput(
            logits=tokens[..., None], hidden_states=[tokens[..., None]]
        )

    with pytest.raises(RuntimeError, match="second forward failed"):
        verify_forward(model, mx.ones((1, 13), dtype=mx.int32), [cache])
    assert calls == [8, 5]
    assert cache[0] is initial
    assert not cache.is_speculating


def test_glm_verification_uses_original_modules_and_preserves_weights():
    model = GlmLanguageModel(_tiny_glm5_next_text_config())
    model.eval()
    inputs = mx.array([[1, 2, 3]])
    before = model(inputs).logits
    mx.eval(before)
    original_projection = model.model.layers[0].self_attn.qkv_proj
    assert not hasattr(model, "speculative_verify_hidden")
    caches = model.make_cache()
    result = _mtp_verify_target(model, inputs, caches, None, sample_target_tokens=False)
    hidden, transaction = result.hidden, result.rollback_state
    mx.eval(hidden)
    transaction.commit(inputs.shape[1])
    after = model(inputs).logits
    mx.eval(after)
    assert mx.array_equal(before, after).item()
    assert model.model.layers[0].self_attn.qkv_proj is original_projection
    assert isinstance(original_projection, nn.Linear)


@pytest.mark.parametrize("batch", [1, 4])
def test_glm_block_verification_matches_stepwise_bfloat16(batch):
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hc_mult = 4
    model = GlmLanguageModel(config)
    # Retain the model's required FP32 parameters.
    from mlx.utils import tree_flatten

    model.load_weights(
        [
            (name, value.astype(mx.bfloat16) if model.cast_predicate(name) else value)
            for name, value in tree_flatten(model.parameters())
        ]
    )
    model.eval()
    caches = [_make_cache(model, left_padding=[0] * batch) for _ in range(2)]
    prefix = mx.array([[1, 2, 3]] * batch)
    for cache in caches:
        mx.eval(model(prefix, cache=cache).logits)
    tokens = mx.array([[4, 5, 6, 7]] * batch)
    steps = []
    for position in range(tokens.shape[1]):
        result = _mtp_verify_target(
            model,
            tokens[:, position : position + 1],
            caches[0],
            None,
            sample_target_tokens=False,
        )
        hidden, transaction = result.hidden, result.rollback_state
        mx.eval(hidden)
        transaction.commit(1)
        steps.append(hidden)
    result = _mtp_verify_target(
        model, tokens, caches[1], None, sample_target_tokens=False
    )
    actual, transaction = result.hidden, result.rollback_state
    mx.eval(actual)
    transaction.commit(tokens.shape[1])
    expected = mx.concatenate(steps, axis=1)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [1, 2])
def test_glm_drafter_rejection_restores_pool_after_multiple_appends(batch):
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    model = GlmLanguageModel(config)
    drafter = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.reset(model, left_padding=[0] * batch if batch > 1 else None)
    hidden = mx.random.normal((batch, 1, config.hidden_size))
    mx.eval(drafter._forward_tokens(mx.array([[1]] * batch), hidden, mx.int32))
    reference = deepcopy(drafter)
    sampler = _greedy
    tokens = drafter.draft_block(
        2 if batch == 1 else mx.full((batch,), 2), hidden, None, 4, sampler, greedy=True
    )
    mx.eval(tokens)
    assert drafter._round_appended == 3
    transaction = drafter._round_transaction
    verified = mx.random.normal((batch, 4, config.hidden_size))
    for candidate in (drafter, reference):
        candidate.accept_verified_tokens_batch(
            verified, tokens, [0] * batch, [[8]] * batch, sampler, greedy=True
        )
        mx.eval(candidate.draft_eval_state())
    assert not transaction.active
    actual_pool, expected_pool = drafter._cache[0][2], reference._cache[0][2]
    assert mx.array_equal(
        mx.array(actual_pool.offset), mx.array(expected_pool.offset)
    ).item()
    assert actual_pool.remainder == expected_pool.remainder
    for actual, expected in zip(actual_pool.state, expected_pool.state):
        if actual is None or expected is None:
            assert actual is expected
        else:
            assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(drafter._seed_token, reference._seed_token).item()
    assert mx.array_equal(drafter._seed_hidden, reference._seed_hidden).item()


class _Target:
    """Minimal target that updates real recurrent and KV caches."""

    def __init__(self, token):
        self.token = token
        self.transaction = None

    def __call__(self, inputs, cache, **kwargs):
        batch, length = inputs.shape
        self.transaction = start_speculative_cache(cache, length)
        initial = cache[0][0]
        states = initial[:, None] + mx.arange(1, length + 1)[None, :, None]
        cache[0][0] = states[:, -1]
        cache[0].record_speculative_states(0, states[:, :-1], states[:, -1])
        kv = mx.zeros((batch, 1, length, 1))
        cache[1].update_and_fetch(kv, kv)
        hidden = mx.zeros((batch, length, 4))
        logits = mx.broadcast_to(mx.eye(8)[self.token], (batch, length, 8))
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[hidden],
            shared_kv_states={},
            gdn_states=self.transaction,
        )

    def speculative_verify_logits(self, inputs, cache, sampler):
        output = self(inputs, cache)
        try:
            return (
                output.hidden_states[0],
                {},
                output.gdn_states,
                sampler(output.logits),
            )
        except BaseException:
            output.gdn_states.abort()
            raise

    def rollback_speculative_cache(self, *args):
        raise AssertionError("transactions must own cache commit")


class _Drafter:
    prefer_requested_block_size = True

    def __init__(self):
        self.config = SimpleNamespace(block_size=2, target_layer_ids=[0])
        self.accept_lens = []
        self.draft_lens = []

    def reset(self, model, left_padding=None):
        return []

    def make_cache(self):
        return []

    def set_shared_kv(self, *args, **kwargs):
        pass

    def draft_block(
        self, bonus, hidden, cache, block_size, sampler, token_dtype, **kwargs
    ):
        return mx.full((hidden.shape[0], block_size - 1), 4, dtype=token_dtype)


def _round_generator(kind, batch, token, sampler):
    target = _Target(token)
    caches = [ArraysCache(1), BatchKVCache([0] * batch) if batch > 1 else KVCache()]
    caches[0][0] = mx.zeros((batch, 1))
    initial = mx.zeros((batch, 1, 2, 1))
    caches[1].update_and_fetch(initial, initial)
    functions = {
        "mtp": (_mtp_rounds, _mtp_rounds_batch),
        "dflash": (_dflash_rounds, _dflash_rounds_batch),
        "eagle3": (_eagle3_rounds, _eagle3_rounds_batch),
    }
    fn = functions[kind][batch > 1]
    kwargs = dict(
        first_bonus=1 if batch == 1 else mx.ones((batch,), dtype=mx.int32),
        max_tokens=4,
        sampler=sampler,
        draft_block_size=2,
    )
    if kind == "mtp":
        kwargs["shared_kv_states"] = {}
    generator = fn(target, _Drafter(), caches, mx.zeros((batch, 1, 4)), **kwargs)
    return generator, target, caches


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("token", [4, 7])
def test_round_commits_before_yield_and_generator_close(kind, batch, token):
    generator, target, caches = _round_generator(
        kind, batch, token, lambda x: mx.argmax(x, axis=-1)
    )
    next(generator)
    assert not target.transaction.active
    assert not caches[0].is_speculating
    retained = 2 if token == 4 else 1
    assert caches[0][0].tolist() == [[float(retained)]] * batch
    offset = caches[1].offset
    assert (offset.tolist() if isinstance(offset, mx.array) else [offset]) == [
        2 + retained
    ] * batch
    generator.close()


@pytest.mark.parametrize("kind", ["mtp", "dflash", "eagle3"])
@pytest.mark.parametrize("batch", [1, 2])
def test_failed_target_sampling_aborts_round(kind, batch):
    def fail(_):
        raise RuntimeError("injected sampler failure")

    generator, target, caches = _round_generator(kind, batch, 4, fail)
    with pytest.raises(RuntimeError, match="injected sampler failure"):
        next(generator)
    assert not target.transaction.active
    assert not caches[0].is_speculating
    assert caches[0][0].tolist() == [[0.0]] * batch
    offset = caches[1].offset
    assert (offset.tolist() if isinstance(offset, mx.array) else [offset]) == [
        2
    ] * batch


def test_mtp_target_failure_aborts_glm_draft_round(monkeypatch):
    import mlx_vlm.speculative.mtp as mtp

    config = _tiny_glm5_next_text_config()
    model = GlmLanguageModel(config)
    model.eval()
    caches = model.make_cache()
    prompt = mx.array([[1, 2]])
    output = model(prompt, cache=caches, return_hidden=True)
    mx.eval(output.logits, output.hidden_states)
    assert mtp._mtp_cache_offset_max(caches) == 2
    drafter = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.prefer_requested_block_size = True

    def fail(*args, **kwargs):
        assert drafter._round_appended == 2
        assert drafter._round_transaction.active
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(mtp, "_mtp_verify_target", fail)
    generator = mtp._mtp_rounds(
        model,
        drafter,
        caches,
        output.hidden_states[-1],
        {},
        prompt_tokens=prompt,
        first_bonus=3,
        max_tokens=8,
        sampler=lambda x: mx.argmax(x, axis=-1),
        draft_block_size=4,
        greedy_sampling=True,
    )
    with pytest.raises(RuntimeError, match="injected target failure"):
        next(generator)
    assert drafter._round_transaction is None
    assert drafter._next_position == 2
    assert drafter._seed_token is not None
    assert not drafter._cache[0][2].is_speculating
    assert mtp._mtp_cache_offset_max(caches) == 2


def test_transaction_scope_aborts_uncommitted_temporal_and_kv_updates():
    recurrent = ArraysCache(1)
    recurrent[0] = mx.zeros((1, 1))
    kv = KVCache()
    initial = mx.zeros((1, 1, 2, 1))
    kv.update_and_fetch(initial, initial)
    with pytest.raises(RuntimeError, match="injected failure"):
        with start_speculative_cache([recurrent, kv], 2):
            recurrent[0] = mx.ones((1, 1))
            kv.update_and_fetch(initial, initial)
            raise RuntimeError("injected failure")
    assert recurrent[0].item() == 0
    assert not recurrent.is_speculating
    assert kv.offset == 2


def test_unsupported_target_argmax_reuses_verified_hidden():
    from mlx_vlm.speculative.mtp import _mtp_verify_target

    calls = []
    caches = [ArraysCache(1)]
    caches[0][0] = mx.zeros((1, 1))

    def forward(inputs, cache):
        calls.append(inputs)
        transaction = start_speculative_cache(cache, inputs.shape[1])
        return mx.ones((1, 2, 4)), {}, transaction

    target = SimpleNamespace(
        speculative_verify_hidden=forward,
        speculative_argmax_from_hidden=lambda hidden: None,
        speculative_logits_from_hidden=lambda hidden: hidden,
    )
    result = _mtp_verify_target(
        target, mx.array([[1, 2]]), caches, lambda x: mx.argmax(x, axis=-1)
    )
    assert len(calls) == 1
    assert result.target_tokens.tolist() == [[0, 0]]
    result.abort()
    assert not caches[0].is_speculating


def test_glm_mtp_generation_matches_ordinary_decode():
    mx.random.seed(2127)
    config = _tiny_glm5_next_text_config()
    config.hc_mult = 4
    model = GlmLanguageModel(config)
    model.eval()
    prompt = mx.array([[1, 2, 3]])
    ordinary_cache, speculative_cache = model.make_cache(), model.make_cache()
    expected = []
    inputs = prompt
    for _ in range(8):
        token = mx.argmax(model(inputs, cache=ordinary_cache).logits[:, -1], axis=-1)
        expected.append(token.item())
        inputs = token[:, None]
    output = model(prompt, cache=speculative_cache, return_hidden=True)
    first = mx.argmax(output.logits[:, -1], axis=-1).item()
    drafter = Glm5NextMTPDraftModel(Glm5NextMTPConfig(text_config=config, block_size=4))
    drafter.eval()
    drafter.prefer_requested_block_size = True
    actual = [first] + [
        token
        for token, _ in _mtp_rounds(
            model,
            drafter,
            speculative_cache,
            output.hidden_states[-1],
            {},
            prompt_tokens=prompt,
            first_bonus=first,
            max_tokens=8,
            sampler=lambda x: mx.argmax(x, axis=-1),
            draft_block_size=4,
            greedy_sampling=True,
        )
    ]
    assert actual == expected
    assert not speculative_cache[0].is_speculating
    assert not speculative_cache[1][2].is_speculating
    assert not drafter._cache[0][2].is_speculating


KV_HEADS, KV_HEAD_DIM = 4, 64


KV_BITS = 4


def _quantized_kv(seq_len, left_padding=(0,)):
    cache = BatchTurboQuantKVCache(list(left_padding), bits=KV_BITS)
    batch = len(left_padding)
    keys, values = cache.update_and_fetch(
        mx.random.normal((batch, KV_HEADS, seq_len, KV_HEAD_DIM)),
        mx.random.normal((batch, KV_HEADS, seq_len, KV_HEAD_DIM)),
    )
    return cache, keys, values


class TestDFlashRunsOnQuantizedKv:
    """DFlash speculation decodes against the quantized cache, not fp16."""

    def test_generate_step_keeps_the_quantized_cache(self):
        target, drafter = _glimmer_target(), MuseGlimmerAssistantModel(
            _glimmer_config()
        )
        caches = []

        def make_cache(lm, left_padding):
            built = [
                BatchTurboQuantKVCache(list(left_padding), bits=KV_BITS)
                for _ in lm.language_model.layers
            ]
            caches.append(built)
            return built

        prompt_cache = make_speculative_prompt_cache(
            target,
            draft_kind="dflash",
            batch_size=1,
            left_padding=[0],
            make_cache=make_cache,
        )
        assert prompt_cache is caches[0]
        assert all(isinstance(c, BatchTurboQuantKVCache) for c in prompt_cache)

        tokens = [
            int(token.item()) if hasattr(token, "item") else int(token)
            for token, _ in generate_step(
                mx.array([[1, 2, 3, 4]], dtype=mx.int32),
                target,
                None,
                None,
                max_tokens=6,
                temperature=0,
                prefill_step_size=None,
                draft_model=drafter,
                draft_kind="dflash",
                prompt_cache=prompt_cache,
            )
        ]

        assert len(tokens) == 6
        # Decoding through a drafter must not swap the cache back to fp16.
        assert all(isinstance(c, BatchTurboQuantKVCache) for c in prompt_cache)
        assert prompt_cache[0].bits == KV_BITS


class TestQuantizedStateSlicing:
    """Target verification narrows the cache one draft token at a time."""

    def test_slice_stays_quantized(self):
        # A slice that dequantized would defeat the point of --kv-bits.
        _, keys, _ = _quantized_kv(64)
        assert type(keys[:, :, :32, :]) is type(keys)

    @pytest.mark.parametrize(
        "key",
        [
            (slice(None), slice(None), slice(5, 10), slice(None)),  # offset start
            (slice(None), 0, slice(None, 10), slice(None)),  # indexes a head
            (slice(None), slice(None), slice(None, 10, 2), slice(None)),  # strided
        ],
    )
    def test_rejects_unsupported_indexing(self, key):
        _, keys, _ = _quantized_kv(64)
        with pytest.raises(TypeError):
            keys[key]


def test_apodex_mtp_splitter_falls_back_to_root_model_type(tmp_path):
    """Apodex names its text stack separately from its architecture.

    config.json carries ``text_config.model_type`` ``qwen3_5_moe_text`` under a
    root ``qwen3_5_moe``. Consulting only the text_config finds no registered
    splitter, so the bundled MTP head cannot be extracted at all.
    """
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"mtp.fc.weight": mx.zeros((4, 4))}
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "num_hidden_layers": 4,
                },
            }
        )
    )

    splitter = detect_mtp_splitter(tmp_path)

    assert splitter is not None
    assert splitter.output_model_type == "qwen3_5_mtp"


@pytest.mark.parametrize("ragged", [False, True], ids=["trim-index", "reject-ragged"])
def test_minimax_m3_speculative_rollback_preserves_index_cache(ragged):
    cache = MiniMaxM3BatchKVCache([0, 0]) if ragged else MiniMaxM3KVCache()
    keys = mx.ones((2 if ragged else 1, 1, 5, 4), dtype=mx.float32)
    cache.update_and_fetch(keys, keys)
    cache.update_index_and_fetch(keys)
    rollback = minimax_language.LanguageModel.rollback_speculative_cache
    if ragged:
        # Ragged acceptance must not leave phantom keys in shorter rows.
        with pytest.raises(RuntimeError, match="uniform"):
            rollback(
                None, [cache], None, mx.array([2, 0], dtype=mx.int32), block_size=4
            )
    else:
        assert rollback(None, [cache], None, accepted=1, block_size=4) == 1
        assert cache.offset == cache.index_offset == 3


@pytest.mark.parametrize(
    "family,temperature,seed",
    [
        pytest.param("dflash2", 0, None, id="dflash2-greedy"),
        pytest.param("dflash2", 1.0, 17, id="dflash2-sampled"),
        pytest.param("glimmer", 0, None, id="glimmer-greedy"),
        pytest.param("dspark-moe", 0, None, id="dspark-moe-greedy"),
        *[
            pytest.param(family, temperature, 41, id=f"{family}-{temperature}")
            for family in ("dspark-lfm2", "dspark-qwen")
            for temperature in (0.25, 0.5, 1.0)
        ],
    ],
)
def test_drafter_generation_matches_baseline(family, temperature, seed):
    target_factory, drafter_cls, config_factory = {
        "dflash2": (_qwen_dflash_target, DFlash2DraftModel, _dflash2_config),
        "glimmer": (_glimmer_target, MuseGlimmerAssistantModel, _glimmer_config),
        "dspark-moe": (_lfm2_moe_target, DSparkDraftModel, _dspark_config),
        "dspark-lfm2": (_lfm2_target, DSparkDraftModel, _dspark_config),
        "dspark-qwen": (_qwen_dflash_target, DSparkDraftModel, _dspark_qwen_config),
    }[family]
    init_seed = {
        "dflash2": 7,
        "glimmer": 7,
        "dspark-moe": 19,
        "dspark-lfm2": 37,
        "dspark-qwen": 37,
    }[family]
    prompt = [1, 2, 3] if family == "glimmer" else [1, 2, 3, 4]
    max_tokens = 24 if family in {"dspark-lfm2", "dspark-qwen"} else 10
    mx.random.seed(init_seed)
    target, drafter = target_factory(), drafter_cls(config_factory())
    mx.eval(target.parameters(), drafter.parameters())
    prompt = mx.array([prompt], dtype=mx.int32)
    options = dict(max_tokens=max_tokens, temperature=temperature, seed=seed)
    baseline = _generated_tokens(target, prompt, **options)
    speculative = _generated_tokens(target, prompt, drafter, **options)
    assert speculative == baseline
    assert drafter.draft_lens


@pytest.mark.parametrize(
    "config_factory,model_type,model_class",
    [
        pytest.param(
            _dflash2_published_config, "dflash2", "DFlash2DraftModel", id="dflash2"
        ),
        pytest.param(
            _laguna_config_dict, "laguna_dflash", "LagunaDFlashDraftModel", id="laguna"
        ),
        pytest.param(
            _published_gemma4_dspark_config,
            "gemma4_dspark",
            "Gemma4DSparkDraftModel",
            id="gemma4-dspark",
        ),
    ],
)
def test_drafter_checkpoint_routes_to_backend(
    tmp_path, config_factory, model_type, model_class
):
    published = config_factory()
    architecture, actual_type = get_model_and_args(published)
    expected = importlib.import_module(f"mlx_vlm.speculative.drafters.{model_type}")
    assert actual_type == model_type
    assert architecture.Model is expected.Model
    assert architecture.Model.__name__ == model_class
    (tmp_path / "config.json").write_text(json.dumps(published))
    assert resolve_drafter_kind(tmp_path) == "dflash"
