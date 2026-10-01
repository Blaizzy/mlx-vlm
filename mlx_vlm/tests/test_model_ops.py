"""Attention, rotary embeddings, weight quantization, and format conversion."""

from __future__ import annotations

import copy
import importlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

import mlx_vlm.models.rope_utils as rope_utils
from mlx_vlm.convert import _preserve_existing_deepseek_v4_quantization
from mlx_vlm.fp8 import _dequantize_fp8_weight, _quantize_fp8_weight
from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.deepseek_v4.language import _sparse_pooled_attention
from mlx_vlm.models.deepseek_v41 import fakequant as deepseek_v41_fakequant
from mlx_vlm.models.deepseek_v41.sparse_attention import (
    sparse_attention as deepseek_v41_sparse_attention,
)
from mlx_vlm.models.mla import max_absorbed_queries
from mlx_vlm.models.paddleocr_vl.config import VisionConfig
from mlx_vlm.models.paddleocr_vl.vision import Attention, VisionModel
from mlx_vlm.models.qwen3_5 import language as lang
from mlx_vlm.models.rope_utils import (
    EagerRoPE,
    MRoPERotaryEmbedding,
    ProportionalRoPE,
    YarnRoPE,
    apply_mrope_frequency_layout,
    apply_multimodal_rotary_pos_emb,
    compute_mrope_frequencies,
    initialize_rope,
    mrope_position_selector,
)
from mlx_vlm.quantization.one_bit import (
    OneBitEmbedding,
    OneBitLinear,
    dequantize_one_bit,
    one_bit_quantized_matmul,
    replace_one_bit_modules,
)
from mlx_vlm.tests.test_models import tiny_config
from mlx_vlm.utils import (
    _transform_modelopt_nvfp4_weights,
    get_model_and_args,
    load_model,
)

prism = importlib.import_module("mlx_vlm.models.prism_hadamard_qwen35")
prism_ops = importlib.import_module(
    "mlx_vlm.models.prism_hadamard_qwen35.prism_hadamard_qwen35"
)
qwen35 = importlib.import_module("mlx_vlm.models.qwen3_5")


def test_mimo_v2_unfuses_tensor_parallel_qkv_shards():
    module = importlib.import_module("mlx_vlm.models.mimo_v2")
    text = module.TextConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=12,
        num_key_value_heads=4,
        head_dim=32,
        v_head_dim=32,
        swa_num_attention_heads=12,
        swa_num_key_value_heads=4,
        swa_head_dim=32,
        swa_v_head_dim=32,
        partial_rotary_factor=0.5,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 0],
        n_routed_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
    )
    language = module.language.LanguageModel(text)
    sections = (("q", 96), ("k", 32), ("v", 32))
    tags = {}
    rows = []
    for rank in range(4):
        for name, count in sections:
            tag = 56 + 4 * rank + {"q": 0, "k": 1, "v": 2}[name]
            tags[name, rank] = tag
            rows.append(mx.full((count, text.hidden_size), tag, dtype=mx.uint8))

    out = language._unfuse_qkv(
        {
            "model.layers.0.self_attn.qkv_proj.weight": mx.concatenate(rows),
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv": mx.ones((8, 1)),
        }
    )

    assert not any("qkv_proj" in key for key in out)
    for projection, rows_per_shard in (
        ("q_proj", 96),
        ("k_proj", 32),
        ("v_proj", 32),
    ):
        weight = out[f"model.layers.0.self_attn.{projection}.weight"]
        assert weight.shape[0] == rows_per_shard * 4
        for rank in range(4):
            block = weight[rank * rows_per_shard : (rank + 1) * rows_per_shard]
            expected = mx.from_fp8(
                mx.full((1, 1), tags[projection[0], rank], dtype=mx.uint8),
                dtype=mx.float32,
            )
            assert mx.allclose(block.astype(mx.float32), expected.astype(mx.float32))


def test_mimo_v2_batched_vision_attention_matches_independent_sequences():
    from mlx_vlm.models.mimo_v2.config import VisionConfig
    from mlx_vlm.models.mimo_v2.vision import VisionAttention

    attention = VisionAttention(
        VisionConfig(
            hidden_size=64,
            num_heads=4,
            num_key_value_heads=2,
            qk_channels=16,
        ),
        use_sinks=True,
        window_size=4,
    )
    q = mx.random.normal((3, 8, 4, 16))
    k = mx.random.normal((3, 8, 2, 16))
    v = mx.random.normal((3, 8, 2, 16))

    batched = attention._attend(q, k, v, full_attn=False)
    independent = mx.concatenate(
        [
            attention._attend(q[i : i + 1], k[i : i + 1], v[i : i + 1], False)
            for i in range(3)
        ],
        axis=0,
    )

    assert mx.allclose(batched, independent)


# Attention kernels

# Dims chosen only to steer the gate; they do not affect the attention maths.
FORCE_MATERIALIZED = (1, 1, 1)  # threshold 1
FORCE_ABSORBED = (512, 511, 511)  # threshold ~261k


def _dense_attentions():
    """Every MLA attention whose __call__ is (x, mask, cache)."""
    from mlx_vlm.models.deepseek_v3.config import ModelConfig as DSV3Config
    from mlx_vlm.models.deepseek_v3.language import DeepseekV3Attention
    from mlx_vlm.models.glm4_moe_lite.config import ModelConfig as GlmConfig
    from mlx_vlm.models.glm4_moe_lite.language import Glm4MoeLiteAttention

    common = dict(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    yield "deepseek_v3", DeepseekV3Attention(
        DSV3Config(model_type="deepseek_v3", **common)
    )
    yield "glm4_moe_lite", Glm4MoeLiteAttention(
        GlmConfig(model_type="glm4_moe_lite", **common)
    )


class TestMaxAbsorbedQueries(unittest.TestCase):
    def test_degenerate_dims_keep_the_decode_path(self):
        assert max_absorbed_queries(64, 128, 128) == 1
        assert max_absorbed_queries(1, 1, 1) >= 1


class TestGateWiring(unittest.TestCase):
    """Run the real __call__ and force each branch, so the gates are executed."""

    def _both_branches(self, attn, L, cache_len):
        mx.random.seed(0)
        outs = []
        for dims in (FORCE_MATERIALIZED, FORCE_ABSORBED):
            mx.random.seed(0)
            x = mx.random.normal((1, L, 256))
            cache = KVCache()
            if cache_len:
                warm = mx.random.normal((1, L, 256))
                mx.random.seed(0)
                attn._absorbed_dims = dims
                attn(warm, cache=cache)
            attn._absorbed_dims = dims
            outs.append(attn(x, cache=cache))
        mx.eval(outs)
        return outs

    def test_branches_agree_through_call(self):
        for name, attn in _dense_attentions():
            mx.eval(attn.parameters())
            with self.subTest(model=name):
                materialized, absorbed = self._both_branches(attn, L=4, cache_len=8)
                assert mx.allclose(
                    materialized, absorbed, atol=0.0001, rtol=0.0001
                ), f"{name}: absorbed and materialized disagree through __call__"


class TestGatePairing(unittest.TestCase):
    """The two gates must never disagree: one boolean drives both."""

    MODELS = [
        "deepseek_v3",
        "deepseek_v32",
        "kimi_k3",
        "kimi_linear",
        "longcat_flash",
        "longcat_flash_sparse",
        "glm4_moe_lite",
        "glm_moe_dsa",
        "youtu_vl",
    ]

    def test_one_boolean_two_uses(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "models"
        for m in self.MODELS:
            src = (root / m / "language.py").read_text()
            with self.subTest(model=m):
                assert (
                    src.count("absorbed = L == 1 or L <= max_absorbed_queries(") == 1
                ), f"{m}: expected exactly one gate decision"
                assert (
                    src.count("if absorbed:") == 2
                ), f"{m}: expected both gates to use it"


class TestIndexerGateUnchanged(unittest.TestCase):
    """The sparse top-k gather stays at L == 1.

    It selects with ``topk_indices[:, :, 0, :]``, the first query's top-k, so
    widening it would apply one query's selection to all of them.
    """

    SPARSE = ["deepseek_v32", "longcat_flash_sparse", "glm_moe_dsa"]

    def test_first_query_gather_is_still_gated_on_one_query(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[1] / "models"
        gate = re.compile(
            r"if L == 1:\s*\n\s*(?:clamped = mx\.clip\(topk_indices|idx = topk_indices)"
        )
        for m in self.SPARSE:
            src = (root / m / "language.py").read_text()
            with self.subTest(model=m):
                self.assertRegex(
                    src,
                    gate,
                    f"{m}: the first-query top-k gather is no longer gated on L == 1",
                )


# Ragged decode launch fallbacks

# Qwen3.5-4B text config, whose head_dim=256 is what makes the pass-2 pipeline
# expensive enough to lose threads on an affected GPU.
Q_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 256
PADS = [0, 8]  # two rows with different left padding -> a ragged batch
SCALE = HEAD_DIM**-0.5

# bf16 inputs, so the one-pass and two-pass accumulation orders differ in the
# last couple of mantissa bits.
TOLERANCE = 5e-3

REJECTION = ValueError(
    "Thread group size (1024) is greater than  the maximum allowed threads per "
    "threadgroup (896)."
)

_RAGGED_SDPA_ONLY = pytest.mark.skipif(
    not mx.metal.is_available(), reason="ragged decode is a Metal-only fast path"
)


@pytest.fixture
def clear_launchability_cache():
    """The cache is process-global; keep verdicts from leaking between tests."""
    lang._QWEN3_5_SDPA_LAUNCHABLE.clear()
    yield
    lang._QWEN3_5_SDPA_LAUNCHABLE.clear()


def two_pass_kv_len():
    """Smallest KV length where every row of the batch takes the two-pass plan.

    The planner's threshold depends on the GPU architecture, so probe it instead
    of hardcoding a length that only works on some parts.
    """
    for kv_len in (2048, 8192, 65536):
        plans = {
            lang._qwen3_5_sdpa_vector_plan(kv_len - pad, Q_HEADS, KV_HEADS)
            for pad in PADS
        }
        if len(plans) == 1 and next(iter(plans))[0] == "two_pass":
            return kv_len
    return None


def inputs(kv_len):
    mx.random.seed(0)
    batch = len(PADS)
    queries = mx.random.normal((batch, Q_HEADS, 1, HEAD_DIM)).astype(mx.bfloat16)
    keys = mx.random.normal((batch, KV_HEADS, kv_len, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((batch, KV_HEADS, kv_len, HEAD_DIM)).astype(mx.bfloat16)
    return queries, keys, values


def reference(queries, keys, values):
    """The portable per-pad-group path the caller uses when the fast path declines."""
    rows = [
        mx.fast.scaled_dot_product_attention(
            queries[i : i + 1],
            keys[i : i + 1, :, pad:],
            values[i : i + 1, :, pad:],
            scale=SCALE,
        )
        for i, pad in enumerate(PADS)
    ]
    return mx.concatenate(rows, axis=0)


def max_abs_diff(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def reject(monkeypatch, factory_name, launches):
    """Replace a kernel factory so its launches raise the way Metal does."""

    def factory(*args, **kwargs):
        def launch(**launch_kwargs):
            launches.append(factory_name)
            raise REJECTION

        return launch

    monkeypatch.setattr(lang, factory_name, factory)


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
def test_rejected_two_pass_degrades_to_one_pass(monkeypatch, two_pass_inputs):
    queries, keys, values = two_pass_inputs
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_two_pass_2_kernel", [])

    out = lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)

    assert out is not None, "should fall back to the one-pass kernel, not decline"
    assert out.shape == (len(PADS), Q_HEADS, 1, HEAD_DIM)
    assert max_abs_diff(out, reference(queries, keys, values)) < TOLERANCE


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
def test_rejection_is_probed_once(monkeypatch, two_pass_inputs):
    queries, keys, values = two_pass_inputs
    launches = []
    reject(monkeypatch, "_qwen3_5_ragged_sdpa_two_pass_2_kernel", launches)

    for _ in range(3):
        assert (
            lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)
            is not None
        )

    assert launches == [
        "_qwen3_5_ragged_sdpa_two_pass_2_kernel"
    ], "a rejected pipeline must be probed once and remembered, not retried per call"


@_RAGGED_SDPA_ONLY
@pytest.mark.usefixtures("clear_launchability_cache")
@pytest.mark.parametrize("kv_len", [512, 2048])
def test_launchable_pipelines_are_untouched(kv_len):
    queries, keys, values = inputs(kv_len)

    out = lang._qwen3_5_ragged_decode_attention(queries, keys, values, PADS, SCALE)

    if out is None:
        pytest.skip("this GPU declines the fast path for this shape")
    # Whichever kernel this GPU ends up on -- one-pass, two-pass, or one-pass
    # after a real rejection of the two-pass reduction -- the answer must match.
    assert max_abs_diff(out, reference(queries, keys, values)) < TOLERANCE
    assert lang._QWEN3_5_SDPA_LAUNCHABLE, "the dispatch should record what it probed"


def _assert_allclose(actual, expected, atol=1e-5, rtol=1e-5):
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual), np.array(expected), atol=atol, rtol=rtol
    )


def _tiny_vision_model():
    return VisionModel(
        VisionConfig(
            model_type="paddleocr_vl",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=4,
            patch_size=2,
            spatial_merge_size=1,
        )
    )


def test_paddle_attention_uses_no_mask_for_single_segment():
    attention = Attention(dim=8, num_heads=2)
    hidden_states = mx.random.uniform(shape=(4, 8))
    seen = {}

    def sdpa(q, k, v, scale=None, mask=None):
        seen["mask"] = mask
        return mx.zeros_like(q)

    with patch.object(mx.fast, "scaled_dot_product_attention", side_effect=sdpa):
        output = attention(
            hidden_states, mx.array([0, 4], dtype=mx.int32), mx.zeros((4, 2))
        )
        mx.eval(output)
    assert seen["mask"] is None


@pytest.mark.parametrize(
    "second_grid,fast_path",
    [([1, 2, 2], True), ([1, 1, 4], False)],
    ids=["same-grid", "mixed-grid"],
)
def test_paddle_batch_matches_packed_fallback(second_grid, fast_path):
    model = _tiny_vision_model()
    pixels = mx.random.uniform(shape=(2, 4, 3, 2, 2))
    grid = mx.array([[1, 2, 2], second_grid], dtype=mx.int32)
    assert model._use_same_grid_batch_path(pixels, grid, False) == fast_path
    output = model(pixels, grid, output_hidden_states=False)
    expected = model(pixels.reshape(1, 8, 3, 2, 2), grid, output_hidden_states=False)
    assert output.shape == (8, 1024)
    _assert_allclose(output, expected)


@pytest.fixture
def two_pass_inputs():
    kv_len = two_pass_kv_len()
    if kv_len is None:
        pytest.skip("no KV length on this GPU selects the two-pass plan")
    return inputs(kv_len)


# DeepSeek V4.1 sparse attention and expert routing


def _deepseek_v41_sparse_inputs(
    batch=1,
    heads=4,
    length=17,
    dim=512,
    history=7,
    topk=31,
    dtype=mx.bfloat16,
    window_size=16,
    pool_length=97,
):
    mx.random.seed(71)
    q = (
        mx.random.normal((batch, length, heads, dim))
        .astype(dtype)
        .transpose(0, 2, 1, 3)
    )
    window = mx.random.normal((batch, length + history, dim)).astype(dtype)
    pool = mx.random.normal((batch, pool_length, dim)).astype(dtype)
    indices = mx.random.randint(0, pool_length, (batch, length, topk))
    indices = mx.where(mx.arange(topk) % 5 == 0, -1, indices)
    positions = mx.arange(length + history)
    ends = mx.arange(length) + history
    mask = ((positions <= ends[:, None]) & (positions > ends[:, None] - window_size))[
        None, None
    ]
    sinks = mx.linspace(-4, 6, heads)
    return q, window, pool, indices, mask, sinks, dim**-0.5, window_size


def _deepseek_v41_sparse_reference(args):
    q, window, pool, indices, mask, sinks, scale, _ = args
    return _sparse_pooled_attention(
        q.astype(mx.float32),
        window[:, None],
        pool,
        indices,
        mask,
        (indices != -1)[:, None],
        scale,
        sinks.astype(mx.float32),
    ).astype(q.dtype)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "batch,length,heads,dim,topk,history",
    [
        (1, 1, 64, 512, 512, 127),
        (4, 1, 64, 512, 512, 127),
        (1, 17, 4, 512, 31, 0),
        (4, 33, 8, 64, 15, 7),
        (1, 129, 4, 512, 512, 127),
        (1, 17, 64, 512, 33, 0),
        (1, 129, 8, 512, 512, 127),
    ],
)
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_matches_reference(
    dtype, batch, length, heads, dim, topk, history
):
    args = _deepseek_v41_sparse_inputs(
        batch, heads, length, dim, history, topk, dtype, 128
    )
    actual, expected = deepseek_v41_sparse_attention(
        *args
    ), _deepseek_v41_sparse_reference(args)
    assert actual is not None
    mx.eval(actual, expected)
    # FP32 reduction ordering can move the final half-precision rounding by one ULP.
    tolerance = 0.004 if dtype == mx.bfloat16 else 0.0005
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()


@pytest.mark.parametrize("sink", [-1000.0, 0.0, 1000.0])
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_masked_rows_and_attention_sinks(sink, heads):
    args = list(_deepseek_v41_sparse_inputs(batch=2, heads=heads, length=3, dim=64))
    args[3] = mx.full(args[3].shape, -1, mx.int64)
    args[4] = mx.zeros((2, 1, 3, args[1].shape[1]), mx.bool_)
    args[5] = mx.full((heads,), sink)
    out = deepseek_v41_sparse_attention(*args)
    assert mx.all(mx.isfinite(out)).item()
    assert mx.all(out == 0).item()


@pytest.mark.parametrize("heads", [3, 8])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_float64_oracle_with_duplicate_indices_and_batch_mask(
    heads,
):
    args = list(
        _deepseek_v41_sparse_inputs(batch=2, heads=heads, length=3, dim=64, topk=4)
    )
    args[3] = mx.broadcast_to(mx.array([3, 3, -1, 5]), (2, 3, 4))
    args[4] = mx.concatenate([args[4], mx.zeros_like(args[4])], axis=0)
    q, window, pool, indices, mask, sinks = [
        np.array(a.astype(mx.float32)) for a in args[:6]
    ]
    expected = np.zeros_like(q, dtype=np.float64)
    for b in range(2):
        for h in range(heads):
            for t in range(3):
                keys = np.concatenate(
                    [
                        window[b, mask[b, 0, t].astype(bool)],
                        pool[b, indices[b, t][indices[b, t] >= 0].astype(int)],
                    ]
                )
                scores = keys.astype(np.float64) @ (
                    q[b, h, t].astype(np.float64) * args[6]
                )
                maximum = max(scores.max(), sinks[h])
                weights = np.exp(scores - maximum)
                expected[b, h, t] = (
                    weights @ keys / (weights.sum() + np.exp(sinks[h] - maximum))
                )
    actual = deepseek_v41_sparse_attention(*args)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, atol=0.004, rtol=0.004
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_empty_pool_and_no_selected_keys():
    args = list(_deepseek_v41_sparse_inputs(length=3, topk=0))
    args[2] = args[2][:, :0]
    actual = deepseek_v41_sparse_attention(*args)
    q, window, _, _, mask, sinks, scale, _ = args
    scores = (q.astype(mx.float32) * scale) @ window[:, None].astype(
        mx.float32
    ).swapaxes(-1, -2)
    scores = mx.where(mask, scores, -mx.inf)
    probabilities = mx.softmax(
        mx.concatenate(
            [
                scores,
                mx.broadcast_to(sinks[None, :, None, None], (*scores.shape[:-1], 1)),
            ],
            -1,
        ),
        -1,
    )
    expected = probabilities[..., :-1] @ window[:, None].astype(mx.float32)
    assert mx.allclose(
        actual.astype(mx.float32), expected, atol=0.004, rtol=0.004
    ).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_unsupported_inputs_return_none():
    args = list(_deepseek_v41_sparse_inputs())
    args[0] = args[0].astype(mx.float32)
    assert deepseek_v41_sparse_attention(*args) is None
    args = list(_deepseek_v41_sparse_inputs())
    args[4] = args[4].astype(mx.float32)
    assert deepseek_v41_sparse_attention(*args) is None
    args = list(_deepseek_v41_sparse_inputs(dim=48))
    assert deepseek_v41_sparse_attention(*args) is None


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_sparse_attention_split_decode_matches_original_reduction_exactly(
    dtype, batch, masked
):
    args = list(_deepseek_v41_sparse_inputs(batch, 8, 1, 512, 127, 512, dtype, 128))
    # Include repeated keys, invalid routes, batch-specific masks, and sink extremes.
    args[3] = mx.broadcast_to(mx.array([3, 3, -1, 5] * 128), (batch, 1, 512))
    args[4] = mx.broadcast_to(args[4], (batch, 1, 1, 128))
    args[4] = args[4] & (mx.arange(batch)[:, None, None, None] % 2 == 0)
    args[5] = mx.array([-1000, -10, -1, 0, 1, 10, 100, 1000], mx.float32)
    if masked:
        args[3] = mx.full(args[3].shape, -1, mx.int32)
        args[4] = mx.zeros_like(args[4])
    actual = deepseek_v41_sparse_attention(*args)
    # Batch 1 retains the original one-threadgroup decode kernel.
    expected = mx.concatenate(
        [
            deepseek_v41_sparse_attention(
                *(a[row : row + 1] for a in args[:5]), *args[5:]
            )
            for row in range(batch)
        ],
        axis=0,
    )
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("owned", ["none", "some", "all"])
def test_deepseek_v41_device_routes_match_compact_owned_routes(dtype, owned):
    from mlx_vlm.models.deepseek_v41.language import DeepseekV41MoE

    mx.random.seed(43)
    config = tiny_config("deepseek_v41")
    model = DeepseekV41MoE(config)
    model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    model.sharding_group = SimpleNamespace(rank=lambda: 1)
    count = model.switch_mlp.gate_proj.weight.shape[0]
    x = mx.random.normal((4, 1, config.hidden_size)).astype(dtype)
    if owned == "none":
        indices = mx.zeros((4, 1, 2), mx.int32)
    elif owned == "all":
        indices = mx.full((4, 1, 2), count, mx.int32)
    else:
        indices = mx.broadcast_to(mx.array([0, count + 1]), (4, 1, 2))
    scores = mx.random.uniform(shape=indices.shape)
    expected = []
    # Independent route-by-route oracle, including duplicate and zero-owned routes.
    for row in range(4):
        value = mx.zeros((1, 1, config.hidden_size), mx.float32)
        for route in range(2):
            index = indices[row, 0, route].item() - count
            if 0 <= index < count:
                y = model.switch_mlp(
                    x[row : row + 1],
                    mx.array([[[index]]]),
                    scores[row : row + 1, :, route : route + 1],
                )
                value = value + y[..., 0, :].astype(mx.float32)
        expected.append(value)
    with patch.object(mx.distributed, "all_sum", side_effect=lambda data, **_: data):
        actual = model._distributed_decode_experts(x, indices, scores)
    expected = mx.concatenate(expected)
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()


# Rotary embeddings

# Multimodal position IDs

QWEN_STYLE_MODULES = [
    "mlx_vlm.models.glm4v.language",
    "mlx_vlm.models.glm4v_moe.language",
    "mlx_vlm.models.paddleocr_vl.language",
    "mlx_vlm.models.qwen2_vl.language",
    "mlx_vlm.models.qwen2_5_vl.language",
    "mlx_vlm.models.qwen3_5.language",
    "mlx_vlm.models.qwen3_omni_moe.language",
    "mlx_vlm.models.qwen3_vl.language",
    "mlx_vlm.models.qwen3_vl_moe.language",
]


def _mrope_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
        vision_start_token_id=100,
    )


@pytest.mark.parametrize("module_name", QWEN_STYLE_MODULES)
def test_mrope_rope_index_handles_fully_masked_rows(module_name):
    module = importlib.import_module(module_name)
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = _mrope_config()

    input_ids = mx.array([[0, 0, 0, 0], [10, 100, 101, 11]], dtype=mx.int32)
    attention_mask = mx.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (3, 2, 4)
    assert rope_deltas.shape == (2, 1)
    assert rope_deltas.tolist()[0] == [0]


def test_glm_ocr_rope_index_handles_fully_masked_rows():
    module = importlib.import_module("mlx_vlm.models.glm_ocr.language")
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
    )

    input_ids = mx.array([[0, 0, 0, 0], [10, 101, 11, 12]], dtype=mx.int32)
    attention_mask = mx.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=mx.int32)
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (3, 2, 4)
    assert rope_deltas.shape == (2, 1)
    assert rope_deltas.tolist()[0] == [0]


def test_ernie_mrope_rope_index_handles_empty_rows():
    module = importlib.import_module("mlx_vlm.models.ernie4_5_moe_vl.language")
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = _mrope_config()

    input_ids = mx.zeros((2, 0), dtype=mx.int32)
    image_grid_thw = mx.zeros((0, 3), dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids, image_grid_thw=image_grid_thw
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (2, 0, 3)
    assert rope_deltas.shape == (2,)
    assert rope_deltas.tolist() == [0, 0]


# Rotary implementations and batched offsets


def _max_diff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def _assert_pair_close(actual, expected, *, atol=1e-4):
    mx.eval(*actual, *expected)
    assert _max_diff(actual[0], expected[0]) < atol
    assert _max_diff(actual[1], expected[1]) < atol


def _disable_metal_fast_path(fn):
    has_metal = rope_utils._HAS_METAL
    rope_utils._HAS_METAL = False
    try:
        return fn()
    finally:
        rope_utils._HAS_METAL = has_metal


def _position_ids(batch=2, seq_len=4):
    base = mx.arange(batch * seq_len, dtype=mx.int32).reshape(batch, seq_len)
    return mx.stack([base, base + 3, base + 7])


def test_eager_rope_per_batch_offset_does_not_expand_seq():
    """Per-batch (array) offsets from batch caches must keep S, not grow it.

    Regression for the batched-path crash on multi-request streaming:
    ``_eager_rope_angles`` used ``arange(S) + offset`` which broadcast the
    batch-sized offset along the seq dim (S=1 -> S=B), so the attention mask
    (N=1) no longer matched the keys (S=B) and
    ``mx.fast.scaled_dot_product_attention`` raised
    ``Shapes (B,1,1,window) and (B,H,B,window+1) cannot be broadcast``.
    """
    rope = initialize_rope(
        dims=128,
        base=500000.0,
        traditional=False,
        scaling_config={"rope_type": "default"},
        implementation="eager",
    )
    assert isinstance(rope, EagerRoPE)

    rng = mx.random.key(0)
    x = mx.random.normal(key=rng, shape=(2, 32, 1, 128)).astype(mx.float32)
    offsets = mx.array([86, 34258], dtype=mx.int32)

    out = rope(x, offset=offsets)
    assert out.shape == x.shape, f"seq dim expanded: {out.shape} != {x.shape}"

    # Per-batch array offset must equal per-row scalar application.
    ref = mx.concatenate([rope(x[0:1], offset=86), rope(x[1:2], offset=34258)], axis=0)
    assert bool(mx.array_equal(out, ref).item())

    # Multi-token batch: (B, S) positions, still per-row equal to scalars.
    x2 = mx.random.normal(key=mx.random.key(1), shape=(2, 32, 3, 128)).astype(
        mx.float32
    )
    out2 = rope(x2, offset=offsets)
    ref2 = mx.concatenate(
        [rope(x2[0:1], offset=86), rope(x2[1:2], offset=34258)], axis=0
    )
    assert bool(mx.array_equal(out2, ref2).item())

    # Traditional layout follows the same rule.
    rope_t = initialize_rope(
        dims=128,
        base=500000.0,
        traditional=True,
        scaling_config={"rope_type": "default"},
        implementation="eager",
    )
    out3 = rope_t(x, offset=offsets)
    assert out3.shape == x.shape
    ref3 = mx.concatenate(
        [rope_t(x[0:1], offset=86), rope_t(x[1:2], offset=34258)], axis=0
    )
    assert bool(mx.array_equal(out3, ref3).item())

    # 0-d array offset (mx.array(86)) takes the same single code path as a
    # scalar int and must stay bit-identical to it.
    out4 = rope(x, offset=mx.array(86, dtype=mx.float32))
    ref4 = rope(x, offset=86)
    assert bool(mx.array_equal(out4, ref4).item())


def test_proportional_rope_evals_private_helper_arrays_on_init(monkeypatch):
    eval_args = []
    monkeypatch.setattr(mx, "eval", lambda *args: eval_args.append(args))

    class Host(nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = ProportionalRoPE(
                dims=8, scaling_config={"partial_rotary_factor": 0.25}
            )

    host = Host()

    assert isinstance(host.rope, nn.Module)
    assert tree_flatten(host.parameters()) == []
    assert tree_flatten(host.trainable_parameters()) == []
    eager_arrays = host.rope.eager_eval_arrays()
    assert eager_arrays[0] is host.rope.freqs
    assert len(eval_args) == 1
    assert eval_args[0][0] is eager_arrays[0]


# TODO: Refactor this file into separate test classes for each RoPE variant.
@pytest.mark.parametrize(
    "style",
    [
        "chunked",
        "interleaved",
        "sectioned_half_split",
        "sectioned_even_odd",
        "split_select",
    ],
)
def test_mrope_apply_rotary_fast_path_matches_fallback(style):
    mx.random.seed(0)
    q = mx.random.normal((2, 3, 4, 10)).astype(mx.float32)
    k = mx.random.normal((2, 2, 4, 10)).astype(mx.float32)
    position_ids = _position_ids()
    kwargs = {"dim": 8, "base": 10000, "mrope_section": [2, 1, 1], "style": style}

    rotary = MRoPERotaryEmbedding(**kwargs)
    fast = rotary.apply_rotary(q, k, position_ids)

    fallback = MRoPERotaryEmbedding(**kwargs)
    fallback.fused_apply = False
    expected = fallback.apply_rotary(q, k, position_ids)

    _assert_pair_close(fast, expected)
    _assert_pair_close((fast[0][..., 8:], fast[1][..., 8:]), (q[..., 8:], k[..., 8:]))


def test_sectioned_mrope_requires_three_sections():
    q = mx.zeros((1, 1, 2, 8))
    k = mx.zeros((1, 1, 2, 8))
    cos = mx.zeros((3, 1, 2, 8))
    sin = mx.zeros((3, 1, 2, 8))

    with pytest.raises(ValueError, match="exactly 3 sections"):
        _disable_metal_fast_path(
            lambda: apply_multimodal_rotary_pos_emb(
                q, k, cos, sin, mrope_section=[2, 2], style="sectioned_half_split"
            )
        )


@pytest.mark.parametrize(
    ("style", "expected_selector"),
    [
        ("chunked", [0, 0, 1, 1, 2, 2]),
        ("interleaved", [0, 1, 2, 0, 1, 2]),
        ("split_select", [0, 0, 1, 1, 2, 2]),
    ],
)
def test_selected_frequency_fast_path_matches_layout_helper(style, expected_selector):
    mx.random.seed(3)
    position_ids = _position_ids(batch=2, seq_len=4)
    mrope_section = [2, 2, 2]
    inv_freq = mx.random.normal((sum(mrope_section),)).astype(mx.float32)
    position_selector = mrope_position_selector(style, mrope_section, inv_freq.shape[0])
    assert position_selector.tolist() == expected_selector

    fast = compute_mrope_frequencies(
        position_ids,
        inv_freq,
        mrope_section,
        style=style,
        position_selector=position_selector,
    )
    freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
    layout = apply_mrope_frequency_layout(freqs, mrope_section, style=style)

    mx.eval(fast, layout)
    assert _max_diff(fast, layout) < 1e-4


def _build_on_worker(factory):
    """Build on one thread, use on another, as the server does."""
    with ThreadPoolExecutor(max_workers=1) as loader:
        rope = loader.submit(factory).result()

    def use():
        return mx.eval(rope(mx.ones((1, 2, 4, 8))))

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(use).result()


def test_yarn_rope_runs_on_a_thread_it_was_not_built_on():
    _build_on_worker(lambda: YarnRoPE(dims=8, traditional=False, base=10000.0))


# Weight quantization

# FP8 weights


def _source_fp8_pair(rows=130, cols=160):
    values = mx.random.uniform(low=-4, high=4, shape=(rows, cols))
    weight = mx.to_fp8(values)
    scales = mx.array([[0.00017, 0.00023], [0.00031, 0.00041]], dtype=mx.bfloat16)
    return weight, scales


def test_fp8_reconstruction_requantizes_to_native_mxfp8():
    weight, scale_inv = _source_fp8_pair()
    restored = _dequantize_fp8_weight(weight, scale_inv)

    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    expanded_scales = mx.repeat(mx.repeat(scale_inv, 128, axis=0), 128, axis=1)[
        : weight.shape[0], : weight.shape[1]
    ]
    direct_restored = decoded * expanded_scales
    expected_weight, expected_scales = mx.quantize(
        direct_restored, group_size=32, bits=8, mode="mxfp8"
    )

    actual_weight, actual_scales = _quantize_fp8_weight(weight, scale_inv)
    mx.eval(
        restored,
        direct_restored,
        expected_weight,
        expected_scales,
        actual_weight,
        actual_scales,
    )

    assert mx.array_equal(restored, direct_restored).item()
    assert mx.array_equal(actual_weight, expected_weight).item()
    assert mx.array_equal(actual_scales, expected_scales).item()
    assert actual_weight.dtype == mx.uint32
    assert actual_weight.shape == (130, 40)
    assert actual_scales.dtype == mx.uint8
    assert actual_scales.shape == (130, 5)


# DeepSeek V4.1 activation quantization and native weight packing


def _assert_deepseek_v41_fp4_matches_reference(x, kind, block):
    fn = getattr(deepseek_v41_fakequant, "fake_quant_fp4_" + kind)
    actual = fn(x, block)
    with patch.object(deepseek_v41_fakequant, "_fp4_roundtrip_kernel", None):
        expected = fn(x, block)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("kind,block", [("ue8m0", 32), ("e4m3", 16)])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_fp4_random_finite_inputs(dtype, kind, block):
    mx.random.seed(197)
    for scale in [0, 1e-38, 1e-5, 0.2, 1, 20, 100, 3000]:
        x = (mx.random.normal((4, 512)) * scale).astype(dtype)
        _assert_deepseek_v41_fp4_matches_reference(x, kind, block)
        _assert_deepseek_v41_fp4_matches_reference(
            x.reshape(4, 16, 32).transpose(1, 0, 2), kind, block
        )
    _assert_deepseek_v41_fp4_matches_reference(mx.zeros((block,), dtype), kind, block)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("kind,block", [("ue8m0", 32), ("e4m3", 16)])
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_fp4_rounding_boundaries(dtype, kind, block):
    ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    neighbors = np.concatenate(
        [np.nextafter(ties, -np.inf), ties, np.nextafter(ties, np.inf)]
    )
    rows = []
    for scale in [2**-9, 0.125, 0.5, 1, 8, 256, 448]:
        for value in neighbors:
            # The anchor controls amax while the remaining values cross FP4 ties.
            row = np.full(block, value * scale, np.float32)
            row[1::2] *= -1
            row[0] = 6 * scale
            rows.append(row)
    if kind == "e4m3":
        # Scale halfway points and both neighboring FP32 values.
        for exponent in range(-9, 9):
            for mantissa in range(8):
                tie = np.float32((1 + (mantissa + 0.5) / 8) * 2**exponent)
                for scale in [
                    np.nextafter(tie, -np.inf),
                    tie,
                    np.nextafter(tie, np.inf),
                ]:
                    rows.append(
                        np.linspace(-6 * scale, 6 * scale, block, dtype=np.float32)
                    )
    _assert_deepseek_v41_fp4_matches_reference(
        mx.array(np.stack(rows)).astype(dtype), kind, block
    )


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_deepseek_v41_fp4_fallback_and_disabled_quantization():
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 13
    for kind in ["ue8m0", "e4m3"]:
        _assert_deepseek_v41_fp4_matches_reference(x, kind, 64)
        with patch.object(deepseek_v41_fakequant, "DISABLE", True):
            assert getattr(deepseek_v41_fakequant, "fake_quant_fp4_" + kind)(x) is x
    with pytest.raises(ValueError, match="not divisible"):
        deepseek_v41_fakequant.fake_quant_fp4_e4m3(x[:, :63])


@pytest.mark.parametrize("bits", [4, 8])
def test_deepseek_v41_native_weight_repacking_preserves_bytes_and_block_scales(bits):
    from mlx_vlm.models.deepseek_v41.deepseek_v41 import _pack_source_weight

    rows, dims = 33, 64
    weight = mx.arange(rows * dims * bits // 8, dtype=mx.uint8).reshape(rows, -1)
    scale_rows = rows if bits == 4 else 2
    scales = mx.arange(scale_rows * 2, dtype=mx.uint8).reshape(scale_rows, 2)
    packed, expanded, mode = _pack_source_weight(weight, scales)
    assert packed.dtype == mx.uint32
    assert packed.shape == (rows, dims * bits // 32)
    assert mx.array_equal(packed.view(mx.uint8), weight).item()
    expected = scales if bits == 4 else mx.repeat(scales, 32, axis=0)[:rows]
    assert mx.array_equal(expanded, expected).item()
    assert mode == f"mxfp{bits}"


# One-bit weights


def _pack_bits(bits: np.ndarray) -> mx.array:
    packed = np.zeros((*bits.shape[:-1], bits.shape[-1] // 32), dtype=np.uint32)
    for shift in range(32):
        packed |= bits[..., shift::32].astype(np.uint32) << shift
    return mx.array(packed)


@pytest.mark.parametrize(
    "group,input_dims,output_dims,shape,seed",
    [
        *[(group, 512, 7, (2, 9, 512), 71 + group) for group in (32, 64, 128)],
        (64, 128, 128, (1, 128), 117),
    ],
    ids=["prompt-32", "prompt-64", "prompt-128", "wide-decode"],
)
def test_one_bit_matmul_matches_dense(group, input_dims, output_dims, shape, seed):
    rng = np.random.default_rng(seed)
    weight = _pack_bits(
        rng.integers(0, 2, size=(output_dims, input_dims), dtype=np.uint32)
    )
    scales, biases = [
        mx.array(rng.normal(size=(output_dims, input_dims // group)).astype(np.float32))
        for _ in range(2)
    ]
    x = mx.array(rng.normal(size=shape).astype(np.float32))
    out = one_bit_quantized_matmul(x, weight, scales, biases, group_size=group)
    reference = x @ dequantize_one_bit(weight, scales, biases, group).T
    mx.eval(out, reference)
    assert out.shape == (*shape[:-1], output_dims)
    assert mx.allclose(out, reference, rtol=1e-5, atol=1e-4).item()


def test_one_bit_linear_applies_output_bias():
    layer = OneBitLinear(64, 3, bias=True, group_size=64)
    layer.weight = _pack_bits(np.ones((3, 64), dtype=np.uint32))
    layer.scales = mx.ones((3, 1))
    layer.biases = mx.zeros((3, 1))
    layer.bias = mx.array([1.0, 2.0, 3.0])
    x = mx.ones((2, 64))

    out = layer(x)
    mx.eval(out)

    assert mx.array_equal(out, mx.array([[65.0, 66.0, 67.0]] * 2)).item()


def test_one_bit_embedding_lookup_and_linear_projection():
    embedding = OneBitEmbedding(3, 64, group_size=64)
    codes = np.stack([np.zeros(64, dtype=np.uint32), np.ones(64, dtype=np.uint32)] * 2)[
        :3
    ]
    embedding.weight = _pack_bits(codes)
    embedding.scales = mx.ones((3, 1)) * 2
    embedding.biases = mx.ones((3, 1)) * -1

    lookup = embedding(mx.array([0, 1]))
    projection = embedding.as_linear(mx.ones((1, 64)))
    mx.eval(lookup, projection)

    assert mx.array_equal(lookup[0], mx.full((64,), -1.0)).item()
    assert mx.array_equal(lookup[1], mx.full((64,), 1.0)).item()
    assert mx.array_equal(projection, mx.array([[-64.0, 64.0, -64.0]])).item()


def test_replace_one_bit_checkpoint_modules_only():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 8, bias=False)
            self.unquantized = nn.Linear(64, 8, bias=False)
            self.embedding = nn.Embedding(16, 64)

    model = Model()
    weights = {"proj.scales": mx.zeros((8, 1)), "embedding.scales": mx.zeros((16, 1))}
    replace_one_bit_modules(
        model, {"group_size": 64, "bits": 1, "mode": "affine"}, weights
    )

    assert isinstance(model.proj, OneBitLinear)
    assert isinstance(model.embedding, OneBitEmbedding)
    assert isinstance(model.unquantized, nn.Linear)


# Loading and utility contracts

# General utilities


@pytest.mark.parametrize(
    ("quant_method", "quant_algo"),
    [("modelopt", "NVFP4"), ("modelopt", "W4A16_NVFP4"), ("modelopt_mixed", "NVFP4")],
)
def test_transform_modelopt_nvfp4_weights(quant_method, quant_algo):
    weights = _nvfp4_weights() | {"layer.bias": mx.ones((2,))}

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": quant_method, "quant_algo": quant_algo}
    )

    assert transformed["layer.weight"].dtype == mx.uint32
    assert transformed["layer.weight"].shape == (2, 4)
    assert transformed["layer.scales"].tolist() == [[48, 56], [64, 72]]
    assert mx.array_equal(transformed["layer.bias"], weights["layer.bias"])
    assert "layer.weight_scale" not in transformed
    assert "layer.weight_scale_2" not in transformed
    assert "layer.input_scale" not in transformed
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


def _nvfp4_weights():
    return {
        "layer.weight": mx.arange(32, dtype=mx.uint8).reshape(2, 16),
        "layer.weight_scale": mx.array([[56, 64], [72, 80]], dtype=mx.uint8),
        "layer.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        "layer.input_scale": mx.array(0.25, dtype=mx.float32),
    }


def test_transform_modelopt_mixed_nvfp4_fp8_weights():
    weights = {
        "experts.weight": mx.arange(32, dtype=mx.uint8).reshape(2, 16),
        "experts.weight_scale": mx.array([[56, 64], [72, 80]], dtype=mx.uint8),
        "experts.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        "experts.input_scale": mx.array(0.25, dtype=mx.float32),
        "attention.weight": mx.array([[56, 64], [68, 72]], dtype=mx.uint8),
        "attention.weight_scale": mx.array([0.5, 0.25], dtype=mx.bfloat16),
        "attention.input_scale": mx.array(0.125, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
    )

    assert transformed["experts.weight"].dtype == mx.uint32
    assert transformed["experts.scales"].dtype == mx.uint8
    assert transformed["attention.weight"].dtype == mx.bfloat16
    assert transformed["attention.weight"].tolist() == [[0.5, 1.0], [0.75, 1.0]]
    assert not any("weight_scale" in key or "input_scale" in key for key in transformed)
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


@pytest.mark.parametrize(
    "width, group, mode, predicate, expected_group, expected_bits, expected_mode, selected",
    [
        (64, 64, "affine", None, 64, 4, "affine", (True, True)),
        (160, 64, "affine", {"fallback_group_size": 32}, 32, 4, "affine", (True, True)),
        (160, 16, "nvfp4", {"fallback_group_size": 32}, 16, 4, "nvfp4", (True, True)),
        (
            128,
            16,
            "nvfp4",
            {"group_size": 64, "bits": 8},
            64,
            8,
            "affine",
            (True, True),
        ),
        (
            128,
            32,
            "mxfp4",
            {"group_size": 64, "bits": 8},
            64,
            8,
            "affine",
            (True, True),
        ),
        (
            96,
            64,
            "affine",
            {"group_size": 32, "bits": 8},
            64,
            4,
            "affine",
            (False, False),
        ),
        (64, 32, "mxfp4", None, 32, 4, "mxfp4", (True, True)),
        (64, 64, "affine", "skip_vision", 64, 4, "affine", (True, False)),
    ],
    ids=[
        "basic",
        "fallback-group",
        "native-group",
        "nvfp4-affine-override",
        "mxfp4-affine-override",
        "incompatible-group",
        "mxfp4",
        "skip-vision",
    ],
)
def test_quantize_module(
    width,
    group,
    mode,
    predicate,
    expected_group,
    expected_bits,
    expected_mode,
    selected,
):
    from mlx_vlm.quant_utils import quantize_model

    module = nn.Module()
    module.language_model = nn.Linear(width, width)
    module.vision_model = nn.Linear(width, width)
    kwargs = {}
    if predicate is not None:
        kwargs["quant_predicate"] = (
            (lambda path, _: "vision_model" not in path)
            if predicate == "skip_vision"
            else (lambda *_: predicate)
        )
    _, config = quantize_model(
        module, {}, group_size=group, bits=4, mode=mode, **kwargs
    )
    defaults = dict(group_size=group, bits=4, mode=mode)
    assert {k: config["quantization"][k] for k in defaults} == defaults
    for name, enabled in zip(("language_model", "vision_model"), selected):
        layer = getattr(module, name)
        assert hasattr(layer, "scales") == enabled
        if enabled:
            assert (layer.group_size, layer.bits, layer.mode) == (
                expected_group,
                expected_bits,
                expected_mode,
            )
            assert layer.scales.shape == (width, width // expected_group)
            if isinstance(predicate, dict):
                expected = {"group_size": expected_group, "bits": expected_bits}
                if "fallback_group_size" in predicate:
                    expected["mode"] = expected_mode
                assert config["quantization"][name] == expected
    if not isinstance(predicate, dict) or not any(selected):
        assert config["quantization"] == defaults


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
def test_convert_preserves_existing_deepseek_v4_quantization(model_type):
    config = {
        "model_type": model_type,
        "quantization_config": {"quant_method": "fp8"},
    }
    existing_quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.model.layers.0.attn.wkv": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
        },
    }

    with patch(
        "mlx_vlm.convert._language_model_quantization_config",
        return_value=MagicMock(return_value=existing_quantization),
    ):
        _preserve_existing_deepseek_v4_quantization(
            config, model=MagicMock(), q_group_size=64, q_bits=4, q_mode="affine"
        )

    assert config["quantization"] is config["quantization_config"]
    assert config["quantization"]["group_size"] == 64
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["mode"] == "affine"
    assert config["quantization"]["language_model.model.layers.0.attn.wkv"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }


def test_modelopt_mixed_drops_fp8_kv_cache_scales():
    """ModelOpt emits per-layer KV-cache scales that MLX has no parameter for.

    A real ``kv_cache_quant_algo: FP8`` export ships ``k_scale``/``v_scale`` on
    every full-attention layer. MLX quantizes its KV cache at runtime, so these
    must be dropped or ``load_weights(strict=True)`` rejects the checkpoint.
    """
    weights = _nvfp4_weights() | {
        f"self_attn.{projection}_proj.{projection}_scale": mx.array(
            scale, dtype=mx.float32
        )
        for projection, scale in [("k", 0.125), ("v", 0.25)]
    }

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
    )

    assert not any(
        key.endswith(".k_scale") or key.endswith(".v_scale") for key in transformed
    )
    assert transformed["layer.weight"].dtype == mx.uint32
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


def _hadamard_rotation(width, block):
    h = np.ones((1, 1), dtype=np.float32)
    while len(h) < block:
        h = np.block([[h, h], [h, -h]])
    return np.kron(np.eye(width // block, dtype=np.float32), h / np.sqrt(block))


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("block", [512, 1024])
def test_signed_hadamard_matches_dense_reference(dtype, block):
    rng = np.random.default_rng(7)
    width = 2 * block
    x = mx.array(rng.normal(size=(2, 3, width)), dtype=dtype)
    signs = mx.array(rng.choice([-1.0, 1.0], size=width), dtype=mx.float32)
    rotation = _hadamard_rotation(width, block)
    expected = (np.asarray(x).astype(np.float32) * np.asarray(signs)) @ rotation
    actual = prism_ops.hadamard_transform(x, block, signs)
    restored = prism_ops.hadamard_transform(actual, block, signs, inverse=True)
    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
    np.testing.assert_allclose(restored, x, atol=2e-3, rtol=1e-3)


@pytest.mark.parametrize("block", [0, 512])
def test_packed_projections_and_embedding_match_unrotated_dense_weights(block):
    mx.random.seed(3)
    width, rows = 1024, 8
    layer = prism_ops.HadamardQuantizedLinear(width, rows, block)
    layer.weight, layer.scales, layer.biases = mx.quantize(
        mx.random.normal((rows, width)), group_size=128, bits=2
    )
    if block:
        layer.signs = mx.where(mx.arange(width) % 3 == 0, -1.0, 1.0)
    dense = np.asarray(
        mx.dequantize(layer.weight, layer.scales, layer.biases, group_size=128, bits=2)
    )
    rotation = _hadamard_rotation(width, block) if block else np.eye(width)
    signs = np.asarray(layer.signs) if block else np.ones(width)
    unrotated = (dense @ rotation) * signs
    x = mx.random.normal((2, 3, width))
    np.testing.assert_allclose(
        layer(x), np.asarray(x) @ unrotated.T, atol=1e-4, rtol=1e-4
    )

    embedding = prism_ops.HadamardQuantizedEmbedding(width, rows, block)
    embedding.load_weights(tree_flatten(layer.parameters()))
    indices = mx.array([[0, 5], [2, 0]])
    expected = (dense.astype(np.float16).astype(np.float32) @ rotation) * signs
    actual = embedding(indices)
    assert actual.dtype == mx.float16
    np.testing.assert_allclose(
        actual, expected[np.asarray(indices)], atol=2e-3, rtol=1e-3
    )
    np.testing.assert_allclose(embedding.as_linear(x), layer(x), atol=1e-5)


@pytest.fixture
def packed_prism_checkpoint(tmp_path):
    text = qwen35.TextConfig(
        model_type="qwen3_5_text",
        hidden_size=512,
        intermediate_size=1024,
        linear_num_value_heads=4,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=128,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=1024,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [4, 6, 6],
        },
    )
    vision = qwen35.VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=512,
        num_heads=4,
        patch_size=16,
        num_position_embeddings=16,
    )
    source = qwen35.Model(
        qwen35.ModelConfig(
            text_config=text,
            vision_config=vision,
            model_type="qwen3_5",
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
        )
    )
    weights = dict(tree_flatten(source.parameters()))
    records = []
    for path, module in source.language_model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        if path.endswith(("in_proj_a", "in_proj_b")):
            continue
        key = "language_model." + path
        raw = weights.pop(key + ".weight")
        arrays = mx.quantize(raw, group_size=128, bits=2)
        for suffix, value in zip(("weight", "scales", "biases"), arrays):
            weights[key + "." + suffix] = value
        weights[key + ".signs"] = mx.where(mx.arange(raw.shape[1]) % 3 == 0, -1.0, 1.0)
        records.append(
            {
                "path": path,
                "block": 512,
                "embedding": isinstance(module, nn.Embedding),
                "dtype": "float16",
            }
        )
    config = asdict(source.config)
    config.update(
        model_type="prism_hadamard_qwen35",
        schema_version=2,
        modules=records,
        tensor_namespace="mlx-vlm-qwen3_5",
        base_model_type="qwen3_5",
        gdn_activation_layout="grouped",
        quantization={"bits": 2, "group_size": 128, "mode": "affine"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path, config, weights


def test_standard_loader_preserves_packed_weights_and_cached_decode(
    packed_prism_checkpoint,
):
    path, config, weights = packed_prism_checkpoint
    module, model_type = get_model_and_args(config)
    assert module is prism
    assert model_type == "prism_hadamard_qwen35"
    model = load_model(path)
    lm = model.language_model
    assert isinstance(lm.model.embed_tokens, prism_ops.HadamardQuantizedEmbedding)
    assert isinstance(lm.lm_head, prism_ops.HadamardQuantizedLinear)
    assert isinstance(model.vision_tower.blocks[0].attn.qkv, nn.Linear)
    assert isinstance(lm.layers[0].linear_attn.in_proj_a, nn.Linear)
    for key, value in tree_flatten(model.parameters()):
        assert mx.array_equal(value, weights[key]).item(), key

    tokens = mx.array([[1, 2, 3, 4]])
    full = lm(tokens).logits
    cache = lm.make_cache()
    prefill = lm(tokens[:, :3], cache=cache).logits
    decode = lm(tokens[:, 3:], cache=cache).logits
    mx.eval(full, prefill, decode)
    assert full.shape == (1, 4, 64)
    assert mx.all(mx.isfinite(full)).item()
    np.testing.assert_allclose(prefill, full[:, :3], atol=3e-3, rtol=3e-3)
    np.testing.assert_allclose(decode, full[:, 3:], atol=3e-3, rtol=3e-3)

    # The same entry point merges image features and supplies multimodal RoPE.
    image_tokens = mx.array([[1, 62, 60, 63, 2]])
    features = model.get_input_embeddings(
        image_tokens,
        mx.zeros((4, 3 * 2 * 16 * 16)),
        image_grid_thw=mx.array([[1, 2, 2]]),
    )
    result = lm(
        image_tokens,
        inputs_embeds=features.inputs_embeds,
        position_ids=features.position_ids,
    ).logits
    assert result.shape == (1, 5, 64)
    assert mx.all(mx.isfinite(result)).item()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 1),
        ("tensor_namespace", "gguf"),
        ("gdn_activation_layout", "ungrouped"),
        ("modules", []),
        ("quantization", {"bits": 4, "group_size": 128, "mode": "affine"}),
    ],
)
def test_rejects_incompatible_pack_config(packed_prism_checkpoint, field, value):
    _, config, _ = packed_prism_checkpoint
    config[field] = value
    with pytest.raises(ValueError):
        prism.ModelConfig.from_dict(config)


@pytest.mark.parametrize("mutation", ["duplicate", "kind", "dtype", "block", "path"])
def test_rejects_invalid_module_manifest(packed_prism_checkpoint, mutation):
    _, config, _ = packed_prism_checkpoint
    config = copy.deepcopy(config)
    if mutation == "duplicate":
        config["modules"].append(config["modules"][0])
    else:
        field, value = {
            "kind": ("embedding", not config["modules"][0]["embedding"]),
            "dtype": ("dtype", "bfloat16"),
            "block": ("block", 2048),
            "path": ("path", "missing"),
        }[mutation]
        config["modules"][0][field] = value
    with pytest.raises(ValueError):
        prism.Model(prism.ModelConfig.from_dict(config))


@pytest.mark.parametrize("missing", [True, False])
def test_rejects_missing_or_invalid_signs(packed_prism_checkpoint, missing):
    _, config, weights = packed_prism_checkpoint
    key = "language_model.model.embed_tokens.signs"
    if missing:
        del weights[key]
    else:
        weights[key] = mx.zeros_like(weights[key])
    with pytest.raises(ValueError, match="sign"):
        prism.Model(prism.ModelConfig.from_dict(config)).sanitize(weights)
