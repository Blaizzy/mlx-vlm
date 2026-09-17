"""Absorbed MLA, ragged decode fallbacks, and vision attention fast paths."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.cache import KVCache
from mlx_vlm.models.mla import max_absorbed_queries
from mlx_vlm.models.paddleocr_vl.config import VisionConfig
from mlx_vlm.models.paddleocr_vl.vision import Attention, VisionModel
from mlx_vlm.models.qwen3_5 import language as lang

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
        self.assertEqual(max_absorbed_queries(64, 128, 128), 1)
        self.assertGreaterEqual(max_absorbed_queries(1, 1, 1), 1)


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
                self.assertTrue(
                    mx.allclose(materialized, absorbed, atol=1e-4, rtol=1e-4),
                    f"{name}: absorbed and materialized disagree through __call__",
                )


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
                self.assertEqual(
                    src.count("absorbed = L == 1 or L <= max_absorbed_queries("),
                    1,
                    f"{m}: expected exactly one gate decision",
                )
                self.assertEqual(
                    src.count("if absorbed:"), 2, f"{m}: expected both gates to use it"
                )


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
