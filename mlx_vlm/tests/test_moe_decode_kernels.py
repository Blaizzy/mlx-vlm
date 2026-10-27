"""The fused single-token MoE decode path of the DeepSeek-V2 decoder
(Unlimited-OCR) is bit-identical to the gather_qmm graph, and only taken for
one bfloat16 token on Metal."""

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.deepseekocr.language as lang
from mlx_vlm.kernels import moe_softmax_topk, moe_swiglu_4bit
from mlx_vlm.models.deepseekocr.config import TextConfig

METAL = mx.metal.is_available() and mx.default_device() == mx.gpu
needs_metal = pytest.mark.skipif(not METAL, reason="needs Metal")

# D is not a multiple of 512 (the general qmv path, like the 1280-wide
# model), and D/8 words are not a multiple of 32 lanes.
D, I, I2, E, K, GS = 320, 192, 384, 16, 4, 64


def _config(**kwargs):
    return TextConfig(
        hidden_size=D,
        intermediate_size=I2,
        moe_intermediate_size=I,
        n_routed_experts=E,
        n_shared_experts=2,
        num_experts_per_tok=K,
        num_attention_heads=4,
        num_key_value_heads=4,
        **kwargs,
    )


def _bf16_scales(module):
    """Checkpoints carry bfloat16 scales; nn.quantize of f32 weights does not."""
    for linear in module.modules():
        if isinstance(linear, (nn.QuantizedLinear, lang.QuantizedSwitchLinear)):
            linear.scales = linear.scales.astype(mx.bfloat16)
            linear.biases = linear.biases.astype(mx.bfloat16)
    return module


def _moe(**kwargs):
    mx.random.seed(0)
    moe = lang.DeepseekV2MoE(_config(**kwargs))
    nn.quantize(moe.switch_mlp, group_size=GS, bits=4)
    nn.quantize(moe.shared_experts, group_size=GS, bits=4)
    moe.gate.weight = mx.random.normal((E, D)).astype(mx.bfloat16)
    return _bf16_scales(moe)


def _same_bits(a, b):
    return a.shape == b.shape and bool(
        mx.array_equal(a.view(mx.uint16), b.view(mx.uint16))
    )


def _unfused(monkeypatch):
    monkeypatch.setattr(lang, "_fused_decode_token", lambda x: False)


def _graph_moe(moe, x, residual):
    inds, scores = moe.gate(x)
    y = (moe.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)
    return residual + (y + moe.shared_experts(x)), inds, scores


@needs_metal
@pytest.mark.parametrize("route_scale", [1.0, 2.5, 1.7])
def test_softmax_topk_matches_graph(route_scale):
    mx.random.seed(1)
    for _ in range(8):
        logits = mx.random.normal((1, 1, E)).astype(mx.bfloat16)
        scores = mx.softmax(logits, axis=-1, precise=True)
        inds = mx.argpartition(scores, kth=-K, axis=-1)[..., -K:]
        selected = mx.take_along_axis(scores, inds, axis=-1) * route_scale
        got_inds, got_scores = moe_softmax_topk(
            logits, top_k=K, route_scale=route_scale
        )
        assert got_inds.dtype == mx.uint32
        assert mx.array_equal(got_inds, inds)
        assert _same_bits(got_scores, selected)


@needs_metal
def test_swiglu_experts_match_graph():
    moe = _moe()
    switch, shared = moe.switch_mlp, moe.shared_experts
    stacks = [
        (p.weight, p.scales, p.biases)
        for p in (switch.gate_proj, switch.up_proj, switch.down_proj)
    ]
    shared_stacks = [
        (p.weight, p.scales, p.biases)
        for p in (shared.gate_proj, shared.up_proj, shared.down_proj)
    ]
    mx.random.seed(2)
    for _ in range(8):
        x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        residual = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        inds = mx.sort(mx.random.randint(0, E, (1, 1, K)).astype(mx.uint32), axis=-1)
        scores = mx.random.uniform(shape=(1, 1, K)).astype(mx.bfloat16)
        routed = (switch(x, inds) * scores[..., None]).sum(axis=-2)
        # Routed experts only; with the shared MLP; with the residual too.
        got = moe_swiglu_4bit(x, inds, scores, *stacks, group_size=GS)
        assert _same_bits(got, routed)
        got = moe_swiglu_4bit(
            x, inds, scores, *stacks, group_size=GS, shared=shared_stacks
        )
        assert _same_bits(got, routed + shared(x))
        got = moe_swiglu_4bit(
            x,
            inds,
            scores,
            *stacks,
            group_size=GS,
            shared=shared_stacks,
            addends=[residual],
        )
        assert _same_bits(got, residual + (routed + shared(x)))
        # A dense MLP as one expert with weight one.
        one, weight = lang._one_expert(mx.bfloat16)
        got = moe_swiglu_4bit(
            x, one, weight, *shared_stacks, group_size=GS, addends=[residual]
        )
        assert _same_bits(got, residual + shared(x))


@needs_metal
def test_moe_layer_matches_graph(monkeypatch):
    moe = _moe()
    mx.random.seed(3)
    cases = []
    for _ in range(4):
        x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        residual = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
        cases.append((x, residual, *_graph_moe(moe, x, residual)))
    for x, residual, ref, inds, scores in cases:
        got_inds, got_scores = moe.gate(x)
        assert mx.array_equal(got_inds, inds)
        assert _same_bits(got_scores, scores)
        assert _same_bits(moe(x, residual=residual), ref)
    # The decoder layer folds the residual in through mlp_residual.
    layer = lang.DeepseekV2DecoderLayer(_config(), layer_idx=1)
    layer.mlp = moe
    h = cases[0][1]
    fused = layer.mlp_residual(h)
    _unfused(monkeypatch)
    assert _same_bits(fused, h + moe(layer.post_attention_layernorm(h)))


@needs_metal
def test_dense_layer_matches_graph(monkeypatch):
    layer = lang.DeepseekV2DecoderLayer(_config(first_k_dense_replace=1), 0)
    assert isinstance(layer.mlp, lang.DeepseekV2MLP)
    nn.quantize(layer.mlp, group_size=GS, bits=4)
    _bf16_scales(layer.mlp)
    h = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    fused = layer.mlp_residual(h)
    _unfused(monkeypatch)
    assert _same_bits(fused, layer.mlp_residual(h))


def _forbid(name):
    def kernel(*args, **kwargs):
        raise AssertionError(f"{name} called outside the fused decode path")

    return kernel


@pytest.fixture
def no_kernels(monkeypatch):
    monkeypatch.setattr(lang, "moe_softmax_topk", _forbid("moe_softmax_topk"))
    monkeypatch.setattr(lang, "moe_swiglu_4bit", _forbid("moe_swiglu_4bit"))


def test_gpu_without_metal_keeps_graph(monkeypatch, no_kernels):
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    moe = _moe()
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    out = moe(x, residual=x)
    mx.eval(out)
    assert out.shape == (1, 1, D)


def test_multi_token_and_other_dtypes_keep_graph(no_kernels):
    moe = _moe()
    for x in (
        mx.random.normal((1, 3, D)).astype(mx.bfloat16),
        mx.random.normal((2, 1, D)).astype(mx.bfloat16),
        mx.random.normal((1, 1, D)).astype(mx.float16),
    ):
        out = moe(x)
        mx.eval(out)
        assert out.shape == x.shape


def test_unquantized_or_8bit_experts_keep_graph(monkeypatch):
    # The router kernel does not depend on the expert format, so only the
    # expert kernel is forbidden here.
    monkeypatch.setattr(lang, "moe_swiglu_4bit", _forbid("moe_swiglu_4bit"))
    mx.random.seed(0)
    dense = lang.DeepseekV2MoE(_config())
    dense.gate.weight = mx.random.normal((E, D)).astype(mx.bfloat16)
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    mx.eval(dense(x))
    eight = lang.SwitchGLU(D, I, E)
    nn.quantize(eight, group_size=GS, bits=8)
    _bf16_scales(eight)
    assert eight.gate_proj.bits == 8
    assert lang.fused_swiglu_experts(x, *lang._one_expert(mx.bfloat16), eight) is None
