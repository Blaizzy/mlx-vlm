"""Masked affine matvecs preserve stock projection rounding on owned routes."""

import mlx.core as mx
import pytest

from mlx_vlm.models.deepseek_v41.language import DeepseekV41SwiGLU, DeepseekV41SwitchGLU
from mlx_vlm.models.deepseek_v41.masked_experts import (
    masked_affine4_linear,
    masked_experts,
    supports_masked_experts,
)
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("dims", [(512, 256), (768, 512), (5120, 2304), (2304, 5120)])
def test_masked_affine_matvec_matches_gather(dtype, dims):
    mx.random.seed(41)
    k, n = dims
    linear = QuantizedSwitchLinear(k, n, 3, bias=False)
    linear.scales = linear.scales.astype(dtype)
    linear.biases = linear.biases.astype(dtype)
    x = mx.random.normal((4, 1, k)).astype(dtype)
    indices = mx.random.randint(0, 3, (4, 1, 6)).astype(mx.int32)
    active = mx.array([True, False, True, False, False, True])
    active = mx.broadcast_to(active, indices.shape)
    expected = linear(mx.expand_dims(x, (-2, -3)), indices).squeeze(-2)
    expected = mx.where(active[..., None], expected, 0)
    actual = masked_affine4_linear(linear, x, indices, active, 6)
    assert mx.array_equal(actual, expected).item()

    # All inactive routes must initialize their output without reading an expert.
    actual = masked_affine4_linear(
        linear,
        x,
        mx.full(indices.shape, -1, mx.int32),
        mx.zeros(indices.shape, mx.bool_),
        6,
    )
    assert mx.array_equal(actual, mx.zeros(actual.shape, dtype)).item()


@pytest.mark.parametrize("owned", ["none", "mixed", "all"])
def test_masked_moe_preserves_weighting_before_activation_rounding(owned):
    mx.random.seed(73)
    switch = DeepseekV41SwitchGLU(512, 768, 3, activation=DeepseekV41SwiGLU(10.0))
    for name in ("up_proj", "gate_proj", "down_proj"):
        q = getattr(switch, name).to_quantized()
        q.scales = q.scales.astype(mx.bfloat16)
        q.biases = q.biases.astype(mx.bfloat16)
        setattr(switch, name, q)
    x = (mx.random.normal((4, 1, 512)) * 7).astype(mx.bfloat16)
    indices = mx.random.randint(0, 3, (4, 1, 6)).astype(mx.int32)
    active = mx.random.uniform(shape=indices.shape) > 0.5
    if owned != "mixed":
        active = mx.full(indices.shape, owned == "all", mx.bool_)
    scores = mx.where(active, mx.random.uniform(shape=indices.shape), 0)
    assert supports_masked_experts(switch, x, indices)
    expected = mx.where(active[..., None], switch(x, indices, scores), 0)
    actual = masked_experts(switch, x, indices, scores, active)
    assert mx.array_equal(actual, expected).item()
    assert not supports_masked_experts(switch, x.astype(mx.float32), indices)
    assert not supports_masked_experts(
        switch, mx.zeros((4, 2, 512), mx.bfloat16), indices
    )
    switch.up_proj.mode = "mxfp4"
    assert not supports_masked_experts(switch, x, indices)
