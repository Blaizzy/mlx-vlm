from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_vlm.models.fast_ops as fast_ops
from mlx_vlm.models import quantized_ops as verifier_linear
from mlx_vlm.models.linear import native_batch_linear
from mlx_vlm.models.quantized_verifier import (
    decode_quantized_argmax,
    decode_quantized_linear,
    exact_quantized_moe_hc_expand,
    exact_quantized_selected_linear,
    exact_quantized_switch_linear,
)
from mlx_vlm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU


def _bf16_quantization_parameters(linear):
    linear.scales = linear.scales.astype(mx.bfloat16)
    if linear.biases is not None:
        linear.biases = linear.biases.astype(mx.bfloat16)
    return linear


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [
        *[
            ("affine", bits, group_size)
            for group_size in (32, 64, 128)
            for bits in (2, 3, 4, 5, 6, 8)
        ],
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ],
)
@pytest.mark.parametrize("batch", [1, 4, 8, 64, 127])
def test_general_quantized_moe_hc_matches_separate_kernels(
    mode, bits, group_size, batch
):
    mx.random.seed(600 + bits + batch)
    routed_linear = QuantizedSwitchLinear(
        512,
        16,
        4,
        False,
        group_size,
        bits,
        mode=mode,
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

    routed = exact_quantized_selected_linear(
        routed_linear,
        routed_inputs,
        indices,
    )
    collapsed = SwitchGLU._combine(routed, weights, shared)
    expected = fast_ops.exact_hc_expand(
        collapsed,
        residual,
        post,
        comb,
    )
    with patch(
        "mlx_vlm.models.quantized_verifier.exact_quantized_selected_linear",
        side_effect=AssertionError("supported formats must use the fused backend"),
    ):
        actual = exact_quantized_moe_hc_expand(
            routed_linear,
            routed_inputs,
            indices,
            weights,
            shared,
            residual,
            post,
            comb,
        )
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [
        *[("affine", bits, 64) for bits in (2, 3, 4, 5, 6, 8)],
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ],
)
@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 9, 16, 32, 64, 127])
def test_general_quantized_verifier_matches_decode(
    mode,
    bits,
    group_size,
    batch,
):
    mx.random.seed(100 + bits + batch)
    dense = nn.Linear(512, 16, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((batch, 3, 512)).astype(mx.bfloat16)
    if mode == "nvfp4":
        expected = verifier_linear._target_verify_singletons(linear, inputs)
    else:
        expected = mx.concatenate(
            [
                linear(mx.contiguous(inputs[:, position : position + 1]))
                for position in range(3)
            ],
            axis=1,
        )

    native_reference = mx.concatenate(
        [linear(mx.contiguous(inputs[:, i : i + 1])) for i in range(inputs.shape[1])],
        axis=1,
    )
    assert mx.array_equal(native_batch_linear(linear, inputs), native_reference).item()
    actual = decode_quantized_linear(linear, inputs)
    tokens = decode_quantized_argmax(linear, inputs)
    mx.eval(expected, actual, tokens)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(tokens, mx.argmax(expected, axis=-1)).item()


@pytest.mark.parametrize("input_dims", [64, 128])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_general_quantized_verifier_matches_narrow_qmv_quad(input_dims, bits):
    mx.random.seed(700 + input_dims + bits)
    dense = nn.Linear(input_dims, 8192, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=64,
        bits=bits,
        mode="affine",
    )
    inputs = mx.random.normal((8, 2, input_dims)).astype(mx.bfloat16)
    expected = mx.concatenate(
        [
            linear(mx.contiguous(inputs[:, position : position + 1]))
            for position in range(2)
        ],
        axis=1,
    )

    actual = decode_quantized_linear(linear, inputs)
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [1, 2, 5, 8, 9, 16, 32, 64])
@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [
        *[("affine", bits, 64) for bits in (2, 3, 4, 5, 6, 8)],
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ],
)
def test_general_quantized_switch_verifier_matches_decode(
    mode, bits, group_size, batch
):
    mx.random.seed(200 + bits)
    linear = QuantizedSwitchLinear(
        512,
        16,
        4,
        bias=False,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((batch, 3, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 3 * 2, dtype=mx.int32).reshape(batch, 3, 2) % 4
    expected = []
    for position in range(inputs.shape[1]):
        hidden = mx.expand_dims(mx.contiguous(inputs[:, position]), (-2, -3))
        projected = linear(hidden, indices[:, position], sorted_indices=False)
        expected.append(projected.squeeze(-2)[:, None])
    expected = mx.concatenate(expected, axis=1)

    actual = exact_quantized_switch_linear(linear, inputs, indices)
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch", [1, 2, 5, 8, 9, 16, 32, 64])
@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [
        *[("affine", bits, 64) for bits in (2, 3, 4, 5, 6, 8)],
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ],
)
def test_general_quantized_selected_verifier_matches_decode(
    mode,
    bits,
    group_size,
    batch,
):
    mx.random.seed(300 + bits)
    linear = QuantizedSwitchLinear(
        512,
        16,
        4,
        bias=False,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((batch, 3, 2, 512)).astype(mx.bfloat16)
    indices = mx.arange(batch * 3 * 2, dtype=mx.int32).reshape(batch, 3, 2) % 4
    expected = []
    for position in range(inputs.shape[1]):
        hidden = mx.expand_dims(mx.contiguous(inputs[:, position]), -2)
        projected = linear(hidden, indices[:, position], sorted_indices=False)
        expected.append(projected.squeeze(-2)[:, None])
    expected = mx.concatenate(expected, axis=1)

    actual = exact_quantized_selected_linear(linear, inputs, indices)
    mx.eval(expected, actual)

    assert actual is not None
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [
        *[("affine", bits, 64) for bits in (2, 3, 4, 5, 6, 8)],
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
    ],
)
def test_general_quantized_argmax_supports_packed_mask(mode, bits, group_size):
    mx.random.seed(400 + bits)
    dense = nn.Linear(512, 16, bias=False)
    dense.weight = dense.weight.astype(mx.bfloat16)
    linear = nn.QuantizedLinear.from_linear(
        dense,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((2, 3, 512)).astype(mx.bfloat16)
    allowed = mx.array([[1, 3, 5], [7, 9, 11]], dtype=mx.int32)
    token_mask = (mx.array(1, dtype=mx.int32) << allowed).reshape(-1, 1)

    actual = decode_quantized_argmax(
        linear,
        inputs,
        token_mask=token_mask,
    )
    mx.eval(actual)

    assert actual is not None
    assert mx.array_equal(actual, allowed).item()


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("shared_routes", [False, True])
def test_fused_fp8_gate_up_matches_native_selected_projections(
    dtype, batch, shared_routes
):
    from mlx_vlm.models.quantized_verifier import exact_quantized_switch_gate_up

    mx.random.seed(53)
    switch = SwitchGLU(512, 256, 6)
    switch.up_proj = switch.up_proj.to_quantized(32, 8, "mxfp8")
    switch.gate_proj = switch.gate_proj.to_quantized(32, 8, "mxfp8")
    # A strided input exercises the shared dispatch's layout normalization.
    x = mx.random.normal((batch, 4, 512)).astype(dtype)[:, ::2]
    routes = [[0, 1, 2], [2, 3, 0] if shared_routes else [3, 4, 5]]
    indices = mx.broadcast_to(mx.array(routes)[None], (batch, 2, 3))
    actual = exact_quantized_switch_gate_up(switch, x, indices)
    expected = tuple(
        exact_quantized_switch_linear(l, x, indices)
        for l in (switch.up_proj, switch.gate_proj)
    )
    mx.eval(actual, expected)
    assert actual is not None
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected))


def test_fp8_gate_up_preserves_fallbacks_for_other_shapes_and_formats():
    from mlx_vlm.models.quantized_verifier import exact_quantized_switch_gate_up

    switch = SwitchGLU(512, 256, 4)
    switch.up_proj = switch.up_proj.to_quantized(32, 8, "mxfp8")
    switch.gate_proj = switch.gate_proj.to_quantized(32, 8, "mxfp8")
    for length in (1, 3):
        x = mx.zeros((1, length, 512), mx.bfloat16)
        indices = mx.zeros((1, length, 1), mx.int32)
        assert exact_quantized_switch_gate_up(switch, x, indices) is None
    switch.gate_proj = QuantizedSwitchLinear(512, 256, 4, False, 32, 4, "mxfp4")
    assert (
        exact_quantized_switch_gate_up(
            switch, mx.zeros((1, 2, 512), mx.bfloat16), mx.zeros((1, 2, 1), mx.int32)
        )
        is None
    )


@pytest.mark.parametrize("embedding", [False, True])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_short_dense_projection_matches_singleton_decode(embedding, dtype):
    from mlx_vlm.models.linear import linear

    mx.random.seed(14)
    module = nn.Embedding(512, 1024) if embedding else nn.Linear(1024, 512, bias=False)
    module.weight = module.weight.astype(dtype)
    module.eval()
    operation = module.as_linear if embedding else module
    x = mx.random.normal((1, 3, 1024)).astype(dtype)
    expected = mx.concatenate([operation(x[:, i : i + 1]) for i in range(3)], axis=1)
    observed = linear(operation, x)
    mx.eval(expected, observed)
    assert mx.array_equal(observed, expected).item()
