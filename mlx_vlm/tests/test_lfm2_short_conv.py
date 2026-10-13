"""Cached LFM2 short-convolution kernel parity and fallback coverage."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.lfm2 import language


def make_layer(width, kernel_size, dtype, bias=False):
    mx.random.seed(73)
    layer = language.ShortConv(
        SimpleNamespace(hidden_size=width, conv_L_cache=kernel_size, conv_bias=bias),
        0,
    )
    if bias:
        for proj in (layer.in_proj, layer.out_proj, layer.conv):
            proj.bias = mx.random.normal(proj.bias.shape) * 0.1
    layer.update(tree_map(lambda x: x.astype(dtype), layer.parameters()))
    layer.eval()
    return layer


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch,width,kernel_size",
    [(1, 37, 3), (16, 64, 3), (3, 65, 2), (16, 128, 5), (2, 257, 8)],
)
@pytest.mark.parametrize("bias", [False, True])
def test_decode_matches_conv1d_across_state_updates(
    monkeypatch, dtype, batch, width, kernel_size, bias
):
    layer = make_layer(width, kernel_size, dtype, bias)
    prompt = mx.random.normal((batch, 7, width)).astype(dtype)
    cache = ArraysCache(1)
    mx.eval(layer(prompt, cache=cache), cache.state)
    # Prefill produces a strided tail view; the fused kernel must read its strides.
    reference_cache = ArraysCache(1)
    reference_cache[0] = mx.array(cache[0])
    inputs = mx.random.normal((batch, 9, width)).astype(dtype)
    kernel = language._short_conv_step_kernel
    calls = []

    def counted_kernel(**kwargs):
        calls.append(True)
        return kernel(**kwargs)

    for step in range(inputs.shape[1]):
        x = inputs[:, step : step + 1]
        monkeypatch.setattr(language, "_short_conv_step_kernel", counted_kernel)
        actual = layer(x, cache=cache)
        mx.eval(actual, cache.state)
        monkeypatch.setattr(language, "_short_conv_step_kernel", None)
        expected = layer(x, cache=reference_cache)
        mx.eval(expected, reference_cache.state)
        tolerance = {mx.float32: 2e-6, mx.float16: 2e-3, mx.bfloat16: 1e-2}[dtype]
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            atol=tolerance,
            rtol=tolerance,
        )
        assert mx.array_equal(cache[0], reference_cache[0]).item()
        assert cache[0].dtype == dtype
    assert len(calls) == inputs.shape[1]


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernel")
def test_noncontiguous_packed_inputs_and_weights():
    mx.random.seed(31)
    batch, width, kernel_size = 3, 37, 3
    dtype = mx.bfloat16
    packed = (
        mx.random.normal((batch, 3 * width, 4)).astype(dtype).transpose(0, 2, 1)[:, 1:2]
    )
    state = mx.random.normal((batch, width, 5)).astype(dtype).transpose(0, 2, 1)[:, 2:4]
    weight = mx.random.normal((kernel_size, width, 1)).astype(dtype).transpose(1, 0, 2)
    y, new_state = language._short_conv_step_kernel(
        inputs=[packed, state, weight, weight],
        template=[("T", dtype), ("D", width), ("K", kernel_size), ("HAS_BIAS", False)],
        grid=(width, batch, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, 1, width), state.shape],
        output_dtypes=[dtype, dtype],
    )
    b, c, x = mx.split(packed, 3, axis=-1)
    inputs = mx.concatenate([state, b * x], axis=1)
    expected = c * mx.conv1d(inputs, weight, groups=width)
    np.testing.assert_allclose(
        np.array(y.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=1e-2,
        rtol=1e-2,
    )
    assert mx.array_equal(new_state, inputs[:, -2:]).item()


@pytest.mark.parametrize(
    "case",
    [
        "training",
        "prefill",
        "no_cache",
        "empty_cache",
        "mask",
        "lengths",
        "mixed_dtype",
        "cpu",
        "large_kernel",
        "state_capture",
    ],
)
def test_unsupported_cases_use_standard_path(monkeypatch, case):
    def unexpected_kernel(**kwargs):
        raise AssertionError("The fused inference kernel must not run here")

    monkeypatch.setattr(language, "_short_conv_step_kernel", unexpected_kernel)
    kernel_size = 9 if case == "large_kernel" else 3
    layer = make_layer(16, kernel_size, mx.float32)
    if case == "training":
        layer.train()
    length = 3 if case == "prefill" else 1
    x = mx.random.normal((2, length, 16))
    cache = ArraysCache(1)
    cache[0] = mx.zeros((2, kernel_size - 1, 16))
    mask = None
    if case == "no_cache":
        cache = None
    elif case == "empty_cache":
        cache[0] = None
    elif case == "mask":
        mask = mx.array([[True], [False]])
    elif case == "lengths":
        cache.lengths = mx.array([0, 1])
    elif case == "mixed_dtype":
        cache[0] = cache[0].astype(mx.float16)
    gdn_sink = [] if case == "state_capture" else None
    with mx.stream(mx.cpu if case == "cpu" else mx.default_device()):
        out = layer(x, cache=cache, mask=mask, gdn_sink=gdn_sink)
        mx.eval(out)
    assert out.shape == x.shape
    assert mx.all(mx.isfinite(out)).item()
    if case == "lengths":
        assert cache.lengths.tolist() == [-1, 0]
    if case == "state_capture":
        assert len(gdn_sink) == 1
        layer_index, previous_state, _, _, conv_inputs = gdn_sink[0]
        assert layer_index == 0
        assert mx.all(previous_state == 0).item()
        assert mx.array_equal(cache[0][:, -1:], conv_inputs).item()


def test_training_gradients_remain_available():
    layer = make_layer(16, 3, mx.float32)
    layer.train()
    x = mx.random.normal((2, 5, 16))
    gradient = mx.grad(lambda v: mx.sum(layer(v)))(x)
    mx.eval(gradient)
    assert gradient.shape == x.shape
    assert mx.all(mx.isfinite(gradient)).item()
