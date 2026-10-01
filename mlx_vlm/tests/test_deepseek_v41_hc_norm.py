"""Check rounding at the hyper-connection collapse/RMSNorm boundary."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.models.deepseek_v41.hc_norm import fused_hc_pre_norm
from mlx_vlm.models.deepseek_v41.language import DeepseekV41Block

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_exact_collapse_normalization(dtype, batch):
    mx.random.seed(139)
    norm = nn.RMSNorm(5120, eps=1e-6)
    norm.weight = mx.random.uniform(0.5, 1.5, (5120,)).astype(dtype)
    mix = mx.random.normal((batch, 1, 4))
    for scale in [0, 0.001, 1, 20]:
        h = (mx.random.normal((batch, 1, 4, 5120)) * scale).astype(dtype)
        actual = fused_hc_pre_norm(h, mix, norm.weight, norm.eps)
        expected = norm(DeepseekV41Block.hc_pre(h, mix))
        assert actual is not None
        assert mx.array_equal(actual, expected).item()


def test_strided_inputs_and_compiled_weight_updates():
    mx.random.seed(141)
    h = mx.random.normal((4, 4, 2, 5120)).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    mix = mx.random.normal((4, 4, 2)).transpose(0, 2, 1)
    norm = nn.RMSNorm(5120)
    norm.weight = norm.weight.astype(h.dtype)

    def fused(h, mix):
        return fused_hc_pre_norm(h, mix, norm.weight, norm.eps)

    fused = mx.compile(fused, inputs=norm)
    first = fused(h, mix)
    mx.eval(first)
    norm.weight = mx.random.uniform(0.5, 1.5, (5120,)).astype(h.dtype)
    actual = fused(h, mix)
    expected = norm(DeepseekV41Block.hc_pre(h, mix))
    assert mx.array_equal(actual, expected).item()
    assert not mx.array_equal(first, actual).item()


def test_unsupported_shapes_and_dtypes():
    for channels, dim, dtype, weight_dtype in [
        (4, 64, mx.bfloat16, mx.bfloat16),
        (2, 5120, mx.bfloat16, mx.bfloat16),
        (4, 5120, mx.float32, mx.float32),
        (4, 5120, mx.bfloat16, mx.float32),
    ]:
        h = mx.zeros((1, 1, channels, dim), dtype)
        mix = mx.ones((1, 1, channels), mx.float32)
        assert fused_hc_pre_norm(h, mix, mx.ones((dim,), weight_dtype), 1e-6) is None
