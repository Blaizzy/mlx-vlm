"""Fused activation quantization must retain FP4 and scale rounding exactly."""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.deepseek_v41 import fakequant as fq

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


def compare(x, kind, block):
    fn = getattr(fq, "fake_quant_fp4_" + kind)
    actual = fn(x, block)
    with patch.object(fq, "_fp4_roundtrip_kernel", None):
        expected = fn(x, block)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("kind,block", [("ue8m0", 32), ("e4m3", 16)])
def test_random_finite_inputs(dtype, kind, block):
    mx.random.seed(197)
    for scale in [0, 1e-38, 1e-5, 0.2, 1, 20, 100, 3000]:
        x = (mx.random.normal((4, 512)) * scale).astype(dtype)
        compare(x, kind, block)
        compare(x.reshape(4, 16, 32).transpose(1, 0, 2), kind, block)
    compare(mx.zeros((block,), dtype), kind, block)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("kind,block", [("ue8m0", 32), ("e4m3", 16)])
def test_rounding_boundaries(dtype, kind, block):
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
    compare(mx.array(np.stack(rows)).astype(dtype), kind, block)


def test_fallback_and_disabled_quantization():
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 13
    for kind in ["ue8m0", "e4m3"]:
        compare(x, kind, 64)
        with patch.object(fq, "DISABLE", True):
            assert getattr(fq, "fake_quant_fp4_" + kind)(x) is x
    with pytest.raises(ValueError, match="not divisible"):
        fq.fake_quant_fp4_e4m3(x[:, :63])
