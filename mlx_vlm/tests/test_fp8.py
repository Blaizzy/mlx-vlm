import mlx.core as mx

from mlx_vlm.fp8 import _dequantize_fp8_weight, _quantize_fp8_weight


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
