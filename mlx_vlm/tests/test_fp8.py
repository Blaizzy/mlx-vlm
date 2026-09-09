from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.fp8 import (
    _dequantize_fp8_weight,
    _quantize_fp8_weight,
    make_quantization_config,
    transform_fp8_weights,
)
from mlx_vlm.models.qwen3_5.qwen3_5 import Model


def _source_fp8_pair(rows=130, cols=160):
    values = mx.random.uniform(low=-4, high=4, shape=(rows, cols))
    weight = mx.to_fp8(values)
    scales = mx.array([[0.00017, 0.00023], [0.00031, 0.00041]], dtype=mx.bfloat16)
    return weight, scales


def test_fp8_quantization_config_requires_128_block_e4m3():
    config = {
        "model_type": "future_model_with_the_same_checkpoint_format",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }
    assert make_quantization_config(config) == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }

    config["quantization_config"]["weight_block_size"] = [64, 64]
    assert make_quantization_config(config) is None


def test_fp8_reconstruction_requantizes_to_native_mxfp8():
    weight, scale_inv = _source_fp8_pair()
    restored = _dequantize_fp8_weight(weight, scale_inv)

    decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
    expanded_scales = mx.repeat(
        mx.repeat(scale_inv, 128, axis=0),
        128,
        axis=1,
    )[: weight.shape[0], : weight.shape[1]]
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


def test_fp8_weight_conversion_replaces_scale_inv_pair():
    weight, scale_inv = _source_fp8_pair(128, 128)
    out, quantization = transform_fp8_weights(
        {
            "proj.weight": weight,
            "proj.weight_scale_inv": scale_inv[:1, :1],
            "norm.weight": mx.ones((128,), dtype=mx.bfloat16),
        },
        {
            "model_type": "another_compatible_model",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        },
    )

    assert quantization == {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    assert "proj.weight_scale_inv" not in out
    assert "proj.scales" in out
    assert out["proj.weight"].dtype == mx.uint32
    assert out["proj.scales"].dtype == mx.uint8
    assert out["norm.weight"].dtype == mx.bfloat16


def test_fp8_weight_conversion_can_target_affine_4bit():
    weight, scale_inv = _source_fp8_pair(128, 128)
    target_quantization = {"group_size": 64, "bits": 4, "mode": "affine"}
    restored = _dequantize_fp8_weight(weight, scale_inv[:1, :1])
    expected_weight, expected_scales, expected_biases = mx.quantize(
        restored, **target_quantization
    )

    out, quantization = transform_fp8_weights(
        {
            "proj.weight": weight,
            "proj.weight_scale_inv": scale_inv[:1, :1],
        },
        {
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        },
        target_quantization=target_quantization,
    )
    mx.eval(
        out["proj.weight"],
        out["proj.scales"],
        out["proj.biases"],
        expected_weight,
        expected_scales,
        expected_biases,
    )

    assert quantization == target_quantization
    assert mx.array_equal(out["proj.weight"], expected_weight).item()
    assert mx.array_equal(out["proj.scales"], expected_scales).item()
    assert mx.array_equal(out["proj.biases"], expected_biases).item()
    assert "proj.weight_scale_inv" not in out


def test_shared_fp8_transform_runs_before_qwen_key_remapping():
    weight, scale_inv = _source_fp8_pair(128, 128)
    context = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=False))
    )
    transformed, _ = transform_fp8_weights(
        {
            "model.language_model.layers.0.mlp.down_proj.weight": weight,
            "model.language_model.layers.0.mlp.down_proj.weight_scale_inv": (
                scale_inv[:1, :1]
            ),
        },
        {
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        },
    )
    out = Model.sanitize(context, transformed)

    prefix = "language_model.model.layers.0.mlp.down_proj"
    assert f"{prefix}.weight" in out
    assert f"{prefix}.scales" in out
    assert not any(key.endswith("weight_scale_inv") for key in out)
