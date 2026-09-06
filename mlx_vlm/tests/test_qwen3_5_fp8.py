from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5.fp8 import (
    _dequantize_qwen_fp8_weight,
    convert_qwen_fp8_weights,
    make_quantization_config,
    quantize_qwen_fp8_weight,
)
from mlx_vlm.models.qwen3_5.qwen3_5 import Model
from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model as MoeModel


def _source_fp8_pair(rows=130, cols=160):
    values = mx.random.uniform(low=-4, high=4, shape=(rows, cols))
    weight = mx.to_fp8(values)
    scales = mx.array([[0.00017, 0.00023], [0.00031, 0.00041]], dtype=mx.bfloat16)
    return weight, scales


def test_qwen_fp8_quantization_config_requires_128_block_e4m3():
    config = {
        "model_type": "qwen3_5",
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


def test_qwen_fp8_quantization_config_accepts_moe():
    config = {
        "model_type": "qwen3_5_moe",
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


def test_qwen_fp8_reconstruction_requantizes_to_native_mxfp8():
    weight, scale_inv = _source_fp8_pair()
    restored = _dequantize_qwen_fp8_weight(weight, scale_inv)

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

    actual_weight, actual_scales = quantize_qwen_fp8_weight(weight, scale_inv)
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


def test_qwen_fp8_weight_conversion_replaces_scale_inv_pair():
    weight, scale_inv = _source_fp8_pair(128, 128)
    out = convert_qwen_fp8_weights(
        {
            "proj.weight": weight,
            "proj.weight_scale_inv": scale_inv[:1, :1],
            "norm.weight": mx.ones((128,), dtype=mx.bfloat16),
        }
    )

    assert "proj.weight_scale_inv" not in out
    assert "proj.scales" in out
    assert out["proj.weight"].dtype == mx.uint32
    assert out["proj.scales"].dtype == mx.uint8
    assert out["norm.weight"].dtype == mx.bfloat16


def test_qwen_fp8_weight_conversion_supports_batched_bare_experts():
    values = mx.random.uniform(low=-4, high=4, shape=(2, 256, 128))
    weight = mx.to_fp8(values)
    scale_inv = mx.ones((2, 2, 1), dtype=mx.bfloat16)

    out = convert_qwen_fp8_weights(
        {
            "experts.gate_up_proj": weight,
            "experts.gate_up_proj_scale_inv": scale_inv,
        }
    )

    assert out["experts.gate_up_proj"].shape == (2, 256, 32)
    assert out["experts.gate_up_proj_scales"].shape == (2, 256, 4)
    assert out["experts.gate_up_proj"].dtype == mx.uint32
    assert out["experts.gate_up_proj_scales"].dtype == mx.uint8
    assert not any(key.endswith("scale_inv") for key in out)


def test_qwen_model_sanitize_converts_fp8_before_key_remapping():
    weight, scale_inv = _source_fp8_pair(128, 128)
    context = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=False))
    )
    out = Model.sanitize(
        context,
        {
            "model.language_model.layers.0.mlp.down_proj.weight": weight,
            "model.language_model.layers.0.mlp.down_proj.weight_scale_inv": (
                scale_inv[:1, :1]
            ),
        },
    )

    prefix = "language_model.model.layers.0.mlp.down_proj"
    assert f"{prefix}.weight" in out
    assert f"{prefix}.scales" in out
    assert not any(key.endswith("weight_scale_inv") for key in out)


def test_qwen_moe_sanitize_converts_and_splits_fused_fp8_experts():
    gate_up = mx.to_fp8(mx.random.normal((2, 256, 128)))
    down = mx.to_fp8(mx.random.normal((2, 128, 128)))
    context = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=1,
                num_experts=2,
            )
        )
    )

    out = MoeModel.sanitize(
        context,
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.language_model.layers.0.mlp.experts.gate_up_proj_scale_inv": mx.ones(
                (2, 2, 1), dtype=mx.bfloat16
            ),
            "model.language_model.layers.0.mlp.experts.down_proj": down,
            "model.language_model.layers.0.mlp.experts.down_proj_scale_inv": mx.ones(
                (2, 1, 1), dtype=mx.bfloat16
            ),
        },
    )

    prefix = "language_model.model.layers.0.mlp.switch_mlp"
    assert out[f"{prefix}.gate_proj.weight"].shape == (2, 128, 32)
    assert out[f"{prefix}.up_proj.weight"].shape == (2, 128, 32)
    assert out[f"{prefix}.down_proj.weight"].shape == (2, 128, 32)
    assert out[f"{prefix}.gate_proj.scales"].shape == (2, 128, 4)
    assert out[f"{prefix}.up_proj.scales"].shape == (2, 128, 4)
    assert out[f"{prefix}.down_proj.scales"].shape == (2, 128, 4)
    assert not any("scale_inv" in key or ".experts." in key for key in out)
