import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_vlm.models.nemotron_h.config import ModelConfig
from mlx_vlm.models.nemotron_h.modelopt import ModelOptMXFP8Linear
from mlx_vlm.models.nemotron_h.nemotron_h import Model
from mlx_vlm.utils import (
    _modelopt_mlx_quantization_config,
    _transform_modelopt_weights,
    get_model_and_args,
)


def tiny_config(**kwargs):
    values = dict(
        model_type="nemotron_h",
        vocab_size=16,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        max_position_embeddings=32,
        num_attention_heads=1,
        num_key_value_heads=1,
        attention_bias=False,
        mamba_num_heads=1,
        mamba_head_dim=4,
        mamba_proj_bias=False,
        ssm_state_size=4,
        conv_kernel=2,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=False,
        layers_block_type=["moe"],
        moe_intermediate_size=3,
        moe_shared_expert_intermediate_size=None,
        moe_latent_size=None,
        n_group=1,
        n_routed_experts=2,
        n_shared_experts=None,
        topk_group=1,
        num_experts_per_tok=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    values.update(kwargs)
    return ModelConfig(**values)


def test_get_model_and_args_routes_nemotron_h_module():
    model_class, model_type = get_model_and_args({"model_type": "nemotron_h"})

    assert model_class.__name__ == "mlx_vlm.models.nemotron_h"
    assert model_type == "nemotron_h"


def test_nemotron_h_config_normalizes_ultra_fields():
    config = ModelConfig.from_dict(
        {
            "model_type": "nemotron_h",
            "layers_block_type": ["mamba", "moe", "attention"],
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
        }
    )
    args = config.to_model_args()

    assert config.num_hidden_layers == 3
    assert args.hybrid_override_pattern == ["M", "E", "*"]
    assert args.time_step_limit == (0.0, math.inf)


def test_nemotron_h_language_model_exposes_text_model_contract():
    model = Model(tiny_config())
    quantization_root = model.language_model._model

    assert quantization_root.language_model is model.language_model
    assert "language_model.backbone.embeddings" in {
        path
        for path, _ in tree_flatten(
            quantization_root.leaf_modules(), is_leaf=nn.Module.is_module
        )
    }


def test_modelopt_static_fp8_linears_quantize_as_mxfp8():
    model = Model(
        tiny_config(
            layers_block_type=["mamba"],
            num_hidden_layers=1,
            quantization_config={"group_size": 16, "bits": 4, "mode": "nvfp4"},
        )
    )
    mixer = model.language_model.backbone.layers[0].mixer

    assert isinstance(mixer.in_proj, ModelOptMXFP8Linear)
    assert isinstance(mixer.out_proj, ModelOptMXFP8Linear)

    quantized = ModelOptMXFP8Linear(32, 64, bias=False).to_quantized(
        group_size=16,
        bits=4,
        mode="nvfp4",
    )

    assert quantized.group_size == 32
    assert quantized.bits == 8
    assert quantized.mode == "mxfp8"


def test_modelopt_nvfp4_transform_remaps_to_mlx_scales():
    prefix = "backbone.layers.0.mixer.experts.0.up_proj"
    weights = {
        f"{prefix}.weight": mx.array([[1, 2, 3, 4]], dtype=mx.uint8),
        f"{prefix}.weight_scale": mx.array([[8]], dtype=mx.uint8),
        f"{prefix}.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        f"{prefix}.input_scale": mx.array(1.0, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_weights(
        weights,
        {"quant_method": "modelopt"},
    )

    assert quantization["quant_method"] == "modelopt"
    assert quantization["group_size"] == 16
    assert quantization["bits"] == 4
    assert quantization["mode"] == "nvfp4"
    assert transformed[f"{prefix}.weight"].dtype == mx.uint32
    assert transformed[f"{prefix}.weight"].shape == (1, 1)
    assert transformed[f"{prefix}.scales"].dtype == mx.uint8
    assert transformed[f"{prefix}.global_scale"].shape == ()
    assert f"{prefix}.weight_scale" not in transformed
    assert f"{prefix}.weight_scale_2" not in transformed
    assert f"{prefix}.input_scale" not in transformed


def test_modelopt_fp8_transform_requantizes_to_mxfp8():
    prefix = "backbone.layers.0.mixer.in_proj"
    weights = {
        f"{prefix}.weight": mx.to_fp8(mx.ones((2, 32), dtype=mx.float32)),
        f"{prefix}.weight_scale": mx.array(0.5, dtype=mx.float32),
        f"{prefix}.input_scale": mx.array(1.0, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_weights(
        weights,
        {
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": [prefix],
                }
            },
        },
    )

    assert quantization["group_size"] == 32
    assert quantization["bits"] == 8
    assert quantization["mode"] == "mxfp8"
    assert quantization[f"language_model.{prefix}"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert transformed[f"{prefix}.weight"].dtype == mx.uint32
    assert transformed[f"{prefix}.weight"].shape == (2, 8)
    assert transformed[f"{prefix}.scales"].dtype == mx.uint8
    assert transformed[f"{prefix}.scales"].shape == (2, 1)
    assert f"{prefix}.weight_scale" not in transformed
    assert f"{prefix}.input_scale" not in transformed


def test_modelopt_mixed_transform_uses_mxfp8_overrides():
    fp8_prefix = "backbone.layers.0.mixer.in_proj"
    nvfp4_prefix = "backbone.layers.1.mixer.experts.0.up_proj"
    weights = {
        f"{fp8_prefix}.weight": mx.to_fp8(mx.ones((2, 32), dtype=mx.float32)),
        f"{fp8_prefix}.weight_scale": mx.array(0.5, dtype=mx.float32),
        f"{nvfp4_prefix}.weight": mx.array([[1, 2, 3, 4]], dtype=mx.uint8),
        f"{nvfp4_prefix}.weight_scale": mx.array([[8]], dtype=mx.uint8),
        f"{nvfp4_prefix}.weight_scale_2": mx.array(0.5, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_weights(
        weights,
        {
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": [fp8_prefix],
                },
                "group_1": {
                    "weights": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float",
                    },
                    "targets": [nvfp4_prefix],
                },
            },
        },
    )

    assert quantization["group_size"] == 16
    assert quantization["bits"] == 4
    assert quantization["mode"] == "nvfp4"
    assert quantization[f"language_model.{fp8_prefix}"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert transformed[f"{fp8_prefix}.weight"].dtype == mx.uint32
    assert transformed[f"{fp8_prefix}.scales"].dtype == mx.uint8
    assert transformed[f"{nvfp4_prefix}.weight"].dtype == mx.uint32
    assert transformed[f"{nvfp4_prefix}.scales"].dtype == mx.uint8
    assert f"{nvfp4_prefix}.global_scale" in transformed


def test_modelopt_converted_config_uses_fp8_targets():
    fp8_prefix = "backbone.layers.0.mixer.in_proj"
    nvfp4_prefix = "backbone.layers.1.mixer.experts.0.up_proj"
    weights = {
        f"language_model.{fp8_prefix}.weight": mx.zeros((2, 8), dtype=mx.uint32),
        f"language_model.{fp8_prefix}.scales": mx.zeros((2, 1), dtype=mx.uint8),
    }

    quantization = _modelopt_mlx_quantization_config(
        {
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": {
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                    "targets": [fp8_prefix],
                },
                "group_1": {
                    "weights": {
                        "dynamic": False,
                        "group_size": 16,
                        "num_bits": 4,
                        "type": "float",
                    },
                    "targets": [nvfp4_prefix],
                },
            },
        },
        weights,
    )

    assert quantization["group_size"] == 16
    assert quantization["bits"] == 4
    assert quantization["mode"] == "nvfp4"
    assert quantization[f"language_model.{fp8_prefix}"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert f"language_model.{nvfp4_prefix}" not in quantization


def test_nemotron_h_sanitize_stacks_expert_weights_and_scales():
    model = Model(tiny_config())
    prefix = "backbone.layers.0.mixer"
    weights = {
        f"{prefix}.experts.0.up_proj.weight": mx.ones((3, 4)),
        f"{prefix}.experts.1.up_proj.weight": mx.ones((3, 4)) * 2,
        f"{prefix}.experts.0.up_proj.scales": mx.ones((3, 1), dtype=mx.uint8),
        f"{prefix}.experts.1.up_proj.scales": mx.ones((3, 1), dtype=mx.uint8) * 2,
        f"{prefix}.experts.0.up_proj.global_scale": mx.array(1.0),
        f"{prefix}.experts.1.up_proj.global_scale": mx.array(2.0),
        f"{prefix}.experts.0.down_proj.weight": mx.ones((4, 3)) * 3,
        f"{prefix}.experts.1.down_proj.weight": mx.ones((4, 3)) * 4,
        f"{prefix}.experts.0.down_proj.scales": mx.ones((4, 1), dtype=mx.uint8) * 3,
        f"{prefix}.experts.1.down_proj.scales": mx.ones((4, 1), dtype=mx.uint8) * 4,
        f"{prefix}.experts.0.down_proj.global_scale": mx.array(3.0),
        f"{prefix}.experts.1.down_proj.global_scale": mx.array(4.0),
        "mtp.layers.0.norm.weight": mx.ones((4,)),
        "backbone.layers.0.mixer.gate.weight": mx.ones((2, 4)),
    }

    sanitized = model.sanitize(weights)

    out_prefix = "language_model.backbone.layers.0.mixer.switch_mlp"
    assert sanitized[f"{out_prefix}.fc1.weight"].shape == (2, 3, 4)
    assert sanitized[f"{out_prefix}.fc1.scales"].shape == (2, 3, 1)
    assert sanitized[f"{out_prefix}.fc1.global_scale"].shape == (2,)
    assert sanitized[f"{out_prefix}.fc2.weight"].shape == (2, 4, 3)
    assert sanitized[f"{out_prefix}.fc2.scales"].shape == (2, 4, 1)
    assert sanitized[f"{out_prefix}.fc2.global_scale"].shape == (2,)
    assert "mtp.layers.0.norm.weight" not in sanitized
    assert "language_model.backbone.layers.0.mixer.gate.weight" in sanitized
