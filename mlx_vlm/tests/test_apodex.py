from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model
from mlx_vlm.utils import get_model_and_args


def test_apodex_1_1_routes_to_qwen3_5_moe():
    module, model_type = get_model_and_args({"model_type": "qwen3_5_moe"})

    assert model_type == "qwen3_5_moe"
    assert module.Model is Model


def test_apodex_nvfp4_expert_sidecars_are_stacked():
    context = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False,
                num_hidden_layers=1,
                num_experts=2,
            )
        )
    )
    weights = {}
    prefix = "model.language_model.layers.0.mlp.experts"
    for expert in range(2):
        for projection in ("up_proj", "down_proj", "gate_proj"):
            weights[f"{prefix}.{expert}.{projection}.weight"] = mx.zeros(
                (8, 2), dtype=mx.uint32
            )
            weights[f"{prefix}.{expert}.{projection}.scales"] = mx.ones(
                (8, 2), dtype=mx.uint8
            )

    out = Model.sanitize(context, weights)

    prefix = "language_model.model.layers.0.mlp.switch_mlp"
    for projection in ("up_proj", "down_proj", "gate_proj"):
        assert out[f"{prefix}.{projection}.weight"].shape == (2, 8, 2)
        assert out[f"{prefix}.{projection}.scales"].shape == (2, 8, 2)
    assert not any(".experts." in key for key in out)


def test_apodex_sanitize_passes_through_native_mlx_weights():
    """Re-loading an already-converted MLX checkpoint must be a no-op.

    Community MLX conversions of Apodex ship fully stacked
    ``switch_mlp.{weight,scales,biases}`` keys with no ``.experts.`` names and
    no ``_scale_inv`` sidecars. Sanitize has to leave those alone rather than
    re-running a conversion over them.
    """
    context = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False, num_hidden_layers=1, num_experts=2
            )
        )
    )
    prefix = "language_model.model.layers.0.mlp.switch_mlp"
    weights = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{prefix}.{projection}.weight"] = mx.zeros((2, 8, 4), mx.uint32)
        weights[f"{prefix}.{projection}.scales"] = mx.ones((2, 8, 2), mx.bfloat16)
        weights[f"{prefix}.{projection}.biases"] = mx.zeros((2, 8, 2), mx.bfloat16)

    out = Model.sanitize(context, dict(weights))

    assert set(out) == set(weights)
    for key, value in weights.items():
        assert out[key].shape == value.shape
        assert out[key].dtype == value.dtype
    assert not any("_scale_inv" in key or ".experts." in key for key in out)
