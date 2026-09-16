from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model


def test_apodex_nvfp4_expert_sidecars_are_stacked():
    context = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                tie_word_embeddings=False, num_hidden_layers=1, num_experts=2
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
