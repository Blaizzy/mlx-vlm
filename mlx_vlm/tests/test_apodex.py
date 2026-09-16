import json
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter


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


def test_apodex_mtp_splitter_falls_back_to_root_model_type(tmp_path):
    """Apodex names its text stack separately from its architecture.

    config.json carries ``text_config.model_type`` ``qwen3_5_moe_text`` under a
    root ``qwen3_5_moe``. Consulting only the text_config finds no registered
    splitter, so the bundled MTP head cannot be extracted at all.
    """
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"mtp.fc.weight": mx.zeros((4, 4))}
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "num_hidden_layers": 4,
                },
            }
        )
    )

    splitter = detect_mtp_splitter(tmp_path)

    assert splitter is not None
    assert splitter.output_model_type == "qwen3_5_mtp"
