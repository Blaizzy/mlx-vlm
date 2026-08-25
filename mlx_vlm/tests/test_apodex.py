import json
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm.models.qwen3_5_moe.qwen3_5_moe import Model
from mlx_vlm.speculative.drafters.mtp_split import detect_mtp_splitter
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel
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


def test_apodex_mtp_separate_experts_are_extracted():
    weights = {"mtp.norm.weight": mx.ones((8,))}
    prefix = "mtp.layers.0.mlp.experts"
    for expert in range(2):
        for projection in ("up_proj", "down_proj", "gate_proj"):
            weights[f"{prefix}.{expert}.{projection}.weight"] = mx.full(
                (8, 8), expert + 1.0
            )

    out = Qwen3_5MTPDraftModel.sanitize(None, weights)

    prefix = "layers.0.mlp.switch_mlp"
    for projection in ("up_proj", "down_proj", "gate_proj"):
        assert out[f"{prefix}.{projection}.weight"].shape == (2, 8, 8)
        assert out[f"{prefix}.{projection}.weight"][:, 0, 0].tolist() == [1.0, 2.0]
    assert out["norm.weight"].tolist() == [2.0] * 8
    assert not any(".experts." in key for key in out)


def test_apodex_fp8_mtp_expert_sidecars_are_converted_and_stacked():
    weights = {}
    prefix = "mtp.layers.0.mlp.experts"
    for expert in range(2):
        for projection in ("up_proj", "down_proj", "gate_proj"):
            weights[f"{prefix}.{expert}.{projection}.weight"] = mx.to_fp8(
                mx.random.normal((128, 128))
            )
            weights[f"{prefix}.{expert}.{projection}.weight_scale_inv"] = mx.ones(
                (1, 1), dtype=mx.bfloat16
            )

    out = Qwen3_5MTPDraftModel.sanitize(None, weights)

    prefix = "layers.0.mlp.switch_mlp"
    for projection in ("up_proj", "down_proj", "gate_proj"):
        assert out[f"{prefix}.{projection}.weight"].shape == (2, 128, 32)
        assert out[f"{prefix}.{projection}.scales"].shape == (2, 128, 4)
    assert not any("scale_inv" in key or ".experts." in key for key in out)


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


def test_apodex_mtp_splitter_still_prefers_text_config_model_type(tmp_path):
    """A registered text_config type must keep winning over the root type."""
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), {"mtp.fc.weight": mx.zeros((4, 4))}
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "not_a_registered_type",
                "text_config": {
                    "model_type": "qwen3_5",
                    "mtp_num_hidden_layers": 1,
                    "num_hidden_layers": 4,
                },
            }
        )
    )

    assert detect_mtp_splitter(tmp_path) is not None
