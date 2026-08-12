from __future__ import annotations

import json

import pytest

from mlx_vlm.generate.capabilities import model_capabilities
from mlx_vlm.generate.edit_image import is_image_edit_model
from mlx_vlm.generate.image import is_image_generation_model

FLUX2_IDS = (
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9b-kv",
)


@pytest.mark.parametrize("model_id", FLUX2_IDS)
def test_model_capabilities_flux2_generation_and_editing(model_id: str) -> None:
    caps = model_capabilities(model_id)
    assert caps == ["image_generation", "image_editing"]
    assert is_image_generation_model(model_id)
    assert is_image_edit_model(model_id)


def test_model_capabilities_mage_flow_edit_only() -> None:
    # Edit variants generate nothing; edit capability only.
    assert model_capabilities("mage-flow-edit-base") == ["image_editing"]
    assert model_capabilities("microsoft/Mage-Flow-Edit-Turbo") == ["image_editing"]
    assert is_image_edit_model("mage-flow-edit-base")
    assert not is_image_generation_model("mage-flow-edit")


def test_model_capabilities_mage_flow_generation_only() -> None:
    assert model_capabilities("mage-flow-turbo") == ["image_generation"]
    assert is_image_generation_model("mage-flow-turbo")
    assert not is_image_edit_model("mage-flow-turbo")


@pytest.mark.parametrize(
    "model_id",
    ["unknown/not-a-model", "org/repo-with-lora", None],
)
def test_model_capabilities_unknown_models_report_empty(model_id: str | None) -> None:
    assert model_capabilities(model_id) == []


def test_model_capabilities_unknown_local_path() -> None:
    assert model_capabilities("/nonexistent/local/snapshot") == []


def _write_config(snapshot_path, config: dict) -> str:
    snapshot_path.mkdir(parents=True, exist_ok=True)
    (snapshot_path / "config.json").write_text(json.dumps(config))
    return str(snapshot_path)


def test_model_capabilities_text_only_from_config(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "qwen2", {"model_type": "qwen2"})
    assert model_capabilities("org/qwen2", snapshot_path=snapshot) == [
        "text_generation"
    ]


def test_model_capabilities_vlm_from_config(tmp_path) -> None:
    snapshot = _write_config(
        tmp_path / "llava",
        {"model_type": "llava", "vision_config": {"model_type": "clip_vit"}},
    )
    assert model_capabilities("org/llava", snapshot_path=snapshot) == [
        "text_generation",
        "vision",
    ]


def test_model_capabilities_audio_from_config(tmp_path) -> None:
    # Audio model types live in mlx_audio (no mlx_vlm.models module); the
    # audio_config alone is the capability signal.
    snapshot = _write_config(
        tmp_path / "bark", {"model_type": "bark", "audio_config": {}}
    )
    assert model_capabilities("org/bark", snapshot_path=snapshot) == ["audio"]


def test_model_capabilities_embeddings_from_config(tmp_path) -> None:
    snapshot = _write_config(tmp_path / "roberta", {"model_type": "xlm-roberta"})
    assert model_capabilities("org/roberta", snapshot_path=snapshot) == ["embeddings"]


def test_model_capabilities_image_flags_suppress_text(tmp_path) -> None:
    # A diffusion model config must not gain text_generation even though
    # get_model_and_args would resolve its type module.
    snapshot = _write_config(tmp_path / "flux2", {"model_type": "flux2"})
    caps = model_capabilities(
        "black-forest-labs/FLUX.2-klein-4B", snapshot_path=snapshot
    )
    assert caps == ["image_generation", "image_editing"]
