from __future__ import annotations

import pytest

from mlx_vlm.generate.edit_image import is_image_edit_model, model_capabilities
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