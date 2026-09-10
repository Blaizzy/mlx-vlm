from pathlib import Path

import pytest
import yaml

from ci.model_path_synthetic_probe import ADAPTERS, run

ROOT = Path(__file__).parents[2]


def test_stress_model_families_have_registered_synthetic_adapters():
    assert {
        "bert",
        "deepseek_vl_v2",
        "granite_vision",
        "internvl_chat",
        "qwen2_vl",
        "qwen2_5_vl",
    } <= ADAPTERS.keys()


@pytest.mark.parametrize("model", sorted(ADAPTERS))
def test_registered_synthetic_adapter_runs_a_finite_forward_pass(model):
    manifest = yaml.safe_load((ROOT / "ci/model_path.yaml").read_text())
    result = run(manifest["models"][model], manifest["synthetic_profiles"])

    assert result["finite"] is True
    assert result["output_shape"]
    assert result["parameter_signature"]
