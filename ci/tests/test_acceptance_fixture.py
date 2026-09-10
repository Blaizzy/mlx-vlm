import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def test_model_path_acceptance_fixture_is_pinned_and_consistent():
    fixture = yaml.safe_load((ROOT / "ci/model-path-acceptance.yaml").read_text())
    profiles = yaml.safe_load((ROOT / "ci/model_path.yaml").read_text())
    model = profiles["models"][fixture["model"]]
    checkpoint = fixture["hf_checkpoint"]

    assert fixture["phases"] == ["synthetic", "hf_checkpoint"]
    assert model["synthetic"]["adapter"] == fixture["synthetic"]["adapter"]
    assert model["synthetic"]["profile"] == fixture["synthetic"]["profile"]
    assert model["hf_checkpoint"]["repo"] == checkpoint["repo"]
    assert model["hf_checkpoint"]["revision"] == checkpoint["revision"]
    assert model["hf_checkpoint"]["weight"]["bytes"] == checkpoint["weight_bytes"]
    assert fixture["input"]["max_tokens"] == 16
    assert fixture["thresholds"]["performance_percent"] == 5.0


def test_model_path_acceptance_asset_hash_is_frozen():
    fixture = yaml.safe_load((ROOT / "ci/model-path-acceptance.yaml").read_text())
    asset = ROOT / fixture["input"]["asset"]

    assert (
        hashlib.sha256(asset.read_bytes()).hexdigest()
        == fixture["input"]["asset_sha256"]
    )
