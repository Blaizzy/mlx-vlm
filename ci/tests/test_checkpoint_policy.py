import json

import pytest

from ci.checkpoint_policy import (
    CheckpointPolicyError,
    validate_checkpoint,
    validate_snapshot,
)


def checkpoint():
    return {
        "repo": "mlx-community/example",
        "revision": "a" * 40,
        "expected_model_type": "example",
        "weight": {"bytes": 1024},
    }


def test_checkpoint_requires_full_immutable_revision():
    value = checkpoint()
    value["revision"] = "main"

    with pytest.raises(CheckpointPolicyError, match="full commit SHA"):
        validate_checkpoint(value)


def test_safe_snapshot_requires_safetensors_and_no_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "example"}))
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    (tmp_path / ".gitattributes").write_text("*.safetensors filter=lfs")
    assert validate_snapshot(tmp_path, checkpoint())["safetensors"] == 1

    (tmp_path / "config.json").write_text(json.dumps({"model_file": "model.py"}))
    with pytest.raises(CheckpointPolicyError, match="remote code"):
        validate_snapshot(tmp_path, checkpoint())


@pytest.mark.parametrize("name", ["weights.bin", "weights.pt", "payload.py"])
def test_snapshot_rejects_executable_serialization(tmp_path, name):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "example"}))
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    (tmp_path / name).write_bytes(b"unsafe")

    with pytest.raises(CheckpointPolicyError, match="unsafe checkpoint file"):
        validate_snapshot(tmp_path, checkpoint())


def test_snapshot_rejects_symlinked_entries(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "example"}))
    (tmp_path / "model.safetensors").write_bytes(b"safe")
    (tmp_path / "escape").symlink_to("/tmp")

    with pytest.raises(CheckpointPolicyError, match="symlink"):
        validate_snapshot(tmp_path, checkpoint())
