"""Tests for `model_file` checkpoint-supplied model classes.

A checkpoint's config.json may declare `model_file: <name>.py`; load_model
then imports the model class from that file inside the checkpoint instead
of the built-in registry (the same mechanism mlx_lm supports). These tests
build a minimal synthetic checkpoint on disk and load it through the real
load_model path.
"""

import json
import textwrap

import mlx.core as mx
import pytest

from mlx_vlm.utils import load_model

MODEL_PY = textwrap.dedent(
    '''
    import mlx.core as mx
    import mlx.nn as nn


    class ModelConfig:
        # Deliberately minimal: exposing text_config/vision_config attributes
        # opts in to update_module_configs, which then requires TextConfig /
        # VisionConfig classes in this module. A model_file module controls
        # both sides of that contract.
        def __init__(self, model_type="custom"):
            self.model_type = model_type

        @classmethod
        def from_dict(cls, params):
            return cls(model_type=params.get("model_type", "custom"))


    class Model(nn.Module):
        loaded_via_model_file = True

        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(4, 4, bias=False)

        def __call__(self, x):
            return self.proj(x)
    '''
)


def _write_checkpoint(path, config_extra=None):
    config = {
        "model_type": "does-not-exist-in-registry",
        "model_file": "model.py",
    }
    config.update(config_extra or {})
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.py").write_text(MODEL_PY, encoding="utf-8")
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {"proj.weight": mx.zeros((4, 4))},
        metadata={"format": "mlx"},
    )


def test_load_model_uses_checkpoint_model_file(tmp_path):
    _write_checkpoint(tmp_path)

    model = _load(tmp_path)

    # the class must come from the checkpoint's model.py, not the registry
    # (the registry would have raised: model_type does not exist there)
    assert getattr(model, "loaded_via_model_file", False) is True
    assert model.proj.weight.shape == (4, 4)


def test_missing_model_file_raises_clearly(tmp_path):
    _write_checkpoint(tmp_path)
    (tmp_path / "model.py").unlink()

    with pytest.raises(FileNotFoundError, match="model_file"):
        _load(tmp_path)


def test_registry_path_untouched_without_model_file(tmp_path):
    """A config WITHOUT model_file must go through the registry exactly as
    before — here that means the unknown model_type raises ValueError."""
    _write_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    del config["model_file"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="not supported"):
        _load(tmp_path)


def _load(path):
    return load_model(path)
