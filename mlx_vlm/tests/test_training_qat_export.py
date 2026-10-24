"""Packed QAT exports reload through the normal inference weight loader."""

import json
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_vlm import export_model
from mlx_vlm.tests.test_training_model_preparation import TinyModel
from mlx_vlm.trainer import export_qat_model, prepare_model_for_training
from mlx_vlm.trainer.common.qat import QATLinear
from mlx_vlm.utils import load_model_weights


@pytest.mark.parametrize("bits,group_size", [(4, 32), (4, 64), (8, 64)])
def test_export_packs_qat_layers_and_reloads(tmp_path, bits, group_size):
    source = tmp_path / "base"
    source.mkdir()
    # Stale original quantization must not survive the export.
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "tiny",
                "quantization": {"bits": 8, "group_size": 128},
                "quantization_config": {"bits": 8},
                "lora": {"fine_tune_type": "full"},
            }
        )
    )
    (source / "tokenizer.json").write_text('{"fixture": true}')
    (source / "model.safetensors.index.json").write_text('{"stale": true}')
    model = prepare_model_for_training(
        TinyModel(),
        train_type="full",
        qat=True,
        quantization_bits=bits,
        quantization_group_size=group_size,
        verbose=False,
    )
    # Export the trained values, rather than repacking the base model.
    model.projection.weight = model.projection.weight + 0.125
    ids = mx.array([[1, 2, 3]])
    expected = model(ids)
    before = dict(tree_flatten(model.parameters()))
    original_layer = model.projection
    destination = tmp_path / "packed"
    result = export_model(model, object(), destination, model_path=str(source))
    assert result == destination
    assert model.projection is original_layer
    assert isinstance(model.projection, QATLinear)
    for name, weight in tree_flatten(model.parameters()):
        assert weight is before[name]
    assert model.training
    assert mx.allclose(model(ids), expected).item()
    config = json.loads((destination / "config.json").read_text())
    assert config["quantization"]["projection"] == {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
    }
    assert config["quantization"] == config["quantization_config"]
    assert "embedding" not in config["quantization"]
    assert "lora" not in config
    assert (destination / "tokenizer.json").read_text() == '{"fixture": true}'
    weights = mx.load(str(destination / "model.safetensors"))
    assert weights["projection.weight"].dtype == mx.uint32
    assert weights["embedding.weight"].dtype != mx.uint32
    assert "projection.scales" in weights and "projection.biases" in weights
    restored = load_model_weights(TinyModel(), config, weights, destination)
    assert isinstance(restored.projection, nn.QuantizedLinear)
    assert isinstance(restored.embedding, nn.Embedding)
    assert mx.allclose(restored(ids), expected, atol=1e-5).item()
    with pytest.raises(FileExistsError):
        export_qat_model(model, object(), destination, model_path=str(source))


def test_failed_export_restores_float_training_layers(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    model = prepare_model_for_training(
        TinyModel(),
        train_type="full",
        qat=True,
        verbose=False,
    )
    original = model.projection
    with (
        patch("mlx_vlm.utils.save_weights", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        export_qat_model(model, object(), tmp_path / "packed", model_path=str(tmp_path))
    assert model.projection is original
    assert (
        dict(tree_flatten(model.trainable_parameters()))["projection.weight"]
        is original.weight
    )


def test_export_requires_qat(tmp_path):
    with pytest.raises(ValueError, match="qat=True"):
        export_qat_model(
            TinyModel(), object(), tmp_path / "packed", model_path="unused"
        )


@pytest.mark.parametrize("quantize", [None, True, False])
def test_export_float_model_or_force_quantization(tmp_path, quantize):
    (tmp_path / "config.json").write_text(
        '{"quantization": {"bits": 8, "group_size": 128}, "quantization_config": {"bits": 8}}'
    )
    model = TinyModel()
    before = model.projection
    output = export_model(
        model,
        object(),
        tmp_path / "export",
        model_path=str(tmp_path),
        quantize=quantize,
    )
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    assert model.projection is before
    if quantize:
        assert weights["embedding.weight"].dtype == mx.uint32
        assert weights["projection.weight"].dtype == mx.uint32
        assert config["quantization"]["bits"] == 4
        assert config["quantization"]["group_size"] == 64
    else:
        assert weights["projection.weight"].dtype != mx.uint32
        assert "quantization" not in config
        assert "quantization_config" not in config
    restored = load_model_weights(TinyModel(), config, weights, output)
    ids = mx.array([[1, 2, 3]])
    if not quantize:
        assert mx.allclose(restored(ids), model(ids)).item()
    assert mx.isfinite(restored(ids)).all().item()


@pytest.mark.parametrize("quantize,bits", [(False, None), (True, 8)])
def test_qat_export_can_keep_masters_or_override_quantization(tmp_path, quantize, bits):
    (tmp_path / "config.json").write_text("{}")
    model = prepare_model_for_training(
        TinyModel(),
        train_type="full",
        qat=True,
        verbose=False,
    )
    output = export_model(
        model,
        object(),
        tmp_path / "export",
        model_path=str(tmp_path),
        quantize=quantize,
        bits=bits,
    )
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    if quantize:
        assert config["quantization"]["projection"]["bits"] == 8
        assert weights["projection.weight"].dtype == mx.uint32
    else:
        assert "quantization" not in config
        assert mx.array_equal(
            weights["projection.weight"], model.projection.weight
        ).item()
    assert isinstance(model.projection, QATLinear)


@pytest.mark.parametrize("train_type", ["lora", "dora"])
def test_export_fuses_adapters_without_mutating_training_model(tmp_path, train_type):
    (tmp_path / "config.json").write_text("{}")
    model = prepare_model_for_training(
        TinyModel(), train_type=train_type, verbose=False
    )
    model.projection.lora_b = mx.full(model.projection.lora_b.shape, 0.05)
    model.eval()
    expected = model(mx.array([[1, 2, 3]]))
    original = model.projection
    output = export_model(
        model, object(), tmp_path / "export", model_path=str(tmp_path)
    )
    weights = mx.load(str(output / "model.safetensors"))
    assert not any("lora_" in name or ".linear." in name for name in weights)
    config = json.loads((output / "config.json").read_text())
    restored = load_model_weights(TinyModel(), config, weights, output)
    assert mx.allclose(expected, restored(mx.array([[1, 2, 3]])), atol=1e-5).item()
    assert model.projection is original


def test_export_preserves_existing_packed_layers(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    model = TinyModel()
    nn.quantize(model, bits=8, group_size=32)
    output = export_model(
        model, object(), tmp_path / "export", model_path=str(tmp_path)
    )
    weights = mx.load(str(output / "model.safetensors"))
    config = json.loads((output / "config.json").read_text())
    assert config["quantization"]["projection"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
    }
    restored = load_model_weights(TinyModel(), config, weights, output)
    ids = mx.array([[1, 2, 3]])
    assert mx.allclose(model(ids), restored(ids)).item()


@pytest.mark.parametrize("restore_adapters", [False, True])
def test_full_checkpoint_restores_qat_detection_automatically(
    tmp_path, restore_adapters
):
    from mlx_vlm.trainer.peft import load_adapters, save_adapter

    (tmp_path / "config.json").write_text("{}")
    model = prepare_model_for_training(
        TinyModel(),
        train_type="full",
        qat=True,
        quantization_bits=8,
        quantization_group_size=32,
        verbose=False,
    )
    checkpoint = tmp_path / "weights.safetensors"
    save_adapter(model, checkpoint)
    restored = prepare_model_for_training(
        TinyModel(),
        train_type="full",
        checkpoint_path=checkpoint,
        verbose=False,
    )
    assert isinstance(restored.projection, QATLinear)
    if restore_adapters:
        restored = load_adapters(TinyModel(), checkpoint)
    output = export_model(
        restored, object(), tmp_path / "export", model_path=str(tmp_path)
    )
    config = json.loads((output / "config.json").read_text())
    assert config["quantization"]["projection"] == {
        "bits": 8,
        "group_size": 32,
        "mode": "affine",
    }
