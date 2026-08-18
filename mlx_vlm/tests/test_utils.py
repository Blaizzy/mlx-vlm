import base64
import json
import logging
import struct
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.convert import _preserve_existing_deepseek_v4_quantization
from mlx_vlm.utils import (
    StoppingCriteria,
    _drop_modules_without_weights,
    _load_safetensors,
    apply_generation_config_defaults,
    get_model_and_args,
    get_model_path,
    load,
    load_config,
    load_image,
    load_model,
    load_processor,
    prepare_inputs,
    process_image,
    process_inputs,
    process_inputs_with_fallback,
    sanitize_weights,
    update_module_configs,
)


class MockTensor:
    def __init__(self, data):
        self.data = data

    def numpy(self):
        return self.data

    def detach(self):
        return self


class MockTorch:
    @staticmethod
    def tensor(data):
        return MockTensor(data)


class MockProcessor:
    def __init__(self, tokenizer_return_value=None):
        self.image_token = "<image>"
        _return_value = tokenizer_return_value

        class DummyTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "[EOS]"

            def __call__(
                self,
                text,
                add_special_tokens=False,
                padding=True,
                padding_side="left",
                return_tensors="mlx",
            ):
                del text, add_special_tokens, padding, padding_side
                if return_tensors != "mlx":
                    raise ValueError(f"Unsupported return_tensors: {return_tensors}")
                if _return_value is not None:
                    return _return_value
                return SimpleNamespace(
                    input_ids=mx.array([[1, 2, 3]]),
                    attention_mask=mx.array([[7, 8, 9]]),
                )

        self.tokenizer = DummyTokenizer()

    def __call__(
        self, text=None, images=None, audio=None, padding=None, return_tensors="mlx"
    ):
        # Count image tokens in text
        image_token_count = text.count("<image>") if text else 0

        # Handle None images case
        if images is None:
            if image_token_count > 0:
                raise ValueError(
                    f"Number of image tokens in prompt_token_ids ({image_token_count}) "
                    f"does not match number of images (0)"
                )
        else:
            # Convert single image to list
            if not isinstance(images, list):
                images = [images]

            images = [img for img in images if img is not None]

            if image_token_count != len(images):
                raise ValueError(
                    f"Number of image tokens in prompt_token_ids ({image_token_count}) "
                    f"does not match number of images ({len(images)})"
                )

        data = {
            "input_ids": [1, 2, 3],
            "attention_mask": [7, 8, 9],
        }

        # Simulate MLX tensor output
        if return_tensors == "mlx":
            inputs = {k: mx.array(v) for k, v in data.items()}
            inputs["pixel_values"] = mx.zeros((4, 5, 6)) if images else []
            return inputs
        else:
            raise ValueError(f"Unsupported return_tensors: {return_tensors}")


def test_load_config_applies_generation_config_sampling_defaults(tmp_path):
    generation_config = {
        "eos_token_id": [2, 3],
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_new_tokens": 4096,
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "demo", "eos_token_id": 1}),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config["generation_config"] == generation_config
    assert config["eos_token_id"] == [2, 3]
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 64
    assert "max_new_tokens" not in config


def test_apply_generation_config_defaults_preserves_model_config_signature():
    class ModelConfig:
        pass

    model_config = apply_generation_config_defaults(
        ModelConfig(),
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "do_sample": True,
            "max_new_tokens": 4096,
        },
    )

    assert model_config.temperature == 1.0
    assert model_config.top_p == 0.95
    assert model_config.top_k == 64
    assert model_config.do_sample is True
    assert not hasattr(model_config, "max_new_tokens")


def test_sanitize_weights():
    class DummyModel:
        def __init__(self, config=None):
            self.config = config

        def sanitize(self, weights):
            weights["sanitized"] = True
            return weights

    weights = {"test": mx.array([1, 2, 3])}
    # Need to instantiate DummyModel first since sanitize is an instance method
    model = DummyModel()
    sanitized = sanitize_weights(model, weights)
    assert sanitized["sanitized"] is True

    # Test with config
    config = {"test": "config"}
    sanitized = sanitize_weights(DummyModel, weights, config)
    assert sanitized["sanitized"] is True


def test_update_module_configs():
    class ModelConfig:
        def __init__(self):
            self.text_config = None
            self.vision_config = None

    class TextConfig:
        @classmethod
        def from_dict(cls, d):
            return "text_config"

    class VisionConfig:
        @classmethod
        def from_dict(cls, d):
            return "vision_config"

    # Define DummyModel after the other classes
    class DummyModel:
        pass

    # Set the classes as attributes after DummyModel is defined
    DummyModel.ModelConfig = ModelConfig
    DummyModel.TextConfig = TextConfig
    DummyModel.VisionConfig = VisionConfig

    config = {
        "text_config": {"test": "text"},
        "vision_config": {"test": "vision"},
    }
    model_config = ModelConfig()
    updated = update_module_configs(
        model_config, DummyModel, config, ["text", "vision"]
    )

    assert updated.text_config == "text_config"
    assert updated.vision_config == "vision_config"


def test_get_model_path_downloads_jsonl_tokenizers(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("mlx_vlm.utils.snapshot_download", fake_snapshot_download)

    assert get_model_path("org/model") == tmp_path
    assert "*.jsonl" in captured["allow_patterns"]


def test_quantize_module():
    from mlx_vlm.quant_utils import quantize_model

    class DummyModule(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.language_model = nn.Linear(shape[1], shape[1])
            self.vision_model = nn.Linear(shape[1], shape[1])

    # Test basic quantization
    module = DummyModule((10, 64))
    config = {}
    _, updated_config = quantize_model(
        module,
        config,
        group_size=64,
        bits=4,
        mode="affine",
    )

    # Check quantization parameters
    assert hasattr(module.language_model, "scales")
    assert hasattr(module.vision_model, "scales")
    assert module.language_model.scales.shape == (64, 1)
    assert module.language_model.bits == 4
    assert module.language_model.group_size == 64
    assert module.vision_model.scales.shape == (64, 1)
    assert module.vision_model.bits == 4
    assert module.vision_model.group_size == 64

    # Check config is updated correctly
    assert updated_config["quantization"] == {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }

    # Test mxfp4 quantization
    module = DummyModule((10, 64))
    config = {}
    _, updated_config = quantize_model(
        module,
        config,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    assert updated_config["quantization"] == {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
    }

    # Test skip_vision=True
    module = DummyModule((10, 64))
    config = {}

    def skip_vision_predicate(path: str, _module: nn.Module):
        return "vision_model" not in path

    _, updated_config = quantize_model(
        module,
        config,
        group_size=64,
        bits=4,
        mode="affine",
        quant_predicate=skip_vision_predicate,
    )

    # Vision module should not be quantized
    assert hasattr(module.language_model, "scales")
    assert not hasattr(module.vision_model, "scales")

    # Check config is updated correctly
    assert updated_config["quantization"] == {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }


def test_convert_preserves_existing_deepseek_v4_quantization():
    config = {
        "model_type": "deepseek_v4",
        "quantization_config": {"quant_method": "fp8"},
    }
    existing_quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.model.layers.0.attn.wkv": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
        },
    }

    with patch(
        "mlx_vlm.models.deepseek_v4.language.make_quantization_config",
        return_value=existing_quantization,
    ):
        _preserve_existing_deepseek_v4_quantization(
            config,
            model=MagicMock(),
            q_group_size=64,
            q_bits=4,
            q_mode="affine",
        )

    assert config["quantization"] is config["quantization_config"]
    assert config["quantization"]["group_size"] == 64
    assert config["quantization"]["bits"] == 4
    assert config["quantization"]["mode"] == "affine"
    assert config["quantization"]["language_model.model.layers.0.attn.wkv"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }


def test_prepare_inputs():
    """Test prepare_inputs function."""

    # Define tokenizer return values
    tok_result = MagicMock()
    tok_result.input_ids = [[1, 2, 3]]
    tok_result.attention_mask = [7, 8, 9]
    # Mock processor
    processor = MockProcessor(tokenizer_return_value=tok_result)

    # Test text-only input
    inputs = prepare_inputs(
        processor, prompts="test", images=None, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([[1, 2, 3]]))

    # Test image-only input with image token
    image = mx.zeros((3, 224, 224))
    inputs = prepare_inputs(
        processor, prompts="<image>", images=image, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))

    # Test both text and image
    image = mx.zeros((3, 224, 224))
    inputs = prepare_inputs(
        processor, prompts="test <image>", images=image, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))
    assert mx.array_equal(inputs["pixel_values"], mx.zeros((4, 5, 6)))
    assert mx.array_equal(inputs["attention_mask"], mx.array([7, 8, 9]))

    # Test image present without image token
    image = mx.zeros((3, 224, 224))
    with pytest.raises(
        ValueError,
        match="Number of image tokens in prompt_token_ids.*does not match number of images",
    ):
        prepare_inputs(
            processor,
            images=image,
            prompts="test without image token",
            image_token_index=None,
        )

    # Text-only calls go straight through the tokenizer, so bare image tokens
    # are not validated here unless actual image inputs are provided.
    inputs = prepare_inputs(
        processor,
        images=None,
        prompts="test with <image> token",
        image_token_index=None,
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([[1, 2, 3]]))


def test_prepare_inputs_preserves_mlx_attention_mask_for_thread_handoff():
    attention_mask = mx.array([[1, 1]], dtype=mx.int32)

    class Processor:
        tokenizer = SimpleNamespace(pad_token="[PAD]", eos_token="[EOS]")

        def __call__(self, text=None, images=None, padding=None, return_tensors="mlx"):
            return {
                "input_ids": mx.array([[1, 2]], dtype=mx.int32),
                "attention_mask": attention_mask,
                "pixel_values": mx.zeros((1, 2), dtype=mx.float32),
            }

    inputs = prepare_inputs(
        Processor(),
        prompts="test <image>",
        images=mx.zeros((3, 8, 8)),
    )
    consumed = []

    def consume_attention_mask():
        consumed.append(inputs["attention_mask"].tolist())

    worker = Thread(target=consume_attention_mask)
    worker.start()
    worker.join(timeout=1)

    assert inputs["attention_mask"] is attention_mask
    assert consumed == [[[1, 1]]]


def test_process_inputs_with_fallback():

    processor = MockProcessor()
    try:
        # Test MLX tensor output
        inputs = process_inputs_with_fallback(
            processor, images=None, audio=None, prompts="test", return_tensors="mlx"
        )
        assert isinstance(inputs["input_ids"], mx.array)
        assert isinstance(inputs["attention_mask"], mx.array)

    except ImportError:
        raise ImportError("MLX is not installed")


def test_stopping_criteria():
    class MockProcessor:
        def __init__(self):
            self.tokenizer = type(
                "DummyTokenizer", (), {"pad_token": None, "eos_token": "[EOS]"}
            )()

        def encode(self, text, add_special_tokens=False):
            # Mock encode method that returns a token ID (32008) for "[EOS]"
            if "[EOS]" in text:
                return [32008]
            return [1]  # Default token ID

    processor = MockProcessor()
    stopping_criteria = StoppingCriteria([2, 32000, 32007], processor)
    assert stopping_criteria.eos_token_ids == [2, 32000, 32007]

    stopping_criteria.add_eos_token_ids("[EOS]")
    assert stopping_criteria.eos_token_ids == [2, 32000, 32007, 32008]

    stopping_criteria.add_eos_token_ids("</answer>")
    assert stopping_criteria.eos_token_ids == [2, 32000, 32007, 32008, 1]


def test_stopping_criteria_reset():
    class MockProcessor:
        def __init__(self):
            self.tokenizer = type(
                "DummyTokenizer", (), {"pad_token": None, "eos_token": "[EOS]"}
            )()

        def encode(self, text, add_special_tokens=False):
            if "[EOS]" in text:
                return [32008]
            return [1]

    processor = MockProcessor()
    stopping_criteria = StoppingCriteria([2], processor)
    stopping_criteria.add_eos_token_ids("[EOS]")

    stopping_criteria.reset([5, 7])
    assert stopping_criteria.eos_token_ids == [5, 7]
    assert stopping_criteria(7) is True


def test_load_processor_preserves_additional_eos_tokens_on_reset():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_ids=[2]),
        additional_eos_token_ids=[3],
    )

    class Detokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    with (
        patch(
            "mlx_vlm.utils.AutoProcessor.from_pretrained",
            return_value=processor,
        ),
        patch("mlx_vlm.utils.load_tokenizer", return_value=Detokenizer),
    ):
        loaded = load_processor("unused-model-path")

    criteria = loaded.tokenizer.stopping_criteria
    assert criteria.eos_token_ids == [2, 3]
    criteria.reset([5])
    assert criteria.eos_token_ids == [5, 3]


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path") as mock_get_model_path,
        patch(
            "mlx_vlm.utils.load_model",
            return_value=model_mock,
        ),
        patch(
            "mlx_vlm.utils.load_processor",
            return_value=processor_mock,
        ),
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
    ):
        mock_get_model_path.return_value = Path("/tmp/model")

        model, processor = load("repo", revision="abc")

        assert model is model_mock
        assert processor is processor_mock
        mock_get_model_path.assert_called_with(
            "repo", revision="abc", force_download=False
        )


def test_get_model_and_args_rejects_unknown_text_configs():
    with pytest.raises(ValueError):
        get_model_and_args({"model_type": "unknown_text_arch"})


def test_get_model_and_args_remaps_mistral_to_llama():
    model_class, model_type = get_model_and_args({"model_type": "mistral"})

    assert model_class.__name__ == "mlx_vlm.models.llama"
    assert model_type == "llama"


@pytest.mark.parametrize(
    ("alias", "native_model_type"),
    [
        ("phi-msft", "phixtral"),
        ("falcon_mamba", "mamba"),
        ("joyai_llm_flash", "deepseek_v3"),
        ("kimi_k2", "deepseek_v3"),
        ("minimax_m2", "minimax"),
        ("iquestcoder", "llama"),
    ],
)
def test_get_model_and_args_remaps_text_model_aliases(alias, native_model_type):
    model_class, model_type = get_model_and_args({"model_type": alias})

    assert model_class.__name__ == f"mlx_vlm.models.{native_model_type}"
    assert model_type == native_model_type


def test_get_model_and_args_rejects_unknown_vision_configs():
    with pytest.raises(ValueError):
        get_model_and_args(
            {"model_type": "unknown-vlm", "vision_config": {"hidden_size": 16}},
        )


def test_load_model_forwards_strict_to_load_weights():
    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            return cls()

    class FakeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def load_weights(self, weights, strict=True):
            self.loaded_weights = weights
            self.loaded_strict = strict

    fake_model_class = SimpleNamespace(ModelConfig=FakeConfig, Model=FakeModel)
    weights = {"weight": mx.zeros((1,), dtype=mx.float16)}

    with (
        patch("mlx_vlm.utils.load_config", return_value={"model_type": "fake"}),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value=weights),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(fake_model_class, "fake"),
        ),
    ):
        model = load_model(Path("/tmp/model"), lazy=True, strict=False)

    assert model.loaded_weights == list(weights.items())
    assert model.loaded_strict is False


class TestDropModulesWithoutWeights:
    class ParameterlessHelper(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
            self.language_model = nn.Linear(2, 2, bias=False)
            self.vision_tower = nn.Linear(2, 2, bias=True)
            self.parameterless_helper = (
                TestDropModulesWithoutWeights.ParameterlessHelper()
            )

    def test_preserves_language_and_parameterless_modules(self, caplog):
        model = self.FakeModel()
        language_model = model.language_model
        parameterless_helper = model.parameterless_helper

        with caplog.at_level(logging.WARNING):
            _drop_modules_without_weights(model, {})

        assert model.language_model is language_model
        assert model.parameterless_helper is parameterless_helper
        assert model.vision_tower is None
        assert "vision_tower" in caplog.text
        assert "language_model" not in caplog.text
        assert "parameterless_helper" not in caplog.text

    def test_keeps_partially_weighted_module_for_strict_validation(self):
        model = self.FakeModel()
        weights = {
            "language_model.weight": mx.zeros((2, 2)),
            "vision_tower.bias": mx.zeros((2,)),
        }

        _drop_modules_without_weights(model, weights)

        assert model.vision_tower is not None
        with pytest.raises(ValueError, match="Missing"):
            model.load_weights(list(weights.items()), strict=True)

    def test_load_model_prunes_and_logs_text_only_modules(self, caplog):
        class FakeConfig:
            @classmethod
            def from_dict(cls, config):
                return cls()

        class FakeModel(self.FakeModel):
            def load_weights(self, weights, strict=True):
                self.loaded_weights = weights
                self.loaded_strict = strict

        fake_model_class = SimpleNamespace(ModelConfig=FakeConfig, Model=FakeModel)
        weights = {"language_model.weight": mx.zeros((2, 2))}

        with (
            patch(
                "mlx_vlm.utils.load_config",
                return_value={"model_type": "fake"},
            ),
            patch(
                "mlx_vlm.utils.glob.glob",
                return_value=["/tmp/model/model.safetensors"],
            ),
            patch("mlx_vlm.utils._load_safetensors", return_value=weights),
            patch(
                "mlx_vlm.utils.get_model_and_args",
                return_value=(fake_model_class, "fake"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            model = load_model(Path("/tmp/model"), lazy=True)

        assert model.language_model is not None
        assert model.vision_tower is None
        assert model.parameterless_helper is not None
        assert "vision_tower" in caplog.text
        assert model.loaded_strict is True


def test_load_safetensors_reinterprets_f8_e8m0_header(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {
        "weight": {
            "dtype": "F8_E8M0",
            "shape": [1],
            "data_offsets": [0, 1],
        }
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00")

    loaded = {"weight": mx.array([1], dtype=mx.uint8)}

    def fake_mx_load(file_path):
        current = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
        if current["weight"]["dtype"] == "F8_E8M0":
            raise RuntimeError("unsupported dtype F8_E8M0")
        assert current["weight"]["dtype"] == "U8"
        return loaded

    with patch("mlx_vlm.utils.mx.load", side_effect=fake_mx_load):
        assert _load_safetensors(str(path)) is loaded

    restored = json.loads(path.read_bytes()[8 : 8 + len(header_bytes)])
    assert restored["weight"]["dtype"] == "F8_E8M0"


def test_load_model_uses_deepseek_v4_fp8_quantization_config():
    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            return cls()

    class FakeDeepseekV4Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.language_model = nn.Linear(2, 2, bias=False)

        def load_weights(self, weights, strict=True):
            self.loaded_weights = weights
            self.loaded_strict = strict

    fake_model_class = SimpleNamespace(
        ModelConfig=FakeConfig, Model=FakeDeepseekV4Model
    )
    quantization = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
        "language_model.weight": {"group_size": 64, "bits": 8, "mode": "affine"},
    }

    with (
        patch(
            "mlx_vlm.utils.load_config",
            return_value={
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            },
        ),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value={}),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(fake_model_class, "deepseek_v4"),
        ),
        patch(
            "mlx_vlm.models.deepseek_v4.language.make_quantization_config",
            return_value=quantization,
        ) as make_quantization_config,
        patch("mlx_vlm.utils.nn.quantize") as quantize,
    ):
        model = load_model(Path("/tmp/model"), lazy=True)

    make_quantization_config.assert_called_once_with(model)
    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 64
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "affine"


def test_load_model_quantizes_projector_with_scales_when_skip_vision():
    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            return cls()

    class FakeProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(64, 64, bias=False)

    class FakeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.vision_tower = nn.Linear(64, 64, bias=False)
            self.multi_modal_projector = FakeProjector()
            self.language_model = nn.Linear(64, 64, bias=False)

        def load_weights(self, weights, strict=True):
            self.loaded_weights = weights
            self.loaded_strict = strict

    fake_model_class = SimpleNamespace(ModelConfig=FakeConfig, Model=FakeModel)
    weights = {
        "language_model.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "language_model.scales": mx.zeros((64, 1), dtype=mx.float16),
        "multi_modal_projector.linear_1.weight": mx.zeros((64, 16), dtype=mx.uint32),
        "multi_modal_projector.linear_1.scales": mx.zeros((64, 1), dtype=mx.float16),
        "vision_tower.weight": mx.zeros((64, 64), dtype=mx.float16),
    }
    selected = {}

    def fake_quantize(model, *args, **kwargs):
        predicate = kwargs["class_predicate"]
        selected["language"] = predicate("language_model", model.language_model)
        selected["projector"] = predicate(
            "multi_modal_projector.linear_1",
            model.multi_modal_projector.linear_1,
        )
        selected["vision"] = predicate("vision_tower", model.vision_tower)

    with (
        patch(
            "mlx_vlm.utils.load_config",
            return_value={
                "model_type": "kimi_vl",
                "quantization": {"group_size": 64, "bits": 8},
                "vision_config": {"skip_vision": True},
            },
        ),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value=weights),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(fake_model_class, "kimi_vl"),
        ),
        patch("mlx_vlm.utils.nn.quantize", side_effect=fake_quantize),
    ):
        load_model(Path("/tmp/model"), lazy=True)

    assert selected == {"language": True, "projector": True, "vision": False}


def test_load_delegates_adapter_loading_to_trainer_entrypoint():
    model = MagicMock()
    adapted_model = MagicMock()
    processor = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path", return_value=Path("/tmp/model")),
        patch("mlx_vlm.utils.load_model", return_value=model),
        patch("mlx_vlm.utils.apply_lora_layers", return_value=adapted_model) as apply,
        patch("mlx_vlm.utils.load_image_processor", return_value=None),
        patch("mlx_vlm.utils.load_processor", return_value=processor),
    ):
        result_model, result_processor = load("model-id", adapter_path="adapter-dir")

    apply.assert_called_once_with(model, "adapter-dir")
    adapted_model.eval.assert_called_once()
    assert result_model is adapted_model
    assert result_processor is processor


def test_load_processor_propagates_auto_processor_errors():
    with patch("mlx_vlm.utils.AutoProcessor.from_pretrained", side_effect=ValueError):
        with pytest.raises(ValueError):
            load_processor(Path("/tmp/model"), eos_token_ids=2)


def _make_test_image_bytes():
    """Create a small valid PNG in memory."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (4, 4), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestLoadImage:
    def test_pil_image_input(self):
        from PIL import Image as PILImage

        source = PILImage.new("RGBA", (4, 4), color="red")
        img = load_image(source)
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_bytesio_input(self):
        buf = _make_test_image_bytes()
        img = load_image(buf)
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_path_input(self, tmp_path):
        filepath = tmp_path / "test.png"
        buf = _make_test_image_bytes()
        filepath.write_bytes(buf.read())

        img = load_image(filepath)
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_string_filepath_input(self, tmp_path):
        filepath = tmp_path / "test.png"
        buf = _make_test_image_bytes()
        filepath.write_bytes(buf.read())

        img = load_image(str(filepath))
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_data_uri_input(self):
        buf = _make_test_image_bytes()
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"

        img = load_image(data_uri)
        assert img.mode == "RGB"
        assert img.size == (4, 4)

    def test_data_uri_missing_comma_raises(self):
        with pytest.raises(ValueError, match="missing comma separator"):
            load_image("data:image/png;base64NOCOMMA")

    def test_http_url_input(self):
        buf = _make_test_image_bytes()
        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        with patch("mlx_vlm.utils.requests.get", return_value=mock_response):
            img = load_image("https://example.com/image.png")
            assert img.mode == "RGB"

    def test_invalid_url_raises(self):
        with patch(
            "mlx_vlm.utils.requests.get",
            side_effect=Exception("Connection error"),
        ):
            with pytest.raises(
                ValueError,
                match=r"Failed to load image from https://example\.com/nonexistent\.png",
            ):
                load_image("https://example.com/nonexistent.png")

    def test_nonexistent_file_raises(self):
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image("/nonexistent/path/image.png")

    def test_nonexistent_path_object_raises(self):
        with pytest.raises(ValueError, match="Failed to load image"):
            load_image(Path("/nonexistent/path/image.png"))


class TestProcessImage:
    def _image(self, width=640, height=480):
        from PIL import Image

        return Image.new("RGB", (width, height), color=(120, 40, 200))

    def test_resize_shape_applied_without_custom_processor(self):
        img = process_image(self._image(), (320, 320), None)
        assert max(img.size) <= 320

    def test_resize_shape_ignored_with_custom_processor_warns(self):
        from mlx_vlm.models.base import BaseImageProcessor

        class DummyProcessor(BaseImageProcessor):
            def preprocess(self, images):
                return images

        original = self._image()
        with pytest.warns(UserWarning, match="resize_shape.*DummyProcessor"):
            img = process_image(original, (320, 320), DummyProcessor())

        assert img.size == original.size

    def test_no_resize_shape_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            img = process_image(self._image(), None, None)
        assert img.size == (640, 480)


class TestProcessorKwargsForwarding:
    class RecordingProcessor:
        """Processor double that consumes extra kwargs via **kwargs, like the
        Qwen-family processors consume max_pixels/min_pixels."""

        def __init__(self):
            self.seen_kwargs = None
            self.tokenizer = SimpleNamespace(pad_token="[PAD]", eos_token="[EOS]")

        def __call__(
            self, text=None, images=None, padding=None, return_tensors="mlx", **kwargs
        ):
            self.seen_kwargs = kwargs
            return {
                "input_ids": mx.array([[1, 2, 3]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }

    def test_process_inputs_forwards_processor_kwargs(self):
        processor = self.RecordingProcessor()
        process_inputs(processor, prompts="hi", processor_kwargs={"max_pixels": 1234})
        assert processor.seen_kwargs.get("max_pixels") == 1234

    def test_unmatched_loose_kwargs_are_still_filtered(self):
        # Kwargs matching no named parameter are dropped by the signature
        # filter (generation kwargs like max_tokens share this code path);
        # the explicit processor_kwargs channel is the supported way through.
        processor = self.RecordingProcessor()
        process_inputs(processor, prompts="hi", max_pixels=1234)
        assert "max_pixels" not in processor.seen_kwargs

    def test_prepare_inputs_forwards_processor_kwargs(self):
        from PIL import Image

        processor = self.RecordingProcessor()
        image = Image.new("RGB", (64, 64))
        prepare_inputs(
            processor,
            prompts="hi",
            images=[image],
            processor_kwargs={"max_pixels": 1234},
        )
        assert processor.seen_kwargs.get("max_pixels") == 1234

    def test_qwen3_vl_per_call_max_pixels_shrinks_grid(self):
        from PIL import Image

        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        processor = Qwen3VLImageProcessor()
        img = Image.new("RGB", (1000, 1400), color=(3, 5, 7))
        default_grid = processor([img])["image_grid_thw"][0]
        capped_grid = processor([img], max_pixels=256 * 256)["image_grid_thw"][0]
        assert capped_grid.prod() < default_grid.prod()

    def test_legacy_image_processor_warns_on_processor_kwargs(self):
        import numpy as np
        from PIL import Image

        from mlx_vlm.models.base import BaseImageProcessor

        class LegacyImageProcessor(BaseImageProcessor):
            def preprocess(self, images):
                return [np.zeros((3, 8, 8), dtype=np.float32) for _ in images]

        class LegacyProcessor:
            def __init__(self):
                self.image_processor = LegacyImageProcessor()
                self.pad_token = "[PAD]"
                self.pad_token_id = 0
                self.eos_token = "[EOS]"

            def __call__(self, text):
                return SimpleNamespace(input_ids=[1, 2])

        with pytest.warns(UserWarning, match="processor_kwargs"):
            prepare_inputs(
                LegacyProcessor(),
                prompts="a <image> b",
                images=[Image.new("RGB", (8, 8))],
                image_token_index=99,
                processor_kwargs={"max_pixels": 1234},
            )
