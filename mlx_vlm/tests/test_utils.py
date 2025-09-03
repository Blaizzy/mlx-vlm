from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_vlm.utils import (
    StoppingCriteria,
    get_class_predicate,
    load,
    prepare_inputs,
    process_inputs_with_fallback,
    quantize_model,
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
    def __init__(self):
        self.tokenizer = type(
            "DummyTokenizer", (), {"pad_token": None, "eos_token": "[EOS]"}
        )()

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
        # Simulate PyTorch tensor output
        elif return_tensors == "pt":
            try:
                inputs = {k: MockTorch.tensor(v) for k, v in data.items()}
                inputs["pixel_values"] = MockTorch.tensor([4, 5, 6]) if images else []
                return inputs
            except ImportError:
                raise ImportError("PyTorch is not installed")
        else:
            raise ValueError(f"Unsupported return_tensors: {return_tensors}")


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


def test_get_class_predicate():
    class DummyModule:
        def __init__(self, shape):
            self.weight = mx.zeros(shape)
            self.to_quantized = True

    # Test skip_vision=True
    pred = get_class_predicate(skip_vision=True)
    module = DummyModule((10, 64))
    assert pred("language_model", module) is True
    assert pred("vision_model", module) is False

    # Test skip_vision=True with weights
    weights = {
        "language_model.scales": mx.array([1, 2, 3]),
        "vision_model.scales": mx.array([4, 5, 6]),
    }
    pred = get_class_predicate(skip_vision=True, weights=weights)
    assert pred("language_model", module) is True
    assert pred("vision_model", module) is False

    # Test skip_vision=False without weights
    pred = get_class_predicate(skip_vision=False)
    assert pred("", module) is True
    module = DummyModule((10, 63))  # Not divisible by 64
    assert pred("", module) is False

    # Test skip_vision=False with weights
    weights = {
        "language_model.scales": mx.array([1, 2, 3]),
        "vision_model.scales": mx.array([4, 5, 6, 7]),  # Not divisible by 64
    }
    pred = get_class_predicate(skip_vision=False, weights=weights)
    assert pred("language_model", DummyModule((10, 64))) is True
    assert pred("vision_model", DummyModule((10, 63))) is False


def test_quantize_module():
    class DummyModule(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.language_model = nn.Linear(shape[1], shape[1])
            self.vision_model = nn.Linear(shape[1], shape[1])

    # Test basic quantization
    module = DummyModule((10, 64))
    config = {}
    _, updated_config = quantize_model(
        module, config, q_group_size=64, q_bits=4, skip_vision=False
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
    assert updated_config["quantization"] == {"group_size": 64, "bits": 4}

    # Test skip_vision=True
    module = DummyModule((10, 64))
    config = {}
    _, updated_config = quantize_model(
        module, config, q_group_size=64, q_bits=4, skip_vision=True
    )

    # Vision module should not be quantized
    assert hasattr(module.language_model, "scales")
    assert not hasattr(module.vision_model, "scales")

    # Check config is updated correctly
    assert updated_config["vision_config"]["skip_vision"] is True


def test_prepare_inputs():
    """Test prepare_inputs function."""

    # Mock processor
    processor = MockProcessor()

    # Test text-only input
    inputs = prepare_inputs(
        processor, prompts="test", images=None, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))

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

    # Test image token without image present
    with pytest.raises(
        ValueError,
        match="Number of image tokens in prompt_token_ids.*does not match number of images",
    ):
        prepare_inputs(
            processor,
            images=None,
            prompts="test with <image> token",
            image_token_index=None,
        )


def test_process_inputs_with_fallback():

    processor = MockProcessor()

    # Test MLX tensor output
    inputs = process_inputs_with_fallback(
        processor, images=None, audio=None, prompts="test", return_tensors="mlx"
    )
    assert isinstance(inputs["input_ids"], mx.array)
    assert isinstance(inputs["attention_mask"], mx.array)

    try:
        # Test PyTorch tensor output with fallback
        inputs = process_inputs_with_fallback(
            processor, images=None, audio=None, prompts="test", return_tensors="pt"
        )
        # Check if the tensors have PyTorch-like attributes without importing torch
        assert hasattr(inputs["input_ids"], "numpy") and hasattr(
            inputs["input_ids"], "detach"
        )
        assert hasattr(inputs["attention_mask"], "numpy") and hasattr(
            inputs["attention_mask"], "detach"
        )
    except ImportError:
        # Test PyTorch not installed scenario
        with patch("builtins.__import__", side_effect=ImportError):
            with pytest.raises(
                ValueError,
                match="Failed to process inputs with error.*PyTorch is not installed.*Please install PyTorch",
            ):
                process_inputs_with_fallback(
                    processor,
                    images=None,
                    audio=None,
                    prompts="test",
                    return_tensors="pt",
                )


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


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with patch("mlx_vlm.utils.get_model_path") as mock_get_model_path, patch(
        "mlx_vlm.utils.load_model",
        return_value=model_mock,
    ) as mock_load_model, patch(
        "mlx_vlm.utils.load_processor",
        return_value=processor_mock,
    ) as mock_load_processor, patch(
        "mlx_vlm.utils.load_image_processor", return_value=None
    ):
        mock_get_model_path.return_value = Path("/tmp/model")

        model, processor = load("repo", revision="abc")

        assert model is model_mock
        assert processor is processor_mock
        mock_get_model_path.assert_called_with("repo", revision="abc")
