import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.utils import (
    get_class_predicate,
    prepare_inputs,
    quantize_model,
    sanitize_weights,
    update_module_configs,
)


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
    class MockProcessor:
        def __init__(self):
            self.tokenizer = type(
                "DummyTokenizer", (), {"pad_token": None, "eos_token": "[EOS]"}
            )()

        def __call__(self, text=None, images=None, padding=None, return_tensors=None):
            return {
                "input_ids": mx.array([1, 2, 3]),
                "pixel_values": mx.array([4, 5, 6]),
                "attention_mask": mx.array([7, 8, 9]),
            }

    processor = MockProcessor()

    # Test text-only input
    inputs = prepare_inputs(
        processor, prompts="test", images=None, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))

    # Test image-only input
    image = mx.zeros((3, 224, 224))
    inputs = prepare_inputs(
        processor, prompts=None, images=image, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))

    # Test both text and image
    image = mx.zeros((3, 224, 224))
    inputs = prepare_inputs(
        processor, prompts="test", images=image, image_token_index=None
    )
    assert "input_ids" in inputs
    assert mx.array_equal(inputs["input_ids"], mx.array([1, 2, 3]))
