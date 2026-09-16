"""Loading, conversion, and general model utilities."""

import base64
import json
import logging
import struct
import textwrap
from io import BytesIO
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_vlm.convert import _preserve_existing_deepseek_v4_quantization
from mlx_vlm.utils import (
    DEFAULT_VIDEO_SAMPLING,
    StoppingCriteria,
    VideoMetadata,
    VideoSampling,
    _drop_modules_without_weights,
    _load_safetensors,
    _transform_modelopt_nvfp4_weights,
    estimate_num_image_tokens,
    get_model_and_args,
    get_model_path,
    load,
    load_config,
    load_image,
    load_model,
    load_processor,
    load_video,
    prepare_inputs,
    process_image,
    resolve_video_sampling,
)

# General utilities


@pytest.mark.parametrize(
    ("quant_method", "quant_algo"),
    [("modelopt", "NVFP4"), ("modelopt", "W4A16_NVFP4"), ("modelopt_mixed", "NVFP4")],
)
def test_transform_modelopt_nvfp4_weights(quant_method, quant_algo):
    packed = mx.arange(32, dtype=mx.uint8).reshape(2, 16)
    weights = {
        "layer.weight": packed,
        "layer.weight_scale": mx.array([[56, 64], [72, 80]], dtype=mx.uint8),
        "layer.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        "layer.input_scale": mx.array(0.25, dtype=mx.float32),
        "layer.bias": mx.ones((2,)),
    }

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": quant_method, "quant_algo": quant_algo}
    )

    assert transformed["layer.weight"].dtype == mx.uint32
    assert transformed["layer.weight"].shape == (2, 4)
    assert transformed["layer.scales"].tolist() == [[48, 56], [64, 72]]
    assert mx.array_equal(transformed["layer.bias"], weights["layer.bias"])
    assert "layer.weight_scale" not in transformed
    assert "layer.weight_scale_2" not in transformed
    assert "layer.input_scale" not in transformed
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


def test_transform_modelopt_mixed_nvfp4_fp8_weights():
    weights = {
        "experts.weight": mx.arange(32, dtype=mx.uint8).reshape(2, 16),
        "experts.weight_scale": mx.array([[56, 64], [72, 80]], dtype=mx.uint8),
        "experts.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        "experts.input_scale": mx.array(0.25, dtype=mx.float32),
        "attention.weight": mx.array([[56, 64], [68, 72]], dtype=mx.uint8),
        "attention.weight_scale": mx.array([0.5, 0.25], dtype=mx.bfloat16),
        "attention.input_scale": mx.array(0.125, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
    )

    assert transformed["experts.weight"].dtype == mx.uint32
    assert transformed["experts.scales"].dtype == mx.uint8
    assert transformed["attention.weight"].dtype == mx.bfloat16
    assert transformed["attention.weight"].tolist() == [[0.5, 1.0], [0.75, 1.0]]
    assert not any("weight_scale" in key or "input_scale" in key for key in transformed)
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


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

        data = {"input_ids": [1, 2, 3], "attention_mask": [7, 8, 9]}

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
        json.dumps({"model_type": "demo", "eos_token_id": 1}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = load_config(tmp_path)

    assert config["generation_config"] == generation_config
    assert config["eos_token_id"] == [2, 3]
    assert config["do_sample"] is True
    assert config["temperature"] == 1.0
    assert config["top_p"] == 0.95
    assert config["top_k"] == 64
    assert "max_new_tokens" not in config


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
        module, config, group_size=64, bits=4, mode="affine"
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

    # A model-owned group override must be evaluated before the default-group
    # divisibility gate. Qwen4-Exp's PLE rows are 160-wide: divisible by 32,
    # but not by the converter's default 64.
    module = DummyModule((10, 160))
    config = {}

    def group32_predicate(_path: str, _module: nn.Module):
        return {"fallback_group_size": 32}

    _, updated_config = quantize_model(
        module,
        config,
        group_size=64,
        bits=4,
        mode="affine",
        quant_predicate=group32_predicate,
    )
    assert module.language_model.group_size == 32
    assert module.vision_model.group_size == 32
    assert updated_config["quantization"]["language_model"]["group_size"] == 32
    assert updated_config["quantization"]["language_model"]["bits"] == 4
    assert updated_config["quantization"]["language_model"]["mode"] == "affine"

    # A compatible requested group remains authoritative. In particular,
    # NVFP4 must retain its required group-16 layout rather than taking the
    # affine fallback used for 160-wide PLE rows.
    module = DummyModule((10, 160))
    config = {}
    _, updated_config = quantize_model(
        module,
        config,
        group_size=16,
        bits=4,
        mode="nvfp4",
        quant_predicate=group32_predicate,
    )
    assert module.language_model.group_size == 16
    assert module.language_model.mode == "nvfp4"
    assert updated_config["quantization"]["language_model"] == {
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
    }

    # Existing partial overrides retain their implicit affine mode when the
    # requested checkpoint format uses a mode with fixed group and bit sizes.
    def affine8_predicate(_path: str, _module: nn.Module):
        return {"group_size": 64, "bits": 8}

    for mode, requested_group_size in (("nvfp4", 16), ("mxfp4", 32)):
        module = DummyModule((10, 128))
        _, updated_config = quantize_model(
            module,
            {},
            group_size=requested_group_size,
            bits=4,
            mode=mode,
            quant_predicate=affine8_predicate,
        )
        for name in ("language_model", "vision_model"):
            quantized = getattr(module, name)
            assert quantized.group_size == 64
            assert quantized.bits == 8
            assert quantized.mode == "affine"
            assert updated_config["quantization"][name] == {"group_size": 64, "bits": 8}

    # Only the explicit fallback protocol may bypass the requested group's
    # divisibility check.
    module = DummyModule((10, 96))
    _, updated_config = quantize_model(
        module,
        {},
        group_size=64,
        bits=4,
        mode="affine",
        quant_predicate=lambda _path, _module: {"group_size": 32, "bits": 8},
    )
    assert not hasattr(module.language_model, "scales")
    assert not hasattr(module.vision_model, "scales")
    assert updated_config["quantization"] == {
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    }

    # Test mxfp4 quantization
    module = DummyModule((10, 64))
    config = {}
    _, updated_config = quantize_model(
        module, config, group_size=32, bits=4, mode="mxfp4"
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
            config, model=MagicMock(), q_group_size=64, q_bits=4, q_mode="affine"
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
        Processor(), prompts="test <image>", images=mx.zeros((3, 8, 8))
    )
    consumed = []

    def consume_attention_mask():
        consumed.append(inputs["attention_mask"].tolist())

    worker = Thread(target=consume_attention_mask)
    worker.start()
    worker.join(timeout=1)

    assert inputs["attention_mask"] is attention_mask
    assert consumed == [[[1, 1]]]


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
        tokenizer=SimpleNamespace(eos_token_ids=[2]), additional_eos_token_ids=[3]
    )

    class Detokenizer:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    with (
        patch("mlx_vlm.utils.AutoProcessor.from_pretrained", return_value=processor),
        patch("mlx_vlm.utils.load_tokenizer", return_value=Detokenizer),
    ):
        loaded = load_processor("unused-model-path")

    criteria = loaded.tokenizer.stopping_criteria
    assert criteria.eos_token_ids == [2, 3]
    criteria.reset([5])
    assert criteria.eos_token_ids == [5, 2, 3]


def test_load_passes_revision():
    model_mock = MagicMock()
    model_mock.config = MagicMock(eos_token_id=None)
    processor_mock = MagicMock()

    with (
        patch("mlx_vlm.utils.get_model_path") as mock_get_model_path,
        patch("mlx_vlm.utils.load_model", return_value=model_mock),
        patch("mlx_vlm.utils.load_processor", return_value=processor_mock),
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

    def test_keeps_module_declared_in_manifest_but_not_loaded(self):
        # Manifest declares vision weights but none loaded -> keep, strict fails (#1963).
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight", "vision_tower.weight", "vision_tower.bias"}

        _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is not None
        with pytest.raises(ValueError, match="Missing"):
            model.load_weights(list(weights.items()), strict=True)

    def test_drops_module_absent_from_manifest(self, caplog):
        # The manifest also omits vision -> an intentional text-only conversion.
        model = self.FakeModel()
        weights = {"language_model.weight": mx.zeros((2, 2))}
        declared = {"language_model.weight"}

        with caplog.at_level(logging.WARNING):
            _drop_modules_without_weights(model, weights, declared)

        assert model.vision_tower is None
        assert "vision_tower" in caplog.text


def test_load_safetensors_reinterprets_f8_e8m0_header(tmp_path):
    path = tmp_path / "model.safetensors"
    header = {"weight": {"dtype": "F8_E8M0", "shape": [1], "data_offsets": [0, 1]}}
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


def test_load_model_matches_deepseek_v4_quantization_aliases():
    from mlx_vlm.models import deepseek_v4

    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            return cls()

    class FakeDeepseekV4Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.language_model = nn.Module()
            self.language_model.model = nn.Module()
            self.language_model.model.layers = [nn.Module()]
            self.language_model.model.layers[0].ffn = nn.Module()
            self.language_model.model.layers[0].ffn.shared_experts = nn.Module()
            self.language_model.model.layers[0].ffn.shared_experts.gate_proj = (
                nn.Linear(64, 64, bias=False)
            )
            self.language_model.lm_head = nn.Linear(64, 64, bias=False)

        def load_weights(self, weights, strict=True):
            self.loaded_weights = weights
            self.loaded_strict = strict

        @staticmethod
        def quantization_path_aliases(path):
            return deepseek_v4.Model.quantization_path_aliases(path)

    fake_model_class = SimpleNamespace(
        ModelConfig=FakeConfig, Model=FakeDeepseekV4Model
    )
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {
        "group_size": 32,
        "bits": 4,
        "mode": "mxfp4",
        "layers.0.ffn.shared_experts.w1": mxfp8,
        "head": False,
    }

    with (
        patch(
            "mlx_vlm.utils.load_config",
            return_value={"model_type": "deepseek_v4", "quantization": quantization},
        ),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch("mlx_vlm.utils._load_safetensors", return_value={}),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(fake_model_class, "deepseek_v4"),
        ),
        patch("mlx_vlm.utils.nn.quantize") as quantize,
    ):
        load_model(Path("/tmp/model"), lazy=True)

    predicate = quantize.call_args.kwargs["class_predicate"]
    fake_model = FakeDeepseekV4Model(FakeConfig())
    shared_expert_spec = predicate(
        "language_model.model.layers.0.ffn.shared_experts.gate_proj",
        fake_model.language_model.model.layers[0].ffn.shared_experts.gate_proj,
    )
    head_spec = predicate("language_model.lm_head", fake_model.language_model.lm_head)

    assert shared_expert_spec == mxfp8
    assert head_spec == {}


def test_load_model_transforms_fine_grained_fp8_by_format():
    class FakeConfig:
        @classmethod
        def from_dict(cls, config):
            return cls()

    class FakeQwenModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.proj = nn.Linear(128, 128, bias=False)

        def load_weights(self, weights, strict=True):
            self.loaded_weights = weights
            self.loaded_strict = strict

    fake_model_class = SimpleNamespace(ModelConfig=FakeConfig, Model=FakeQwenModel)
    source_config = {
        "model_type": "future_compatible_model",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    }

    with (
        patch("mlx_vlm.utils.load_config", return_value=source_config),
        patch("mlx_vlm.utils.glob.glob", return_value=["/tmp/model/model.safetensors"]),
        patch(
            "mlx_vlm.utils._load_safetensors",
            return_value={
                "proj.weight": mx.zeros((128, 128), dtype=mx.uint8),
                "proj.weight_scale_inv": mx.ones((1, 1), dtype=mx.bfloat16),
            },
        ),
        patch(
            "mlx_vlm.utils.get_model_and_args",
            return_value=(fake_model_class, "future_compatible_model"),
        ),
        patch("mlx_vlm.utils.nn.quantize") as quantize,
    ):
        model = load_model(Path("/tmp/model"), lazy=True)

    quantize.assert_called_once()
    assert quantize.call_args.kwargs["group_size"] == 32
    assert quantize.call_args.kwargs["bits"] == 8
    assert quantize.call_args.kwargs["mode"] == "mxfp8"
    loaded = dict(model.loaded_weights)
    assert loaded["proj.weight"].dtype == mx.uint32
    assert loaded["proj.scales"].dtype == mx.uint8
    assert "proj.weight_scale_inv" not in loaded


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
            "multi_modal_projector.linear_1", model.multi_modal_projector.linear_1
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


class TestEstimateNumImageTokens:
    def _processor(self):
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

        return Qwen3VLImageProcessor()

    def _actual_tokens(self, processor, width, height, **kwargs):
        import numpy as np
        from PIL import Image

        img = Image.new("RGB", (width, height), color=(9, 30, 51))
        grid = processor([img], **kwargs)["image_grid_thw"][0]
        return int(np.prod(grid)) // processor.merge_size**2

    @pytest.mark.parametrize(
        "width,height", [(64, 64), (640, 480), (1000, 1400), (2500, 1200), (333, 517)]
    )
    def test_estimate_matches_actual_processing(self, width, height):
        processor = self._processor()
        estimate = estimate_num_image_tokens(processor, height, width)
        assert estimate == self._actual_tokens(processor, width, height)

    def test_estimate_matches_actual_with_resized_dimensions(self):
        processor = self._processor()
        estimate = estimate_num_image_tokens(
            processor, 1400, 1000, resized_height=448, resized_width=448
        )
        assert estimate == self._actual_tokens(
            processor, 1000, 1400, resized_height=448, resized_width=448
        )

    def test_unsupported_processor_raises(self):
        with pytest.raises(NotImplementedError, match="num_image_tokens"):
            estimate_num_image_tokens(SimpleNamespace(), 480, 640)


def test_modelopt_mixed_drops_fp8_kv_cache_scales():
    """ModelOpt emits per-layer KV-cache scales that MLX has no parameter for.

    A real ``kv_cache_quant_algo: FP8`` export ships ``k_scale``/``v_scale`` on
    every full-attention layer. MLX quantizes its KV cache at runtime, so these
    must be dropped or ``load_weights(strict=True)`` rejects the checkpoint.
    """
    weights = {
        "layer.weight": mx.arange(32, dtype=mx.uint8).reshape(2, 16),
        "layer.weight_scale": mx.array([[56, 64], [72, 80]], dtype=mx.uint8),
        "layer.weight_scale_2": mx.array(0.5, dtype=mx.float32),
        "self_attn.k_proj.k_scale": mx.array(0.125, dtype=mx.float32),
        "self_attn.v_proj.v_scale": mx.array(0.25, dtype=mx.float32),
    }

    transformed, quantization = _transform_modelopt_nvfp4_weights(
        weights, {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
    )

    assert not any(
        key.endswith(".k_scale") or key.endswith(".v_scale") for key in transformed
    )
    assert transformed["layer.weight"].dtype == mx.uint32
    assert quantization == {"group_size": 16, "bits": 4, "mode": "nvfp4"}


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """A deterministic 600-frame 64x64 clip at 30 fps, i.e. 20 seconds."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for i in range(600):
        writer.write(np.full((64, 64, 3), i % 256, np.uint8))
    writer.release()
    return str(path)


class _AttributeVideoProcessor:
    """A video processor naming its knobs the way load_video does."""

    fps = 1.0
    min_frames = 8
    max_frames = 100


class TestVideoSampling:
    def test_library_defaults_match_the_historical_load_video_signature(self):
        assert DEFAULT_VIDEO_SAMPLING == VideoSampling(
            fps=2.0, nframes=None, min_frames=4, max_frames=768, frame_factor=2
        )


class TestLoadVideo:
    def test_unknown_keyword_is_rejected(self, synthetic_video):
        with pytest.raises(TypeError, match="fpss"):
            load_video(synthetic_video, fpss=1.0)

    def test_timestamps_span_the_clip_at_the_source_frame_rate(self, synthetic_video):
        _, metadata = load_video(synthetic_video, fps=1.0)
        assert metadata.timestamps[0] == pytest.approx(0.0)
        assert metadata.timestamps[-1] == pytest.approx(20.0, abs=0.05)


class TestResolveVideoSampling:
    def test_processor_beats_defaults(self):
        processor = SimpleNamespace(video_processor=_AttributeVideoProcessor())
        resolved = resolve_video_sampling(processor, {})
        assert (resolved.fps, resolved.min_frames, resolved.max_frames) == (1.0, 8, 100)


class TestVideoMetadataForwarding:
    def test_metadata_is_only_forwarded_to_declaring_processors(self):
        class Processor:
            tokenizer = SimpleNamespace(pad_token="<pad>")

            def __call__(self, text, images=None, videos=None, fps=None, **kwargs):
                self.kwargs = kwargs
                self.fps = fps
                return {"input_ids": np.array([[1]]), "attention_mask": np.array([[1]])}

        processor = Processor()
        metadata = VideoMetadata(total_num_frames=30, fps=30, frames_indices=[0, 29])
        video = np.zeros((2, 3, 32, 32), dtype=np.uint8)
        with patch("mlx_vlm.utils.load_video", return_value=(video, metadata)):
            prepare_inputs(processor, videos=["clip.mp4"], prompts="Describe this.")
        assert "video_metadata" not in processor.kwargs
        assert processor.fps == [metadata.sampled_fps]


# Local Python model files

MODEL_PY = textwrap.dedent("""
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
    """)


def _write_checkpoint(path, config_extra=None):
    config = {"model_type": "does-not-exist-in-registry", "model_file": "model.py"}
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


def _load(path):
    return load_model(path)
