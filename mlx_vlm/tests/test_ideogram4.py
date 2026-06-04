from __future__ import annotations

import importlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.ideogram4.config import (
    IDEOGRAM_4_FP8_REPO_ID,
    Ideogram4TransformerConfig,
    get_variant,
    validate_dimensions,
)
from mlx_vlm.models.ideogram4.download import validate_model_layout
from mlx_vlm.models.ideogram4.model import Ideogram4ImageGenerationModel
from mlx_vlm.models.ideogram4.pipeline import (
    Ideogram4ImagePipeline,
    Ideogram4RuntimeConfig,
)
from mlx_vlm.models.ideogram4.prompting import (
    IDEOGRAM4_CAPTION_SCHEMA,
    NormalizedPrompt,
    PromptExpansionCaptionError,
    PromptExpansionResult,
    format_caption,
    is_structured_caption,
    normalize_prompt,
    prepare_prompt,
)
from mlx_vlm.models.ideogram4.scheduler import get_preset, make_step_intervals
from mlx_vlm.models.ideogram4.transformer import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Transformer,
)
from mlx_vlm.models.ideogram4.weights import dequantize_fp8_weight_only

image_module = importlib.import_module("mlx_vlm.generate.image")
dispatch_module = importlib.import_module("mlx_vlm.generate.dispatch")
prompt_utils_module = importlib.import_module("mlx_vlm.prompt_utils")
prompting_module = importlib.import_module("mlx_vlm.models.ideogram4.prompting")
structured_module = importlib.import_module("mlx_vlm.structured")
utils_module = importlib.import_module("mlx_vlm.utils")

EXPANDED_CAPTION = format_caption(
    {
        "high_level_description": "A detailed photo of a red cube.",
        "style_description": {
            "aesthetics": "minimal product photography",
            "lighting": "soft studio lighting",
            "photo": "close-up product photo",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "A quiet studio with a matte grey backdrop.",
            "elements": [{"type": "obj", "desc": "A translucent red glass cube."}],
        },
    }
)


def _write_layout(root: Path) -> None:
    (root / "model_index.json").write_text('{"_class_name": "Ideogram4Pipeline"}')
    for relative in (
        "transformer/diffusion_pytorch_model.safetensors",
        "unconditional_transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "unconditional_transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def test_ideogram4_declares_image_generation_model_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ANN001
        assert repo_id == "ideogram-ai/ideogram-4-fp8"
        assert kwargs["allow_patterns"] == [
            "model_index.json",
            "config.json",
            "manifest.json",
            "**/config.json",
        ]
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert Ideogram4ImageGenerationModel.is_image_generation_model
    assert Ideogram4ImageGenerationModel.model_type == "ideogram4"
    assert (
        image_generation_model_class("ideogram-ai/ideogram-4-fp8")
        is Ideogram4ImageGenerationModel
    )
    assert (
        image_generation_model_class(tmp_path.as_posix())
        is Ideogram4ImageGenerationModel
    )
    assert is_image_generation_model(tmp_path.as_posix())


def test_ideogram4_variant_resolution_is_exact() -> None:
    assert get_variant(IDEOGRAM_4_FP8_REPO_ID).repo_id == IDEOGRAM_4_FP8_REPO_ID

    for shorthand in ("ideogram4", "ideogram-4", "ideogram-4-fp8"):
        with pytest.raises(ValueError):
            get_variant(shorthand)


def test_ideogram4_validate_model_layout_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text('{"_class_name": "Ideogram4Pipeline"}')

    with pytest.raises(FileNotFoundError, match="transformer"):
        validate_model_layout(tmp_path)


def test_ideogram4_caption_schema_matches_prompting_contract() -> None:
    properties = IDEOGRAM4_CAPTION_SCHEMA["properties"]
    composition = properties["compositional_deconstruction"]
    elements = composition["properties"]["elements"]["items"]["anyOf"]
    object_element, text_element = elements
    style_variants = properties["style_description"]["anyOf"]
    photo_style, art_style = style_variants

    assert IDEOGRAM4_CAPTION_SCHEMA["required"] == ["compositional_deconstruction"]
    assert composition["required"] == ["background", "elements"]
    assert object_element["required"] == ["type", "desc"]
    assert text_element["required"] == ["type", "text", "desc"]
    assert object_element["properties"]["bbox"]["minItems"] == 4
    assert object_element["properties"]["bbox"]["maxItems"] == 4
    assert "photo" in photo_style["properties"]
    assert "art_style" not in photo_style["properties"]
    assert "art_style" in art_style["properties"]
    assert "photo" not in art_style["properties"]


def test_ideogram4_plain_prompt_wraps_as_minimal_json_caption() -> None:
    prepared = normalize_prompt("A red cube on a marble plinth.", warn=False)
    caption = json.loads(prepared.text)

    assert prepared.was_wrapped
    assert prepared.is_json_caption
    assert prepared.is_structured_caption
    assert caption["high_level_description"] == "A red cube on a marble plinth."
    assert caption["compositional_deconstruction"]["elements"] == [
        {"type": "obj", "desc": "A red cube on a marble plinth."}
    ]
    assert is_structured_caption(prepared.text)


def test_ideogram4_json_caption_passes_through_unchanged() -> None:
    prepared = normalize_prompt(EXPANDED_CAPTION, warn=False)

    assert prepared.text == EXPANDED_CAPTION
    assert not prepared.was_wrapped
    assert prepared.is_json_caption
    assert prepared.is_structured_caption


def test_ideogram4_caption_warnings_cover_elements_and_bounding_boxes() -> None:
    prompt = format_caption(
        {
            "compositional_deconstruction": {
                "background": "A studio.",
                "elements": [
                    {
                        "type": "text",
                        "desc": "A title.",
                        "bbox": [900, 100, 100, 800],
                    }
                ],
            }
        }
    )

    with pytest.warns(UserWarning) as records:
        prepared = normalize_prompt(prompt)

    messages = [str(record.message) for record in records]
    assert not prepared.is_structured_caption
    assert any(".text" in message for message in messages)
    assert any("y_min < y_max" in message for message in messages)


def test_ideogram4_prompt_expansion_uses_structured_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = object()
    processor = SimpleNamespace(tokenizer=tokenizer)
    model = SimpleNamespace(config={})
    logits_processor = object()
    observed: dict[str, object] = {}

    def fake_load(model_id: str):
        assert model_id == "tiny-text-model"
        return model, processor

    def fake_apply_chat_template(received_processor, config, messages):
        assert received_processor is processor
        assert config == {}
        observed["messages"] = messages
        return "formatted prompt"

    def fake_build_json_schema_logits_processor(received_tokenizer, schema):
        assert received_tokenizer is tokenizer
        assert schema is IDEOGRAM4_CAPTION_SCHEMA
        return logits_processor

    def fake_generate(received_model, received_processor, prompt, **kwargs):
        assert received_model is model
        assert received_processor is processor
        assert prompt == "formatted prompt"
        assert kwargs["logits_processors"] == [logits_processor]
        return SimpleNamespace(text=EXPANDED_CAPTION)

    monkeypatch.setattr(utils_module, "load", fake_load)
    monkeypatch.setattr(
        prompt_utils_module, "apply_chat_template", fake_apply_chat_template
    )
    monkeypatch.setattr(
        structured_module,
        "build_json_schema_logits_processor",
        fake_build_json_schema_logits_processor,
    )
    monkeypatch.setattr(dispatch_module, "generate", fake_generate)

    result = prompting_module.generate_prompt_expansion_caption(
        "A red cube.",
        model="tiny-text-model",
        aspect_ratio="1:1",
    )

    messages = observed["messages"]
    assert isinstance(messages, list)
    assert "visible wording" in messages[0]["content"]
    assert "do not add an aspect_ratio field" in messages[1]["content"]
    assert result.text == EXPANDED_CAPTION
    assert result.model == "tiny-text-model"


def test_ideogram4_prompt_expansion_model_expands_plain_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_prompt_expansion(prompt: str, **kwargs):
        assert prompt == "A red cube."
        assert kwargs["model"] == "tiny-text-model"
        assert kwargs["aspect_ratio"] == "1:1"
        return PromptExpansionResult(
            text=EXPANDED_CAPTION,
            raw_text=EXPANDED_CAPTION,
            model=kwargs["model"],
        )

    monkeypatch.setattr(
        prompting_module,
        "generate_prompt_expansion_caption",
        fake_prompt_expansion,
    )

    prepared = prepare_prompt(
        "A red cube.",
        prompt_expansion_model="tiny-text-model",
        width=1024,
        height=1024,
        warn=False,
    )

    assert prepared.text == EXPANDED_CAPTION
    assert prepared.prompt_expansion_model == "tiny-text-model"
    assert prepared.prompt_expansion_used
    assert not prepared.was_wrapped


def test_ideogram4_json_caption_skips_prompt_expansion_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_prompt_expansion(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("prompt expansion model should not be called")

    monkeypatch.setattr(
        prompting_module,
        "generate_prompt_expansion_caption",
        unexpected_prompt_expansion,
    )

    prepared = prepare_prompt(
        EXPANDED_CAPTION,
        prompt_expansion_model="tiny-text-model",
        warn=False,
    )

    assert prepared.text == EXPANDED_CAPTION
    assert not prepared.prompt_expansion_used
    assert not prepared.was_wrapped


def test_ideogram4_invalid_prompt_expansion_falls_back_to_minimal_caption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_prompt_expansion(*args, **kwargs):  # noqa: ARG001
        raise PromptExpansionCaptionError("bad json")

    monkeypatch.setattr(
        prompting_module,
        "generate_prompt_expansion_caption",
        fake_prompt_expansion,
    )

    with pytest.warns(UserWarning, match="falling back"):
        prepared = prepare_prompt(
            "A red cube.",
            prompt_expansion_model="bad-model",
        )

    assert prepared.was_wrapped
    assert not prepared.prompt_expansion_used
    assert prepared.prompt_expansion_error == "bad json"


def test_ideogram4_prompt_expansion_runtime_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_prompt_expansion(*args, **kwargs):  # noqa: ARG001
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(
        prompting_module,
        "generate_prompt_expansion_caption",
        fake_prompt_expansion,
    )

    with pytest.raises(RuntimeError, match="model failed to load"):
        prepare_prompt(
            "A red cube.",
            prompt_expansion_model="bad-model",
        )


@pytest.mark.parametrize("width,height", [(255, 512), (512, 2050), (2048, 256)])
def test_ideogram4_validate_dimensions_rejects_bad_sizes(
    width: int, height: int
) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_ideogram4_default_sampler_preset() -> None:
    preset = get_preset(None)

    assert preset.num_steps == 20
    assert preset.guidance_schedule[:2] == (3.0, 3.0)
    assert preset.guidance_schedule[-1] == 7.0
    assert make_step_intervals(2) == (0.0, 0.5, 1.0)


def test_ideogram4_dequantizes_weight_only_fp8() -> None:
    scale = mx.array([0.5, 2.0], dtype=mx.float32)
    expected = mx.array([[1.0, -2.0], [0.5, 4.0]], dtype=mx.float32)
    raw = {
        "linear.weight": mx.to_fp8(expected / mx.expand_dims(scale, axis=-1)),
        "linear.weight_scale": scale,
        "linear.bias": mx.array([1.0, -1.0], dtype=mx.float32),
    }

    converted = dequantize_fp8_weight_only(raw, precision=mx.float32)

    assert "linear.weight_scale" not in converted
    np.testing.assert_allclose(
        np.array(converted["linear.weight"]),
        np.array(expected),
        rtol=0,
        atol=0,
    )
    assert converted["linear.bias"].dtype == mx.float32


class _FakeTokenizer:
    def apply_chat_template(
        self, messages, add_generation_prompt, tokenize
    ):  # noqa: ANN001, ARG002
        return messages[0]["content"][0]["text"]

    def __call__(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        return {"input_ids": [11, 22, 33]}


def test_ideogram4_build_inputs_packs_text_and_image_tokens() -> None:
    pipeline = Ideogram4ImagePipeline.__new__(Ideogram4ImagePipeline)
    pipeline.tokenizer = _FakeTokenizer()

    inputs = pipeline._build_inputs("prompt", height=256, width=256)

    assert inputs["text_token_ids"].shape == (1, 3)
    assert inputs["num_text_tokens"] == 3
    assert inputs["num_image_tokens"] == 16 * 16
    assert inputs["position_ids"].shape == (1, 3 + 16 * 16, 3)
    assert int(inputs["indicator"][0, 0].item()) == LLM_TOKEN_INDICATOR
    assert int(inputs["indicator"][0, -1].item()) == OUTPUT_IMAGE_INDICATOR


def test_ideogram4_pipeline_uses_prepared_prompt_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = Ideogram4ImagePipeline.__new__(Ideogram4ImagePipeline)
    pipeline.model_path = Path("/tmp/fake-ideogram")
    pipeline.runtime_config = Ideogram4RuntimeConfig(
        evict_text_encoder=False,
        evict_transformers=False,
    )
    pipeline.text_encoder = object()
    pipeline.conditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.unconditional_transformer = lambda **kwargs: mx.zeros_like(kwargs["x"])
    pipeline.vae = object()
    captured: dict[str, object] = {}

    def fake_prepare_prompt(prompt: str, **kwargs):
        assert prompt == "plain prompt"
        assert kwargs["prompt_expansion_model"] == "tiny-text-model"
        captured["prepare_kwargs"] = kwargs
        return NormalizedPrompt(
            text=EXPANDED_CAPTION,
            is_json_caption=True,
            is_structured_caption=True,
            was_wrapped=False,
            prompt_expansion_model="tiny-text-model",
            prompt_expansion_used=True,
        )

    def fake_build_inputs(prompt: str, **kwargs):
        captured["tokenized_prompt"] = prompt
        return {
            "text_token_ids": mx.array([[1]], dtype=mx.int32),
            "position_ids": mx.zeros((1, 2, 3), dtype=mx.int32),
            "segment_ids": mx.ones((1, 2), dtype=mx.int32),
            "indicator": mx.ones((1, 2), dtype=mx.int32),
            "num_text_tokens": 1,
            "num_image_tokens": 1,
            "grid_h": 1,
            "grid_w": 1,
        }

    monkeypatch.setattr(pipeline, "prepare_prompt", fake_prepare_prompt)
    monkeypatch.setattr(pipeline, "_build_inputs", fake_build_inputs)
    monkeypatch.setattr(
        pipeline,
        "_encode_text",
        lambda token_ids, num_image_tokens: mx.zeros((1, 2, 4)),
    )
    monkeypatch.setattr(pipeline, "_ensure_transformers_and_vae", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "_decode",
        lambda z, grid_h, grid_w: mx.zeros((16, 16, 3), dtype=mx.uint8),
    )

    array, metadata = pipeline.generate_array(
        "plain prompt",
        steps=1,
        width=256,
        height=256,
        guidance=7.0,
        prompt_expansion_model="tiny-text-model",
    )

    assert array.shape == (16, 16, 3)
    assert captured["tokenized_prompt"] == EXPANDED_CAPTION
    assert metadata["revised_prompt"] == EXPANDED_CAPTION
    assert metadata["prompt_expansion_model"] == "tiny-text-model"
    assert metadata["prompt_expansion_used"]
    assert metadata["prompt_is_structured_caption"]


def test_ideogram4_tiny_transformer_forward_shape() -> None:
    config = Ideogram4TransformerConfig(
        emb_dim=12,
        num_layers=1,
        num_heads=3,
        intermediate_size=16,
        adanln_dim=4,
        in_channels=4,
        llm_features_dim=8,
        mrope_section=(1, 1, 0),
    )
    model = Ideogram4Transformer(config)

    out = model(
        llm_features=mx.zeros((1, 3, 8), dtype=mx.float32),
        x=mx.zeros((1, 3, 4), dtype=mx.float32),
        t=mx.array([0.5], dtype=mx.float32),
        position_ids=mx.zeros((1, 3, 3), dtype=mx.int32),
        segment_ids=mx.ones((1, 3), dtype=mx.int32),
        indicator=mx.array(
            [[LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, OUTPUT_IMAGE_INDICATOR]],
            dtype=mx.int32,
        ),
    )

    assert out.shape == (1, 3, 4)


class _FakeVAE:
    def __init__(self) -> None:
        self.latents = None

    def decode(self, latents):
        self.latents = latents
        return mx.zeros((1, 3, 32, 32), dtype=mx.float32)

    def decode_packed_latents(self, packed):  # noqa: ANN001
        raise AssertionError("Ideogram decode should use latent_norm + vae.decode")


def test_ideogram4_decode_uses_ideogram_latent_norm_path() -> None:
    pipeline = Ideogram4ImagePipeline.__new__(Ideogram4ImagePipeline)
    pipeline.vae = _FakeVAE()

    array = pipeline._decode(mx.zeros((1, 16 * 16, 128)), grid_h=16, grid_w=16)

    assert array.shape == (32, 32, 3)
    assert pipeline.vae.latents.shape == (1, 32, 32, 32)


class _FakePipeline:
    variant = type("Variant", (), {"name": "ideogram-4-fp8"})()

    def generate_array(self, prompt: str, **kwargs):  # noqa: ANN001
        return mx.zeros((8, 10, 3), dtype=mx.uint8), {
            "steps": kwargs["steps"],
            "guidance": kwargs["guidance"],
            "prompt_tokens": 3,
        }


def test_ideogram4_model_wrapper_returns_image_result() -> None:
    model = Ideogram4ImageGenerationModel(
        pipeline=_FakePipeline(),
        model_id="ideogram-ai/ideogram-4-fp8",
    )

    result = model.generate(
        ImageGenerationRequest(
            prompt="caption",
            seed=9,
            steps=2,
            width=10,
            height=8,
            guidance=7.0,
        )
    )

    assert result.width == 10
    assert result.height == 8
    assert result.prompt_tokens == 3
    assert result.variant == "ideogram-4-fp8"


def test_image_generation_cli_forwards_gen_kwargs_and_prompt_expansion_model(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "image.png"
    args = Namespace(
        model="ideogram-ai/ideogram-4-fp8",
        task="generate",
        prompt=["caption"],
        output=str(output_path),
        size="256x256",
        steps=4,
        seed=7,
        guidance=1.0,
        gen_kwargs={"sampler_preset": "V4_TURBO_12"},
        prompt_expansion_model="tiny-text-model",
    )
    result = SimpleNamespace(
        path=output_path,
        seed=7,
        width=256,
        height=256,
        steps=12,
        variant="ideogram-4-fp8",
    )
    model = SimpleNamespace()

    with (
        patch.object(image_module, "load_image_model", return_value=model),
        patch.object(
            image_module, "generate_image", return_value=result
        ) as mock_generate,
    ):
        image_module.run_image_generation_cli(args)

    request = mock_generate.call_args.args[1]
    assert request.extra == {
        "sampler_preset": "V4_TURBO_12",
        "prompt_expansion_model": "tiny-text-model",
    }
