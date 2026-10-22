"""Flux3 short-prompt expansion and multimodal editing without network IO."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.generate.edit_image import ImageEditRequest
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.flux3 import bbox
from mlx_vlm.models.flux3.config import Flux3Config
from mlx_vlm.models.flux3.model import Flux3ImageEditModel, Flux3ImageGenerationModel
from mlx_vlm.models.flux3.pipeline import Flux3Image
from mlx_vlm.models.flux3.upsampler import (
    DEFAULT_SYSTEM_PROMPTS,
    DEFAULT_UPSAMPLER,
    MAX_TOKENS,
    Flux3PromptUpsampler,
    prepare_reference,
    reference_size,
)

dispatch = importlib.import_module("mlx_vlm.generate.dispatch")
pipeline_module = importlib.import_module("mlx_vlm.models.flux3.pipeline")


def test_upsampler_checkpoint_is_not_dispatched_as_an_image_model(tmp_path):
    root = tmp_path / "flux-3-image-klein-pup"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"qwen3_5"}')
    assert not is_image_generation_model(DEFAULT_UPSAMPLER)
    assert image_generation_model_class(str(root)) is None


@pytest.fixture
def mocked_upsampler(monkeypatch):
    upsampler = Flux3PromptUpsampler()
    processor = SimpleNamespace(apply_chat_template=Mock(return_value="formatted"))
    model = object()

    def load():
        upsampler.model, upsampler.processor = model, processor
        upsampler.prompts = {
            key: f"system-{key}" for key in ("t2i", "i2i", "t2i_bbox", "i2i_bbox")
        }

    monkeypatch.setattr(upsampler, "_load", load)
    generate = Mock(
        return_value=SimpleNamespace(
            text=" A complete scene caption. ", finish_reason="stop"
        )
    )
    monkeypatch.setattr(dispatch, "generate", generate)
    return upsampler, processor, generate


@pytest.mark.parametrize("edit", [False, True])
def test_upsampler_selects_task_scaffold_and_native_generation_settings(
    mocked_upsampler, edit
):
    upsampler, processor, generate = mocked_upsampler
    images = (
        [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")]
        if edit
        else []
    )
    assert (
        upsampler("make it green" if edit else "a green teapot", images)
        == "A complete scene caption."
    )
    messages = processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"] == ("system-i2i" if edit else "system-t2i")
    assert processor.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    settings = generate.call_args.kwargs
    assert settings["temperature"] == 0
    assert settings["max_tokens"] == MAX_TOKENS
    assert settings["presence_penalty"] == (0 if edit else 1.5)
    assert settings["presence_context_size"] == MAX_TOKENS
    assert settings["eos_tokens"] == ["<|im_end|>", "<|endoftext|>"]
    assert settings["enable_thinking"] is False
    if edit:
        assert messages[1]["content"][:2] == [{"type": "image"}, {"type": "image"}]
        assert "User request: make it green" in messages[1]["content"][2]["text"]
        assert len(settings["image"]) == 2
        assert settings["image"][0].getpixel((0, 0))[0] > 240
        assert settings["image"][1].getpixel((0, 0))[2] > 240
    else:
        assert messages[1]["content"] == "a green teapot"
        assert settings["image"] is None
    assert upsampler.model is upsampler.processor is None


@pytest.mark.parametrize(
    "text,reason,error",
    [(" ", "stop", "empty"), ("unfinished", "length", "token limit")],
)
def test_failed_expansion_releases_model_and_does_not_use_raw_prompt(
    mocked_upsampler, text, reason, error
):
    upsampler, _, generate = mocked_upsampler
    generate.return_value = SimpleNamespace(text=text, finish_reason=reason)
    with pytest.raises(RuntimeError, match=error):
        upsampler("a cat")
    assert upsampler.model is upsampler.processor is None


def test_generation_error_releases_model(mocked_upsampler):
    upsampler, _, generate = mocked_upsampler
    generate.side_effect = ValueError("inference failed")
    with pytest.raises(ValueError, match="inference failed"):
        upsampler("a cat")
    assert upsampler.model is upsampler.processor is None


def test_finished_layout_caption_bypasses_upsampler_without_loading(monkeypatch):
    upsampler = Flux3PromptUpsampler()
    load = Mock(side_effect=AssertionError("no load"))
    monkeypatch.setattr(upsampler, "_load", load)
    prompt = 'A green teapot <pot>. [{"id":"pot","bbox":[100,200,800,900],"desc":"green teapot"}]'
    assert upsampler(prompt) == prompt
    load.assert_not_called()


@pytest.mark.parametrize("edit", [False, True])
def test_layout_expansion_preserves_locked_coordinates_and_reference_provenance(
    mocked_upsampler, edit
):
    upsampler, processor, generate = mocked_upsampler
    box = {
        "id": "pot",
        "desc": "green teapot",
        **(
            {"from": "ref_image_0", "src_bbox": [0.1, 0.2, 0.8, 0.9], "tgt_bbox": None}
            if edit
            else {"bbox": [0.1, 0.2, 0.8, 0.9]}
        ),
    }
    generate.return_value.text = 'A green teapot <pot>. [{"id":"pot","bbox":[0,0,1000,1000],"desc":"a glossy green teapot"}]'
    caption = upsampler(
        "a green teapot", [Image.new("RGB", (64, 64))] if edit else [], boxes=[box]
    )
    messages = processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"] == ("system-i2i_bbox" if edit else "system-t2i_bbox")
    row = bbox.split(caption)[1][0]
    assert row["id"] == "pot"
    assert row["desc"] == "a glossy green teapot"
    if edit:
        assert row["from"] == "ref_image_0"
        assert row["src_bbox"] == [100, 200, 800, 900]
        assert row["tgt_bbox"] is None
    else:
        assert row["bbox"] == [100, 200, 800, 900]


def test_load_prefers_checkpoint_prompts_and_falls_back_to_inline_prompts(
    monkeypatch, tmp_path
):
    from mlx_vlm import utils

    monkeypatch.setattr(utils, "get_model_path", Mock(return_value=tmp_path))
    load = Mock(return_value=(object(), object()))
    monkeypatch.setattr(utils, "load", load)
    upsampler = Flux3PromptUpsampler(str(tmp_path))
    assert upsampler.model is None
    upsampler._load()
    assert upsampler.prompts == DEFAULT_SYSTEM_PROMPTS
    custom = {key: f"custom-{key}" for key in upsampler.prompts}
    (tmp_path / "system_prompts.json").write_text(json.dumps(custom))
    upsampler.unload()
    upsampler._load()
    assert upsampler.prompts == custom


@pytest.mark.parametrize(
    "count,cap", [(1, 1024**2), (2, 724**2), (3, 512**2), (10, 512**2)]
)
def test_reference_size_follows_per_reference_budget(count, cap):
    for w, h in [(4000, 3000), (1, 200), (200, 1), (67, 99), (4096, 2048)]:
        width, height = reference_size(w, h, count)
        assert width % 32 == height % 32 == 0
        assert width * height <= cap


def test_reference_alpha_is_composited_over_white():
    out = prepare_reference(Image.new("RGBA", (64, 64), (255, 0, 0, 0)), 1)
    assert out.mode == "RGB"
    assert np.asarray(out).min() >= 254


def test_invalid_references_fail_before_model_load(monkeypatch):
    upsampler = Flux3PromptUpsampler()
    load = Mock(side_effect=AssertionError("no load"))
    monkeypatch.setattr(upsampler, "_load", load)
    with pytest.raises(ValueError, match="at most 10"):
        upsampler("caption", [Image.new("RGB", (32, 32))] * 11)
    with pytest.raises(ValueError, match="200:1"):
        upsampler("caption", [Image.new("RGB", (1, 201))])
    load.assert_not_called()


def pipeline_stub():
    pipeline = Flux3Image.__new__(Flux3Image)
    pipeline.upsample_prompt = True
    pipeline.prompt_expansion_model = DEFAULT_UPSAMPLER
    pipeline.evict_upsampler = True
    pipeline.upsampler = None
    pipeline.upsampled_prompts = {}
    pipeline.last_prompt_metadata = {}
    pipeline.text_encoder = pipeline.transformer = pipeline.vae = object()
    return pipeline


def test_pipeline_expands_by_default_caches_text_only_and_releases_diffusion_stage(
    monkeypatch,
):
    upsampler = Mock(model_id=DEFAULT_UPSAMPLER, return_value="expanded caption")
    factory = Mock(return_value=upsampler)
    monkeypatch.setattr(pipeline_module, "Flux3PromptUpsampler", factory)
    pipeline = pipeline_stub()
    args = {"upsample_prompt": None, "prompt_expansion_model": None, "boxes": None}
    assert pipeline._prepare_prompt("a cat", [], **args) == "expanded caption"
    assert pipeline.transformer is pipeline.vae is pipeline.text_encoder is None
    assert pipeline.last_prompt_metadata["upsampled"] is True
    assert pipeline.last_prompt_metadata["revised_prompt"] == "expanded caption"
    assert pipeline._prepare_prompt("a cat", [], **args) == "expanded caption"
    assert upsampler.call_count == 1
    images = [Image.new("RGB", (64, 64))]
    pipeline._prepare_prompt("a cat", images, **args)
    pipeline._prepare_prompt("a cat", images, **args)
    assert upsampler.call_count == 3
    assert upsampler.call_args.args[1] is images


def test_opt_out_never_loads_upsampler_and_keeps_ready_models(monkeypatch):
    factory = Mock(side_effect=AssertionError("no load"))
    monkeypatch.setattr(pipeline_module, "Flux3PromptUpsampler", factory)
    pipeline = pipeline_stub()
    before = pipeline.transformer
    assert (
        pipeline._prepare_prompt(
            "full caption",
            [],
            upsample_prompt=False,
            prompt_expansion_model=None,
            boxes=None,
        )
        == "full caption"
    )
    assert pipeline.last_prompt_metadata["upsampled"] is False
    assert pipeline.transformer is before
    factory.assert_not_called()


@pytest.mark.parametrize("edit", [False, True])
def test_public_wrappers_forward_upsampling_options_and_return_caption(edit):
    call = Mock(return_value=mx.zeros((64, 64, 3), dtype=mx.uint8))
    tokenizer = Mock(count_tokens=Mock(return_value=123))
    pipeline = SimpleNamespace(
        config=Flux3Config(),
        model_path=Path("unused"),
        generate_array=call,
        edit_array=call,
        tokenizer=tokenizer,
        last_prompt_metadata={"revised_prompt": "expanded caption", "upsampled": True},
    )
    extra = {
        "upsample_prompt": False,
        "prompt_expansion_model": "local-pup",
        "boxes": [{"id": "cat", "bbox": [0, 0, 1, 1]}],
    }
    if edit:
        result = Flux3ImageEditModel(pipeline, "flux3").edit(
            ImageEditRequest("a cat", ("reference.png",), extra=extra)
        )
    else:
        result = Flux3ImageGenerationModel(pipeline, "flux3").generate(
            ImageGenerationRequest("a cat", extra=extra)
        )
    assert {key: call.call_args.kwargs[key] for key in extra} == extra
    assert result.metadata["revised_prompt"] == "expanded caption"
    assert result.prompt_tokens == 123
    tokenizer.count_tokens.assert_called_once_with("expanded caption")
