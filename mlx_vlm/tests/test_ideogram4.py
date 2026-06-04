from __future__ import annotations

import importlib
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
from mlx_vlm.models.ideogram4.pipeline import Ideogram4ImagePipeline
from mlx_vlm.models.ideogram4.scheduler import get_preset, make_step_intervals
from mlx_vlm.models.ideogram4.transformer import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4Transformer,
)
from mlx_vlm.models.ideogram4.weights import dequantize_fp8_weight_only

image_module = importlib.import_module("mlx_vlm.generate.image")


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


def test_image_generation_cli_forwards_gen_kwargs(tmp_path: Path) -> None:
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
    assert request.extra == {"sampler_preset": "V4_TURBO_12"}
