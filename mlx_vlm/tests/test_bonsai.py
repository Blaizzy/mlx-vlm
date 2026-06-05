from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_vlm.models.bonsai.download as download_module
from mlx_vlm.generate.image import (
    ImageGenerationResult,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.bonsai.config import get_variant, parse_size, validate_dimensions
from mlx_vlm.models.bonsai.download import (
    REQUIRED_FILES,
    download_model,
    validate_model_layout,
)
from mlx_vlm.models.bonsai.model import BonsaiImageGenerationModel
from mlx_vlm.models.bonsai.pipeline import BonsaiImage, BonsaiRuntimeConfig

image_module = importlib.import_module("mlx_vlm.generate.image")


class FakeTransformer:
    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["hidden_states"])


class FakeVAE:
    def decode_packed_latents(self, packed, tiling_config=None):  # noqa: ARG002
        return mx.zeros(
            (1, 3, packed.shape[2] * 16, packed.shape[3] * 16),
            dtype=mx.bfloat16,
        )


def _write_layout(root: Path) -> None:
    for relative in REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _fake_pipeline() -> BonsaiImage:
    pipeline = BonsaiImage.__new__(BonsaiImage)
    pipeline.variant = get_variant("ternary")
    pipeline.model_path = None
    pipeline.runtime_config = BonsaiRuntimeConfig(tiled_vae="off")
    pipeline.transformer = FakeTransformer()
    pipeline.vae = FakeVAE()
    pipeline.tokenizer = None
    pipeline._encode_prompt = lambda prompt, max_sequence_length: (
        mx.zeros((1, 32, 7680), dtype=mx.bfloat16),
        mx.zeros((1, 32, 4), dtype=mx.int32),
    )
    pipeline._ensure_transformer_and_vae = lambda: None
    return pipeline


def test_bonsai_variant_aliases_are_ternary_only() -> None:
    assert get_variant("bonsai").precision == "2bit"
    assert get_variant("bonsai-ternary").name == "ternary"
    assert get_variant("prism-ml/bonsai-image-ternary-4B-mlx-2bit").name == "ternary"
    with pytest.raises(ValueError, match="Unknown Bonsai variant"):
        get_variant("binary")


def test_bonsai_declares_image_generation_model_type(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    (tmp_path / "manifest.json").write_text(
        """
        {
          "files": [
            {"remote_path": "transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
            {"remote_path": "text_encoder-mlx-4bit/model.safetensors"},
            {"remote_path": "tokenizer/tokenizer.json"}
          ]
        }
        """
    )

    assert BonsaiImageGenerationModel.is_image_generation_model
    assert BonsaiImageGenerationModel.model_type == "bonsai"
    assert image_generation_model_class("bonsai-ternary") is BonsaiImageGenerationModel
    assert (
        image_generation_model_class(tmp_path.as_posix()) is BonsaiImageGenerationModel
    )
    assert is_image_generation_model("bonsai-ternary")
    assert not is_image_generation_model("mlx-community/nanoLLaVA-1.5-8bit")


def test_bonsai_image_model_class_uses_remote_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    (tmp_path / "manifest.json").write_text(
        """
        {
          "files": [
            {"remote_path": "transformer-packed-mflux/diffusion_pytorch_model.safetensors"},
            {"remote_path": "text_encoder-mlx-4bit/model.safetensors"},
            {"remote_path": "tokenizer/tokenizer.json"}
          ]
        }
        """
    )

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        assert repo_id == "example/custom-image-model"
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert (
        image_generation_model_class("example/custom-image-model")
        is BonsaiImageGenerationModel
    )


def test_bonsai_parse_size() -> None:
    assert parse_size("1248x832") == (1248, 832)
    assert parse_size("832x1248") == (832, 1248)


@pytest.mark.parametrize("width,height", [(255, 512), (512, 2050), (513, 512)])
def test_bonsai_validate_dimensions_rejects_bad_sizes(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_bonsai_validate_model_layout_accepts_required_files(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    assert validate_model_layout(tmp_path) == tmp_path


def test_bonsai_validate_model_layout_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transformer-packed-mflux"):
        validate_model_layout(tmp_path)


def test_bonsai_download_model_uses_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        _write_layout(Path(kwargs["local_dir"]))
        return kwargs["local_dir"]

    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot_download)
    path = download_model("ternary", models_dir=tmp_path, max_workers=3)
    assert path == tmp_path / "bonsai-image-4B-ternary-mlx"
    assert calls["repo_id"] == "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
    assert calls["local_dir"] == str(tmp_path / "bonsai-image-4B-ternary-mlx")
    assert calls["max_workers"] == 3


def test_bonsai_download_model_uses_default_hf_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {}
    cached = tmp_path / "hf-cache" / "snapshot"
    _write_layout(cached)

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(cached)

    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot_download)
    path = download_model("ternary", max_workers=2)

    assert path == cached
    assert calls["repo_id"] == "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
    assert "local_dir" not in calls
    assert calls["max_workers"] == 2


def test_bonsai_generate_returns_pil_image() -> None:
    pipeline = _fake_pipeline()
    image = pipeline.generate("prompt", seed=7, steps=4, width=512, height=512)
    assert image.size == (512, 512)


def test_bonsai_generate_rejects_empty_prompt() -> None:
    pipeline = _fake_pipeline()
    with pytest.raises(ValueError, match="prompt"):
        pipeline.generate("", width=512, height=512)


def test_image_generation_result_serializes_array(tmp_path: Path) -> None:
    data = ImageGenerationResult(
        array=mx.zeros((8, 8, 3), dtype=mx.uint8),
        seed=1,
        width=8,
        height=8,
        steps=1,
        model="bonsai",
        family="bonsai",
        variant="ternary",
        guidance=1.0,
        peak_memory=0.0,
    )

    output_path = data.save(tmp_path / "image.png")

    assert data.array.shape == (8, 8, 3)
    assert output_path.exists()
    assert data.to_b64_json()
    assert np.array(data.to_pil()).shape == (8, 8, 3)
