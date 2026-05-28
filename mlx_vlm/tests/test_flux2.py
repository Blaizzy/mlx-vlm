from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_vlm.models.flux2.download as download_module
from mlx_vlm.generate.image import (
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.flux2.config import (
    get_variant,
    validate_dimensions,
    variant_from_local_path,
)
from mlx_vlm.models.flux2.download import DOWNLOAD_PATTERNS, validate_model_layout
from mlx_vlm.models.flux2.model import Flux2ImageGenerationModel
from mlx_vlm.models.flux2.pipeline import Flux2Image, Flux2RuntimeConfig

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
    for relative in (
        "transformer/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "tokenizer/tokenizer.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _fake_pipeline() -> Flux2Image:
    pipeline = Flux2Image.__new__(Flux2Image)
    pipeline.variant = get_variant("flux2-klein-4b")
    pipeline.model_path = None
    pipeline.runtime_config = Flux2RuntimeConfig(tiled_vae="off")
    pipeline.transformer = FakeTransformer()
    pipeline.vae = FakeVAE()
    pipeline.tokenizer = None
    pipeline._encode_prompt = lambda prompt, max_sequence_length: (
        mx.zeros((1, 32, 7680), dtype=mx.bfloat16),
        mx.zeros((1, 32, 4), dtype=mx.int32),
    )
    pipeline._ensure_transformer_and_vae = lambda: None
    return pipeline


@pytest.mark.parametrize(
    "alias,variant",
    [
        ("flux2-klein-4b", "flux2-klein-4b"),
        ("black-forest-labs/FLUX.2-klein-9B", "flux2-klein-9b"),
        ("flux2-base-4B", "flux2-klein-base-4b"),
        ("klein-base-9b", "flux2-klein-base-9b"),
    ],
)
def test_flux2_variant_aliases(alias: str, variant: str) -> None:
    assert get_variant(alias).name == variant


def test_flux2_declares_image_generation_model_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    (tmp_path / "text_encoder" / "config.json").write_text('{"hidden_size": 4096}')
    (tmp_path / "model_index.json").write_text('{"_class_name": "Flux2KleinPipeline"}')

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        assert repo_id == "black-forest-labs/FLUX.2-klein-9B"
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert Flux2ImageGenerationModel.is_image_generation_model
    assert Flux2ImageGenerationModel.model_type == "flux2"
    assert image_generation_model_class("flux2-klein-4b") is Flux2ImageGenerationModel
    assert (
        image_generation_model_class("black-forest-labs/FLUX.2-klein-9B")
        is Flux2ImageGenerationModel
    )
    assert (
        image_generation_model_class(tmp_path.as_posix()) is Flux2ImageGenerationModel
    )
    assert is_image_generation_model("klein-base-9b")


def test_flux2_image_model_class_uses_remote_model_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name": "Flux2KleinPipeline"}')

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        assert repo_id == "example/custom-image-model"
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert (
        image_generation_model_class("example/custom-image-model")
        is Flux2ImageGenerationModel
    )


def test_is_image_generation_model_does_not_probe_remote_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        raise AssertionError("remote model path should not be resolved")

    monkeypatch.setattr(image_module, "get_model_path", fail_get_model_path)

    assert not is_image_generation_model("example/custom-image-model")


def test_flux2_validate_model_layout_accepts_required_files(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    assert validate_model_layout(tmp_path) == tmp_path


def test_flux2_validate_model_layout_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transformer"):
        validate_model_layout(tmp_path)


def test_flux2_variant_from_local_path_reads_config(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    config = tmp_path / "transformer" / "config.json"
    config.write_text('{"num_layers": 8, "num_attention_heads": 32}')

    assert variant_from_local_path(tmp_path).name == "flux2-klein-9b"


@pytest.mark.parametrize("width,height", [(255, 512), (512, 2050), (513, 512)])
def test_flux2_validate_dimensions_rejects_bad_sizes(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_flux2_download_model_uses_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {}
    cached = tmp_path / "hf-cache" / "snapshot"
    _write_layout(cached)

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(cached)

    monkeypatch.setattr(
        download_module, "find_valid_cached_snapshot", lambda variant: None
    )
    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot_download)
    path = download_module.download_model("flux2-klein-9b", max_workers=2)

    assert path == cached
    assert calls["repo_id"] == "black-forest-labs/FLUX.2-klein-9B"
    assert calls["allow_patterns"] == list(DOWNLOAD_PATTERNS)
    assert "model_index.json" in calls["allow_patterns"]
    assert "local_dir" not in calls
    assert calls["max_workers"] == 2


def test_flux2_generate_returns_pil_image() -> None:
    pipeline = _fake_pipeline()
    image = pipeline.generate("prompt", seed=7, steps=1, width=512, height=512)
    assert image.size == (512, 512)


def test_flux2_generate_rejects_empty_prompt() -> None:
    pipeline = _fake_pipeline()
    with pytest.raises(ValueError, match="prompt"):
        pipeline.generate("", width=512, height=512)
