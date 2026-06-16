from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import mlx_vlm.models.flux2.download as download_module
from mlx_vlm.generate.edit_image import (
    ImageEditRequest,
    image_edit_model_class,
    is_image_edit_model,
)
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
from mlx_vlm.models.flux2.model import Flux2ImageEditModel, Flux2ImageGenerationModel
from mlx_vlm.models.flux2.pipeline import (
    Flux2Image,
    Flux2ImageEdit,
    Flux2RuntimeConfig,
    _reference_image_array,
)

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


def _fake_edit_pipeline(variant: str = "flux2-klein-9b-kv") -> Flux2ImageEdit:
    pipeline = Flux2ImageEdit.__new__(Flux2ImageEdit)
    pipeline.variant = get_variant(variant)
    pipeline.model_path = None
    pipeline.runtime_config = Flux2RuntimeConfig(tiled_vae="off")
    pipeline.tokenizer = type("Tokenizer", (), {"count_tokens": lambda self, text: 7})()
    pipeline.edit_array = lambda *args, **kwargs: mx.zeros((20, 24, 3), dtype=mx.uint8)
    return pipeline


@pytest.mark.parametrize(
    "alias,variant",
    [
        ("flux2-klein-4b", "flux2-klein-4b"),
        ("black-forest-labs/FLUX.2-klein-9B", "flux2-klein-9b"),
        ("flux2-base-4B", "flux2-klein-base-4b"),
        ("klein-base-9b", "flux2-klein-base-9b"),
        ("black-forest-labs/FLUX.2-klein-9b-kv", "flux2-klein-9b-kv"),
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


def test_flux2_declares_image_edit_model_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name": "Flux2KleinPipeline"}')
    (tmp_path / "flux-2-klein-9b-kv.safetensors").write_bytes(b"x")

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        assert repo_id == "black-forest-labs/FLUX.2-klein-9b-kv"
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)

    assert Flux2ImageEditModel.is_image_edit_model
    assert Flux2ImageEditModel.model_type == "flux2"
    assert (
        image_edit_model_class("black-forest-labs/FLUX.2-klein-9b-kv")
        is Flux2ImageEditModel
    )
    assert image_edit_model_class(tmp_path.as_posix()) is Flux2ImageEditModel
    assert is_image_edit_model("black-forest-labs/FLUX.2-klein-9b-kv")
    assert is_image_edit_model("black-forest-labs/FLUX.2-klein-9B")


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


def test_flux2_variant_from_local_path_prefers_kv_marker(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    config = tmp_path / "transformer" / "config.json"
    config.write_text('{"num_layers": 8, "num_attention_heads": 32}')
    (tmp_path / "flux-2-klein-9b-kv.safetensors").write_bytes(b"x")

    assert variant_from_local_path(tmp_path).name == "flux2-klein-9b-kv"


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


def test_flux2_edit_model_returns_image_result() -> None:
    model = Flux2ImageEditModel(pipeline=_fake_edit_pipeline(), model_id="kv")
    result = model.edit(
        ImageEditRequest(
            prompt="add sunglasses",
            image_paths=("reference.png",),
            seed=9,
            steps=2,
            guidance=1.0,
        )
    )

    assert result.width == 24
    assert result.height == 20
    assert result.metadata["uses_reference_kv_cache"] is True
    assert result.metadata["reference_count"] == 1


def test_flux2_standard_edit_model_reports_non_kv_path() -> None:
    model = Flux2ImageEditModel(
        pipeline=_fake_edit_pipeline("flux2-klein-9b"),
        model_id="standard",
    )
    result = model.edit(
        ImageEditRequest(
            prompt="add sunglasses",
            image_paths=("reference.png",),
            seed=9,
            steps=2,
            guidance=1.0,
        )
    )

    assert result.variant == "flux2-klein-9b"
    assert result.metadata["uses_reference_kv_cache"] is False


def test_flux2_edit_model_defaults_to_untiled_vae_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_from_pretrained(cls, variant, **kwargs):  # noqa: ARG001
        calls.append(kwargs)
        return _fake_edit_pipeline("flux2-klein-9b-kv")

    monkeypatch.setattr(
        Flux2ImageEdit, "from_pretrained", classmethod(fake_from_pretrained)
    )

    Flux2ImageEditModel.from_model_id("flux2-klein-9b-kv")
    Flux2ImageEditModel.from_model_id(
        "flux2-klein-9b-kv", bucketed_seq_len=True, tiled_vae="auto"
    )

    assert calls[0]["tiled_vae"] == "off"
    assert calls[0]["bucketed_seq_len"] is False
    assert calls[1]["tiled_vae"] == "auto"
    assert calls[1]["bucketed_seq_len"] is True


def test_flux2_reference_image_array_keeps_float32_input() -> None:
    image = Image.new("RGB", (1, 1), color=(255, 127, 0))
    array = _reference_image_array(image)

    assert array.dtype == mx.float32
    assert np.array(array).shape == (1, 3, 1, 1)
