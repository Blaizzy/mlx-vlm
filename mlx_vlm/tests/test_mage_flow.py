from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_vlm.models.mage_flow.download as download_module
from mlx_vlm.generate.edit_image import image_edit_model_class, is_image_edit_model
from mlx_vlm.generate.image import (
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.mage_flow.config import (
    VARIANTS,
    get_variant,
    validate_dimensions,
    variant_from_local_path,
)
from mlx_vlm.models.mage_flow.download import DOWNLOAD_PATTERNS, validate_model_layout
from mlx_vlm.models.mage_flow.model import (
    MageFlowImageEditModel,
    MageFlowImageGenerationModel,
)
from mlx_vlm.models.mage_flow.scheduler import FlowMatchEulerDiscreteScheduler
from mlx_vlm.models.mage_flow.transformer import (
    MageFlowTransformer,
    image_rope_frequencies,
)
from mlx_vlm.models.mage_flow.weights import (
    sanitize_transformer_weights,
    sanitize_vae_weights,
)

image_module = importlib.import_module("mlx_vlm.generate.image")


def _write_layout(root: Path) -> None:
    for relative in (
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/tokenizer.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if path.suffix == ".json" else b"x")
    (root / "model_index.json").write_text('{"_class_name":"MageFlowPipeline"}')


@pytest.mark.parametrize(
    "model_id,variant,task,steps,guidance",
    [
        ("microsoft/Mage-Flow-Base", "mage-flow-base", "generate", 30, 5.0),
        ("microsoft/Mage-Flow", "mage-flow", "generate", 20, 5.0),
        ("microsoft/Mage-Flow-Turbo", "mage-flow-turbo", "generate", 4, 1.0),
        (
            "microsoft/Mage-Flow-Edit-Base",
            "mage-flow-edit-base",
            "edit",
            30,
            5.0,
        ),
        ("microsoft/Mage-Flow-Edit", "mage-flow-edit", "edit", 30, 5.0),
        (
            "microsoft/Mage-Flow-Edit-Turbo",
            "mage-flow-edit-turbo",
            "edit",
            4,
            1.0,
        ),
    ],
)
def test_mage_flow_variants(
    model_id: str, variant: str, task: str, steps: int, guidance: float
) -> None:
    spec = get_variant(model_id)
    assert spec.name == variant
    assert spec.task == task
    assert spec.default_steps == steps
    assert spec.default_guidance == guidance


def test_mage_flow_registers_generation_and_edit_families() -> None:
    assert (
        image_generation_model_class("microsoft/Mage-Flow")
        is MageFlowImageGenerationModel
    )
    assert (
        image_edit_model_class("microsoft/Mage-Flow-Edit-Turbo")
        is MageFlowImageEditModel
    )
    assert is_image_generation_model("mage-flow-base")
    assert not is_image_generation_model("mage-flow-edit")
    assert is_image_edit_model("mage-flow-edit-base")
    assert not is_image_edit_model("mage-flow-turbo")


def test_mage_flow_remote_metadata_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)

    def fake_get_model_path(repo_id: str, **kwargs):  # noqa: ARG001
        assert repo_id == "example/custom-mage"
        return tmp_path

    monkeypatch.setattr(image_module, "get_model_path", fake_get_model_path)
    assert (
        image_generation_model_class("example/custom-mage")
        is MageFlowImageGenerationModel
    )


def test_mage_flow_local_variant_uses_cache_parent_name(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "models--microsoft--Mage-Flow-Edit-Turbo" / "snapshots" / "hash"
    )
    _write_layout(snapshot)
    assert variant_from_local_path(snapshot).name == "mage-flow-edit-turbo"


def test_mage_flow_layout_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transformer"):
        validate_model_layout(tmp_path)
    _write_layout(tmp_path)
    assert validate_model_layout(tmp_path) == tmp_path


def test_mage_flow_download_uses_family_patterns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot_download)
    assert download_module.download_model("mage-flow-turbo", max_workers=2) == tmp_path
    assert calls["repo_id"] == "microsoft/Mage-Flow-Turbo"
    assert calls["allow_patterns"] == list(DOWNLOAD_PATTERNS)
    assert calls["max_workers"] == 2


@pytest.mark.parametrize(
    "width,height", [(511, 512), (512, 2049), (513, 512), (512, 496)]
)
def test_mage_flow_rejects_unsupported_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_mage_flow_scheduler_matches_static_shift() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(num_inference_steps=4, shift=6.0)
    expected = np.array([1.0, 4.5 / 4.75, 3.0 / 3.5, 1.5 / 2.25, 0.0])
    np.testing.assert_allclose(np.array(scheduler.sigmas), expected, rtol=1e-6)


def test_mage_flow_tiny_transformer_forward() -> None:
    transformer = MageFlowTransformer(
        in_channels=8,
        out_channels=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=4,
        depth=2,
        axes_dim=(2, 2, 4),
    )
    output = transformer(
        img=mx.zeros((1, 4, 8), dtype=mx.float32),
        txt=mx.zeros((1, 3, 16), dtype=mx.float32),
        timesteps=mx.array([1.0]),
        img_shapes=[(1, 2, 2)],
    )
    mx.eval(output)
    assert output.shape == (1, 4, 8)
    assert bool(mx.all(mx.isfinite(output)))


def test_mage_flow_rope_covers_target_and_references() -> None:
    cosine, sine = image_rope_frequencies([(1, 2, 3), (1, 2, 3)], axes_dim=(2, 2, 4))
    assert cosine.shape == (12, 4)
    assert sine.shape == (12, 4)
    assert not np.allclose(np.array(cosine[:6, 0]), np.array(cosine[6:, 0]))
    np.testing.assert_allclose(np.array(cosine[:6, 1:]), np.array(cosine[6:, 1:]))


def test_mage_flow_weight_sanitizers() -> None:
    transformer = sanitize_transformer_weights(
        {
            "transformer_blocks.0.img_mod.1.weight": mx.zeros((6, 1)),
            "transformer_blocks.0.attn.to_out.0.bias": mx.zeros((1,)),
            "transformer_blocks.0.txt_mlp.net.0.proj.weight": mx.zeros((4, 1)),
        }
    )
    assert "transformer_blocks.0.img_mod.linear.weight" in transformer
    assert "transformer_blocks.0.attn.to_out.bias" in transformer
    assert "transformer_blocks.0.txt_mlp.linear_in.weight" in transformer

    vae = sanitize_vae_weights(
        {
            "student.dconv_encoder.blocks.0.ca.1.weight": mx.zeros((2, 2, 1, 1)),
            "pipeline.dec_net.res_blocks.0.mlp.2.weight": mx.zeros((2, 2)),
            "pipeline.y_embedder.encoder.encoder.conv_in.weight": mx.zeros(
                (2, 2, 3, 3)
            ),
        }
    )
    assert vae["dconv_encoder.blocks.0.ca_conv.weight"].shape == (2, 1, 1, 2)
    assert "decoder_model.dec_net.res_blocks.0.linear_2.weight" in vae
    assert len(vae) == 2


def test_all_six_variants_are_present() -> None:
    assert set(VARIANTS) == {
        "mage-flow-base",
        "mage-flow",
        "mage-flow-turbo",
        "mage-flow-edit-base",
        "mage-flow-edit",
        "mage-flow-edit-turbo",
    }
