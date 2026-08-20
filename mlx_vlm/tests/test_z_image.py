from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx_vlm.generate.edit_image import (
    ImageEditRequest,
    image_edit_model_class,
    is_image_edit_model,
)
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.z_image.config import (
    ZImageConfig,
    ZImageTextEncoderConfig,
    ZImageTransformerConfig,
    ZImageVAEConfig,
    detect_z_image_layout,
)
from mlx_vlm.models.z_image.convert import (
    _sanitize_vae_for_conversion,
    _save_component,
    is_z_image_model_path,
)
from mlx_vlm.models.z_image.model import ZImageEditModel, ZImageGenerationModel
from mlx_vlm.models.z_image.pipeline import ZImagePipeline, _img2img_start_index
from mlx_vlm.models.z_image.text_encoder import (
    ZImageTextEncoder,
    sanitize_text_encoder_weights,
)
from mlx_vlm.models.z_image.transformer import (
    ZImageTransformer,
    sanitize_transformer_weights,
)
from mlx_vlm.models.z_image.vae import ZImageVAE, sanitize_vae_weights

# --- Config / Layout tests ---


def test_detect_z_image_layout(tmp_path: Path) -> None:
    """Positive and negative layout detection."""
    # Missing tokenizer → False
    assert not detect_z_image_layout(tmp_path)
    # Create full layout
    for rel in (
        "model_index.json",
        "transformer/config.json",
        "transformer/model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "vae/config.json",
        "vae/model.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("{}")
    assert detect_z_image_layout(tmp_path)


def test_detects_diffusers_z_image_model_index(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}')
    assert is_z_image_model_path(tmp_path)
    (tmp_path / "model_index.json").write_text('{"_class_name":"FluxPipeline"}')
    assert not is_z_image_model_path(tmp_path)


@pytest.mark.parametrize(
    "mode,expected_bits,expected_group_size",
    [("affine", 4, 64), ("mxfp8", 8, 32)],
)
def test_generic_convert_dispatches_z_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_bits: int,
    expected_group_size: int,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model_index.json").write_text('{"_class_name":"ZImagePipeline"}')
    output = tmp_path / "output"
    calls = {}
    convert_module = importlib.import_module("mlx_vlm.convert")
    z_image_convert = importlib.import_module("mlx_vlm.models.z_image.convert")
    monkeypatch.setattr(
        convert_module, "get_model_path", lambda *args, **kwargs: source
    )

    def fake_convert(model_path, output_path, **kwargs):
        calls.update(model_path=model_path, output_path=output_path, **kwargs)
        return output

    monkeypatch.setattr(z_image_convert, "convert_z_image", fake_convert)
    result = convert_module.convert(
        "Tongyi-MAI/Z-Image-Turbo",
        str(output),
        quantize=True,
        q_mode=mode,
    )
    assert result == output
    assert calls["model_path"] == source
    assert calls["q_group_size"] == expected_group_size
    assert calls["q_bits"] == expected_bits
    assert calls["q_mode"] == mode
    assert calls["quantize_vae"] is False


# --- Dispatch tests ---


def test_dispatch_via_image_generation_model_class(tmp_path: Path) -> None:
    for relative in (
        "model_index.json",
        "transformer/config.json",
        "transformer/model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "vae/config.json",
        "vae/model.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"_class_name":"ZImagePipeline"}')
    assert ZImageGenerationModel.supports_model(str(tmp_path))
    cls = image_generation_model_class(str(tmp_path))
    assert cls is ZImageGenerationModel
    assert is_image_generation_model(str(tmp_path))
    assert image_generation_model_class("Tongyi-MAI/Z-Image") is ZImageGenerationModel


def test_dispatch_via_image_edit_model_class(tmp_path: Path) -> None:
    for relative in (
        "model_index.json",
        "transformer/config.json",
        "transformer/model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "vae/config.json",
        "vae/model.safetensors",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"_class_name":"ZImagePipeline"}')
    assert ZImageEditModel.supports_model(str(tmp_path))
    assert image_edit_model_class(str(tmp_path)) is ZImageEditModel
    assert is_image_edit_model(str(tmp_path))


def test_edit_model_forwards_img2img_options() -> None:
    calls = {}

    class FakePipeline:
        config = ZImageConfig(
            default_steps=9,
            default_guidance=0.0,
            scheduler_shift=3.0,
            variant="turbo",
        )
        model_path = Path("/tmp/z-image")

        def edit_array(self, prompt: str, image_paths, **kwargs):
            calls.update(prompt=prompt, image_paths=image_paths, **kwargs)
            return mx.zeros((16, 32, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = ZImageEditModel(
        pipeline=FakePipeline(),
        model_id="Tongyi-MAI/Z-Image-Turbo",
    )
    result = model.edit(
        ImageEditRequest(
            prompt="replace the cart",
            image_paths=("source.png",),
            extra={"strength": 0.55},
        )
    )
    assert calls["steps"] == 8
    assert calls["guidance"] == 0.0
    assert calls["strength"] == 0.55
    assert result.width == 32
    assert result.height == 16

    model.edit(
        ImageEditRequest(
            prompt="replace the cart",
            image_paths=("source.png",),
            steps=4,
            guidance=1.0,
        )
    )
    assert calls["steps"] == 4
    assert calls["guidance"] == 1.0


@pytest.mark.parametrize(
    "steps,strength,expected",
    [(9, 0.6, 3), (9, 0.5, 4), (8, 0.6, 3), (8, 0.3, 5)],
)
def test_img2img_start_index_matches_diffusers(
    steps: int,
    strength: float,
    expected: int,
) -> None:
    assert _img2img_start_index(steps, strength) == expected


def test_base_model_forwards_cfg_options() -> None:
    calls = {}

    class FakePipeline:
        config = ZImageConfig(
            default_steps=50,
            default_guidance=4.0,
            scheduler_shift=6.0,
            variant="base",
        )
        model_path = Path("/tmp/z-image")

        def generate_array(self, prompt: str, **kwargs):
            calls.update(prompt=prompt, **kwargs)
            return mx.zeros((16, 16, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = ZImageGenerationModel(
        pipeline=FakePipeline(),
        model_id="Tongyi-MAI/Z-Image",
    )
    result = model.generate(
        ImageGenerationRequest(
            prompt="fox",
            seed=42,
            steps=50,
            width=512,
            height=512,
            guidance=4.0,
            extra={
                "negative_prompt": "blurry",
                "cfg_truncation": 0.75,
            },
        )
    )
    assert model.variant == "base"
    assert calls["guidance"] == 4.0
    assert calls["negative_prompt"] == "blurry"
    assert calls["cfg_truncation"] == 0.75
    assert result.metadata["guidance_mode"] == "classifier-free"


def test_base_model_applies_variant_defaults() -> None:
    calls = {}

    class FakePipeline:
        config = ZImageConfig(
            default_steps=50,
            default_guidance=4.0,
            scheduler_shift=6.0,
            variant="base",
        )
        model_path = Path("/tmp/z-image")

        def generate_array(self, prompt: str, **kwargs):
            calls.update(prompt=prompt, **kwargs)
            return mx.zeros((16, 16, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = ZImageGenerationModel(
        pipeline=FakePipeline(),
        model_id="Tongyi-MAI/Z-Image",
    )
    result = model.generate(ImageGenerationRequest(prompt="fox"))
    assert calls["steps"] == 50
    assert calls["guidance"] == 4.0
    assert result.steps == 50
    assert result.guidance == 4.0


def test_base_model_preserves_explicit_generic_values() -> None:
    calls = {}

    class FakePipeline:
        config = ZImageConfig(
            default_steps=50,
            default_guidance=4.0,
            scheduler_shift=6.0,
            variant="base",
        )
        model_path = Path("/tmp/z-image")

        def generate_array(self, prompt: str, **kwargs):
            calls.update(prompt=prompt, **kwargs)
            return mx.zeros((16, 16, 3), dtype=mx.uint8)

        def count_prompt_tokens(self, prompt: str) -> int:
            return 1

    model = ZImageGenerationModel(
        pipeline=FakePipeline(),
        model_id="Tongyi-MAI/Z-Image",
    )
    result = model.generate(ImageGenerationRequest(prompt="fox", steps=4, guidance=1.0))
    assert calls["steps"] == 4
    assert calls["guidance"] == 1.0
    assert result.metadata["guidance_mode"] == "disabled"


def test_generation_evicts_components_before_reloading_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object.__new__(ZImagePipeline)
    pipeline.evict_text_encoder = True
    pipeline.transformer = object()
    pipeline.vae = object()
    pipeline.text_encoder = None
    reloaded = False

    def reload_encoder() -> None:
        nonlocal reloaded
        assert pipeline.transformer is None
        assert pipeline.vae is None
        reloaded = True

    monkeypatch.setattr(pipeline, "_reload_encoder", reload_encoder)
    monkeypatch.setattr(
        pipeline,
        "_encode_prompt",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("stop after reload")),
    )

    with pytest.raises(RuntimeError, match="stop after reload"):
        pipeline.generate_array("fox", steps=2, width=16, height=16)
    assert reloaded


# --- Tiny random-weight shape tests ---


def test_transformer_forward_shape() -> None:
    cfg = ZImageTransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        in_channels=16,
        text_embed_dim=32,
        num_hidden_layers=2,
        n_refiner_layers=1,
        n_context_refiner_layers=1,
        adaln_embed_dim=256,
        rope_sections=(4, 6, 6),
    )
    model = ZImageTransformer(cfg)

    class CaptureLength(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.length = 0

        def __call__(self, hidden: mx.array, *args, **kwargs) -> mx.array:
            self.length = hidden.shape[1]
            return hidden

    noise_refiner = CaptureLength()
    context_refiner = CaptureLength()
    unified_layer = CaptureLength()
    model.noise_refiner = [noise_refiner]
    model.context_refiner = [context_refiner]
    model.layers = [unified_layer]
    # Input: [B=1, C=16, F=1, H=4, W=4] (patch_size=2 → 2x2 grid)
    x = mx.random.normal((1, 16, 1, 4, 4))
    t = mx.array([0.5])
    cap = mx.random.normal((1, 8, 32))
    out = model(x, t, cap)
    mx.eval(out)
    assert out.shape == (1, 16, 1, 4, 4)
    assert noise_refiner.length == 32
    assert context_refiner.length == 32
    assert unified_layer.length == 64


def test_text_encoder_forward_shape() -> None:
    cfg = ZImageTextEncoderConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        head_dim=16,
    )
    model = ZImageTextEncoder(cfg)
    ids = mx.array([[1, 2, 3, 4, 5]])
    out = model(ids)
    mx.eval(out)
    assert out.shape == (1, 5, 64)


def test_vae_decoder_shape() -> None:
    cfg = ZImageVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=1,
    )
    vae = ZImageVAE(cfg)
    # Latent: [B=1, H=4, W=4, C=4]
    z = mx.random.normal((1, 4, 4, 4))
    out = vae.decode(z)
    mx.eval(out)
    # After 2 up_blocks with upsample (first block upsamples, second doesn't)
    # input 4×4 → 8×8 after first upsample → stays 8×8 (last block no upsample)
    assert out.shape[0] == 1
    assert out.shape[-1] == 3  # 3 output channels


def test_rejects_classifier_free_guidance() -> None:
    model = object.__new__(ZImageGenerationModel)
    model.pipeline = SimpleNamespace(
        config=ZImageConfig(),
    )
    with pytest.raises(ValueError, match="does not support classifier-free guidance"):
        model.generate(ImageGenerationRequest(prompt="test", guidance=2.0))


def test_config_loads_original_metadata(tmp_path: Path) -> None:
    configs = {
        "transformer/config.json": {
            "dim": 3840,
            "n_heads": 30,
            "cap_feat_dim": 2560,
        },
        "text_encoder/config.json": {
            "hidden_size": 2560,
            "num_attention_heads": 32,
        },
        "vae/config.json": {},
        "scheduler/scheduler_config.json": {"shift": 3.0},
    }
    for relative, content in configs.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    config = ZImageConfig.from_model_path(tmp_path)
    assert config.transformer.hidden_size == 3840
    assert config.transformer.num_attention_heads == 30
    assert config.transformer.text_embed_dim == 2560
    assert config.text_encoder.hidden_size == 2560
    assert config.text_encoder.num_attention_heads == 32
    assert config.scheduler_shift == 3.0
    assert config.variant == "turbo"
    assert config.default_steps == 9
    assert config.default_guidance == 0.0


def test_config_detects_base_variant(tmp_path: Path) -> None:
    configs = {
        "transformer/config.json": {},
        "text_encoder/config.json": {},
        "vae/config.json": {},
        "scheduler/scheduler_config.json": {"shift": 6.0},
    }
    for relative, content in configs.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    config = ZImageConfig.from_model_path(tmp_path)
    assert config.variant == "base"
    assert config.default_steps == 50
    assert config.default_guidance == 4.0


def test_transformer_config_allows_distinct_refiner_depths() -> None:
    config = ZImageTransformerConfig.from_dict(
        {"n_refiner_layers": 2, "n_context_refiner_layers": 4}
    )
    assert config.n_refiner_layers == 2
    assert config.n_context_refiner_layers == 4


def test_sanitize_transformer_weights() -> None:
    weights = {
        "all_final_layer.2-1.linear.weight": mx.zeros((3,)),
        "all_final_layer.2-1.adaLN_modulation.1.weight": mx.zeros((3,)),
        "all_x_embedder.2-1.weight": mx.zeros((3,)),
        "layers.0.attention.to_q.weight": mx.zeros((3,)),
        "t_embedder.mlp.0.weight": mx.zeros((3,)),
    }
    sanitized = sanitize_transformer_weights(weights)
    assert "final_layer.linear.weight" in sanitized
    assert "final_layer.adaLN_modulation.0.weight" in sanitized
    assert "x_embedder.weight" in sanitized
    assert "layers.0.attention.to_q.weight" in sanitized
    assert "t_embedder.linear1.weight" in sanitized


def test_sanitize_source_text_encoder_weights() -> None:
    sanitized = sanitize_text_encoder_weights(
        {
            "model.embed_tokens.weight": mx.zeros((2, 2)),
            "model.rotary_emb.inv_freq": mx.zeros((2,)),
        }
    )
    assert set(sanitized) == {"embed_tokens.weight"}


def test_sanitize_source_vae_weights() -> None:
    conv = mx.zeros((8, 4, 3, 3))
    sanitized = sanitize_vae_weights(
        {
            "encoder.conv_in.weight": conv,
            "decoder.conv_norm_out.weight": mx.zeros((8,)),
        }
    )
    assert sanitized["encoder.conv_in.weight"].shape == (8, 3, 3, 4)
    assert "decoder.conv_norm_out.weight" in sanitized

    native = sanitize_vae_weights({"encoder.conv_in.conv2d.weight": conv})
    assert set(native) == {"encoder.conv_in.weight"}

    converted = sanitize_vae_weights(
        sanitized,
        source_layout=False,
    )
    assert mx.array_equal(
        converted["encoder.conv_in.weight"],
        sanitized["encoder.conv_in.weight"],
    )


def test_conversion_preserves_native_vae_layout(tmp_path: Path) -> None:
    native = mx.zeros((8, 3, 3, 4))
    vae_path = tmp_path / "vae"
    vae_path.mkdir()
    (vae_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"mlx_vlm_format": "z_image"}})
    )

    converted = _sanitize_vae_for_conversion(
        vae_path,
        {"encoder.conv_in.weight": native},
    )

    assert converted["encoder.conv_in.weight"].shape == native.shape
    assert mx.array_equal(converted["encoder.conv_in.weight"], native)


@pytest.mark.parametrize("width,height", [(0, 512), (513, 512), (512, 513)])
def test_rejects_invalid_image_dimensions(width: int, height: int) -> None:
    pipeline = object.__new__(ZImagePipeline)
    with pytest.raises(ValueError, match="positive multiple of 16"):
        pipeline.generate_array("prompt", width=width, height=height)


@pytest.mark.parametrize(
    "mode,bits,group_size",
    [
        ("mxfp4", 4, 32),
        ("mxfp8", 8, 32),
        ("nvfp4", 4, 16),
        ("affine", 4, 64),
    ],
)
def test_native_quantization_metadata_is_supported(
    mode: str, bits: int, group_size: int
) -> None:
    from mlx.utils import tree_flatten

    from mlx_vlm.models.z_image.weights import _apply_weights

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

    quantized = TinyModel()
    nn.quantize(
        quantized,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    weights = dict(tree_flatten(quantized.parameters()))
    loaded = _apply_weights(
        TinyModel(),
        weights,
        {
            "quantization_mode": mode,
            "quantization_level": str(bits),
            "quantization_group_size": str(group_size),
        },
    )
    assert loaded.quantization_config == {
        "group_size": group_size,
        "bits": bits,
        "mode": mode,
    }
    if mode == "affine":
        assert hasattr(loaded.proj, "biases")


def test_saved_affine_metadata_marks_native_layout(tmp_path: Path) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

    model = TinyModel()
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    _save_component(
        tmp_path,
        "transformer",
        model,
        {
            "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine",
            }
        },
    )
    index = json.loads(
        (tmp_path / "transformer" / "model.safetensors.index.json").read_text()
    )
    assert index["metadata"]["mlx_vlm_format"] == "z_image"
    assert index["metadata"]["quantization_group_size"] == "64"
    assert index["metadata"]["quantization_level"] == "4"
    assert index["metadata"]["quantization_mode"] == "affine"
