from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from PIL import Image

import mlx_vlm.models.ernie_image.convert as ernie_convert
from mlx_vlm.generate.edit_image import (
    ImageEditRequest,
    image_edit_model_class,
    is_image_edit_model,
)
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    generate_image,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.models.ernie_image.config import (
    ErnieImageTransformerConfig,
    get_variant,
    validate_dimensions,
    variant_from_local_path,
)
from mlx_vlm.models.ernie_image.convert import (
    _quantization_parameters,
    _quantize_component,
    _source_layout,
    _vae_checkpoint_has_encoder,
    _write_missing_configs,
    is_ernie_image_checkpoint,
)
from mlx_vlm.models.ernie_image.download import validate_model_layout
from mlx_vlm.models.ernie_image.model import (
    ErnieImageEditModel,
    ErnieImageGenerationModel,
)
from mlx_vlm.models.ernie_image.pipeline import (
    ErnieImagePipeline,
    ErnieImageRuntimeConfig,
    _load_edit_image,
    _pad_text,
)
from mlx_vlm.models.ernie_image.scheduler import ErnieImageFlowMatchScheduler
from mlx_vlm.models.ernie_image.text_encoder import (
    ErnieImageTextConfig,
    ErnieImageTextEncoder,
)
from mlx_vlm.models.ernie_image.transformer import (
    ErnieImageTransformer,
    rope_frequencies,
    rotate_half,
    timestep_embedding,
)
from mlx_vlm.models.ernie_image.weights import (
    _require_vae_encoder_weights,
    apply_weights,
    match_conv_layout,
    sanitize_text_encoder_weights,
    sanitize_transformer_weights,
)
from mlx_vlm.models.flux2.vae import Flux2VAE

convert_module = importlib.import_module("mlx_vlm.convert")


def _write_layout(root: Path, *, turbo: bool = True) -> None:
    for relative in (
        "transformer/0.safetensors",
        "text_encoder/0.safetensors",
        "vae/0.safetensors",
        "tokenizer/tokenizer.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "ErnieImagePipeline"})
    )
    (root / "mlx_ernie_image.json").write_text(
        json.dumps(
            {
                "model_type": "ernie_image",
                "variant": f"ernie-image{'-turbo' if turbo else ''}",
            }
        )
    )


@pytest.mark.parametrize(
    "model_id,name,steps,guidance",
    [
        ("baidu/ERNIE-Image", "ernie-image", 50, 4.0),
        ("baidu/ERNIE-Image-Turbo", "ernie-image-turbo", 8, 1.0),
    ],
)
def test_ernie_variants(model_id: str, name: str, steps: int, guidance: float) -> None:
    variant = get_variant(model_id)
    assert variant.name == name
    assert variant.default_steps == steps
    assert variant.default_guidance == guidance


def test_ernie_local_variant_uses_native_metadata(tmp_path: Path) -> None:
    _write_layout(tmp_path, turbo=False)
    assert variant_from_local_path(tmp_path).name == "ernie-image"


def test_ernie_dispatches_ids_metadata_and_mflux_indexes(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    assert (
        image_generation_model_class(tmp_path.as_posix()) is ErnieImageGenerationModel
    )
    assert (
        image_generation_model_class("baidu/ERNIE-Image-Turbo")
        is ErnieImageGenerationModel
    )
    assert is_image_generation_model("ernie-image")
    assert image_edit_model_class(tmp_path.as_posix()) is ErnieImageEditModel
    assert is_image_edit_model("baidu/ERNIE-Image-Turbo")

    (tmp_path / "model_index.json").unlink()
    (tmp_path / "mlx_ernie_image.json").unlink()
    index = {
        "weight_map": {
            "adaln_modulation.weight": "0.safetensors",
            "final_norm.linear.weight": "0.safetensors",
            "layers.0.adaLN_sa_ln.weight": "0.safetensors",
        }
    }
    (tmp_path / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps(index)
    )
    assert (
        image_generation_model_class(tmp_path.as_posix()) is ErnieImageGenerationModel
    )


def test_ernie_layout_accepts_mflux_checkpoint_without_configs(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    assert validate_model_layout(tmp_path) == tmp_path


def test_ernie_layout_requires_complete_prompt_enhancer(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    (tmp_path / "pe" / "model.safetensors").parent.mkdir()
    (tmp_path / "pe" / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="pe_tokenizer"):
        validate_model_layout(tmp_path)


def test_transformer_config_parses_official_fields() -> None:
    config = ErnieImageTransformerConfig.from_dict(
        {
            "_class_name": "ErnieImageTransformer2DModel",
            "hidden_size": 32,
            "ffn_hidden_size": 64,
            "num_attention_heads": 4,
            "rope_axes_dim": [2, 2, 4],
            "unknown": 1,
        }
    )
    assert config.head_dim == 8
    assert config.rope_axes_dim == (2, 2, 4)


@pytest.mark.parametrize("width,height", [(15, 16), (1025, 1024), (1024, 1000)])
def test_ernie_dimensions_must_be_positive_multiples_of_16(
    width: int, height: int
) -> None:
    with pytest.raises(ValueError):
        validate_dimensions(width=width, height=height)


def test_ernie_scheduler_matches_official_static_shift() -> None:
    scheduler = ErnieImageFlowMatchScheduler(num_inference_steps=8)
    expected = np.array(
        [
            1.0,
            0.96551724,
            0.92307692,
            0.86956522,
            0.8,
            0.70588235,
            0.57142857,
            0.36363636,
            0.0,
        ]
    )
    np.testing.assert_allclose(np.array(scheduler.sigmas), expected, rtol=1e-6)
    assert scheduler.timesteps.shape == (8,)


def test_timestep_embedding_uses_sin_then_cos() -> None:
    result = timestep_embedding(mx.array([0.0]), 4)
    np.testing.assert_allclose(np.array(result), [[0.0, 0.0, 1.0, 1.0]])


def test_ernie_rope_matches_reference_hybrid_convention() -> None:
    ids = np.array([[[3.0, 1.0, 2.0], [4.0, 2.0, 1.0]]], dtype=np.float32)
    axes = (2, 2, 4)
    angles = []
    for axis, dim in enumerate(axes):
        omega = 1.0 / (256.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        angles.append(ids[..., axis, None] * omega)
    angles = np.concatenate(angles, axis=-1)
    angles = np.stack([angles, angles], axis=-1).reshape(1, 2, 1, 8)

    cos, sin = rope_frequencies(mx.array(ids), axes_dim=axes, theta=256.0)
    np.testing.assert_allclose(
        np.array(cos.transpose(0, 2, 1, 3)), np.cos(angles), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.array(sin.transpose(0, 2, 1, 3)), np.sin(angles), rtol=1e-6
    )

    values = mx.arange(16, dtype=mx.float32).reshape(1, 1, 2, 8)
    expected_rotated = np.concatenate(
        [-np.array(values)[..., 4:], np.array(values)[..., :4]], axis=-1
    )
    np.testing.assert_array_equal(np.array(rotate_half(values)), expected_rotated)


def test_tiny_ernie_transformer_forward() -> None:
    config = ErnieImageTransformerConfig(
        hidden_size=32,
        ffn_hidden_size=64,
        in_channels=8,
        out_channels=8,
        num_layers=2,
        num_attention_heads=4,
        rope_axes_dim=(2, 2, 4),
        text_in_dim=16,
    )
    transformer = ErnieImageTransformer(config)
    output = transformer(
        mx.zeros((2, 8, 2, 2), dtype=mx.bfloat16),
        timestep=mx.array([1000.0, 500.0], dtype=mx.bfloat16),
        text_hidden_states=mx.zeros((2, 3, 16), dtype=mx.bfloat16),
        text_lengths=mx.array([1, 3]),
    )
    mx.eval(output)
    assert output.shape == (2, 8, 2, 2)
    assert bool(mx.all(mx.isfinite(output)))


def test_conditioning_skips_last_text_block_and_final_norm() -> None:
    class FakeEmbedding(nn.Module):
        def __call__(self, input_ids):
            return mx.zeros((*input_ids.shape, 2))

    class AddOne(nn.Module):
        def __call__(self, hidden_states, mask, cache=None):  # noqa: ARG002
            return hidden_states + 1

    class TimesTen(nn.Module):
        def __call__(self, hidden_states):
            return hidden_states * 10

    encoder = ErnieImageTextEncoder(
        ErnieImageTextConfig(
            vocab_size=4,
            hidden_size=2,
            intermediate_size=4,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
            rope_parameters=None,
        )
    )
    encoder.embed_tokens = FakeEmbedding()
    encoder.layers = [AddOne(), AddOne(), AddOne()]
    encoder.norm = TimesTen()
    ids = mx.array([[1, 2]])

    np.testing.assert_array_equal(np.array(encoder(ids)), np.full((1, 2, 2), 2))
    np.testing.assert_array_equal(
        np.array(encoder(ids, normalize=True)), np.full((1, 2, 2), 30)
    )


def test_pad_text_preserves_cfg_order_and_lengths() -> None:
    negative = mx.ones((1, 1, 2))
    positive = mx.full((1, 3, 2), 2)
    padded, lengths = _pad_text([negative, positive])
    assert tuple(padded.shape) == (2, 3, 2)
    np.testing.assert_array_equal(np.array(lengths), [1, 3])
    np.testing.assert_array_equal(np.array(padded[0, 1:]), np.zeros((2, 2)))


def test_ernie_weight_sanitizers_and_layouts() -> None:
    native = mx.zeros((8, 1, 1, 4))
    source = native.transpose(0, 3, 1, 2)
    sanitized = sanitize_transformer_weights(
        {
            "adaLN_modulation.1.weight": mx.zeros((6, 4)),
            "layers.0.self_attention.to_out.0.weight": mx.zeros((4, 4)),
            "x_embedder.proj.weight": source,
        },
        target_shapes={
            "adaln_modulation.weight": (6, 4),
            "layers.0.self_attention.to_out.weight": (4, 4),
            "x_embedder.proj.weight": tuple(native.shape),
        },
    )
    assert "adaln_modulation.weight" in sanitized
    assert "layers.0.self_attention.to_out.weight" in sanitized
    assert sanitized["x_embedder.proj.weight"].shape == native.shape
    assert (
        match_conv_layout(
            native, target_shape=tuple(native.shape), key="conv.weight"
        ).shape
        == native.shape
    )

    text = sanitize_text_encoder_weights(
        {
            "language_model.model.embed_tokens.weight": mx.zeros((4, 2)),
            "language_model.model.rotary_emb.inv_freq": mx.zeros((1,)),
            "vision_tower.weight": mx.zeros((1,)),
        }
    )
    assert set(text) == {"embed_tokens.weight"}


def test_ambiguous_rgb_conv_uses_source_layout_metadata() -> None:
    source = mx.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
    expected = source.transpose(0, 2, 3, 1)
    converted = match_conv_layout(
        source,
        target_shape=(2, 3, 3, 3),
        key="encoder.conv_in.weight",
        source_layout="pytorch_nchw",
    )
    unchanged = match_conv_layout(
        expected,
        target_shape=(2, 3, 3, 3),
        key="encoder.conv_in.weight",
        source_layout="mlx_nhwc",
    )
    np.testing.assert_array_equal(np.array(converted), np.array(expected))
    np.testing.assert_array_equal(np.array(unchanged), np.array(expected))


@pytest.mark.parametrize(
    "mode,group_size,bits",
    [
        ("affine", 32, 4),
        ("mxfp4", 32, 4),
        ("nvfp4", 16, 4),
        ("mxfp8", 32, 8),
    ],
)
def test_ernie_strict_quantized_loading_modes(
    mode: str, group_size: int, bits: int
) -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 16, bias=False)

    dense = mx.arange(16 * 64, dtype=mx.float32).reshape(16, 64)
    arrays = mx.quantize(dense, group_size=group_size, bits=bits, mode=mode)
    weights = {"proj.weight": arrays[0], "proj.scales": arrays[1]}
    if len(arrays) == 3:
        weights["proj.biases"] = arrays[2]
    model = apply_weights(
        TinyModel(),
        weights,
        {
            "quantization_mode": mode,
            "quantization_group_size": str(group_size),
            "quantization_level": str(bits),
        },
    )
    assert isinstance(model.proj, nn.QuantizedLinear)
    assert model.quantization_config == {
        "mode": mode,
        "group_size": group_size,
        "bits": bits,
    }


def test_ernie_rejects_incompatible_native_quantization_options() -> None:
    with pytest.raises(ValueError, match="requires"):
        _quantization_parameters("mxfp8", 64, 8)


def test_conversion_quantizes_compatible_vae_attention() -> None:
    vae = Flux2VAE(
        decoder_block_out_channels=(32, 32),
        include_encoder=True,
        encoder_block_out_channels=(32, 32),
    )
    _quantize_component(
        vae,
        {"mode": "mxfp8", "group_size": 32, "bits": 8},
        lambda path, module: hasattr(module, "to_quantized"),
    )
    assert isinstance(vae.decoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.encoder.mid_block.attentions[0].to_q, nn.QuantizedLinear)
    assert isinstance(vae.decoder.conv_in, nn.Conv2d)


class FakePipeline:
    def __init__(self, variant: str) -> None:
        self.variant = get_variant(variant)
        self.model_path = Path("/tmp/ernie")
        self.runtime_config = ErnieImageRuntimeConfig(use_prompt_enhancer=False)
        self.calls = []
        self.quantization_config = None
        self.last_revised_prompt = None

    def generate_array(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return mx.zeros((16, 16, 3), dtype=mx.uint8)

    def edit_array(self, prompt, image, **kwargs):
        self.calls.append((prompt, image, kwargs))
        return mx.zeros((16, 32, 3), dtype=mx.uint8)

    def count_prompt_tokens(self, prompt):  # noqa: ARG002
        return 3

    def _should_enhance_prompt(self):
        return False


@pytest.mark.parametrize(
    "variant,steps,guidance,cfg",
    [
        ("ernie-image", 50, 4.0, True),
        ("ernie-image-turbo", 8, 1.0, False),
    ],
)
def test_model_request_defaults_follow_variant(
    variant: str, steps: int, guidance: float, cfg: bool
) -> None:
    pipeline = FakePipeline(variant)
    model = ErnieImageGenerationModel(pipeline=pipeline, model_id=variant)
    result = generate_image(
        model,
        ImageGenerationRequest(
            prompt="a lighthouse",
            seed=7,
            extra={"negative_prompt": "fog"},
        ),
    )
    assert result.steps == steps
    assert result.guidance == guidance
    assert result.metadata["classifier_free_guidance"] is cfg
    assert result.width == result.height == 1024
    assert pipeline.calls[0][1]["negative_prompt"] == "fog"


def test_generation_request_defaults_remain_valid_for_edit_bridge() -> None:
    class FakeEditModel:
        def edit(self, request):
            assert request.steps == 4
            assert request.guidance == 1.0
            return SimpleNamespace(path=None)

    generate_image(
        FakeEditModel(),
        ImageGenerationRequest(prompt="edit"),
        task="edit",
        image_paths=("reference.png",),
    )


def test_ernie_edit_model_forwards_img2img_strength() -> None:
    pipeline = FakePipeline("ernie-image-turbo")
    model = ErnieImageEditModel(pipeline=pipeline, model_id="ernie")
    result = model.edit(
        ImageEditRequest(
            prompt="make it a convertible",
            image_paths=("source.png",),
            seed=3,
            steps=8,
            guidance=1.0,
            extra={"image_strength": 0.55},
        )
    )
    assert result.width == 32
    assert result.height == 16
    assert result.metadata["image_strength"] == 0.55
    assert result.metadata["native_instruction_edit"] is False
    assert pipeline.calls[0][2]["image_strength"] == 0.55


def test_ernie_base_edit_uses_variant_defaults() -> None:
    pipeline = FakePipeline("ernie-image")
    model = ErnieImageEditModel(pipeline=pipeline, model_id="ernie")
    result = model.edit(
        ImageEditRequest(
            prompt="make it a convertible",
            image_paths=("source.png",),
            seed=3,
        )
    )
    assert result.steps == 50
    assert result.guidance == 4.0


class FakeTransformer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, hidden_states, **kwargs):
        self.calls.append((hidden_states.shape, kwargs["text_lengths"]))
        batch = hidden_states.shape[0]
        values = mx.arange(batch, dtype=hidden_states.dtype).reshape(batch, 1, 1, 1)
        return mx.broadcast_to(values, hidden_states.shape)


class FakeVAE:
    quantization_config = None

    def __init__(self) -> None:
        self.encoder = object()
        self.bn = SimpleNamespace(
            running_mean=mx.zeros((128,)),
            running_var=mx.ones((128,)),
        )

    def encode(self, pixels):
        return mx.zeros(
            (pixels.shape[0], 32, pixels.shape[2] // 8, pixels.shape[3] // 8),
            dtype=mx.bfloat16,
        )

    def decode_packed_latents(self, latents):
        return mx.zeros((1, 3, latents.shape[2] * 16, latents.shape[3] * 16))


def _fake_runtime_pipeline(variant: str, *, evict: bool = False) -> ErnieImagePipeline:
    pipeline = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipeline.variant = get_variant(variant)
    pipeline.model_path = Path("/tmp/ernie")
    pipeline.runtime_config = ErnieImageRuntimeConfig(
        evict_text_encoder=False,
        evict_transformer=evict,
        use_prompt_enhancer=False,
    )
    pipeline.tokenizer = SimpleNamespace(count_tokens=lambda prompt: len(prompt))
    pipeline.text_encoder = None
    pipeline.prompt_enhancer = None
    pipeline.component_quantization = {}
    pipeline.transformer = FakeTransformer()
    pipeline.vae = FakeVAE()
    pipeline.prompt_cache = {}
    pipeline._encode_prompts = lambda prompts: (
        mx.zeros((len(prompts), 3, 4), dtype=mx.bfloat16),
        mx.array([1, 3] if len(prompts) == 2 else [3]),
    )
    pipeline._ensure_components = lambda **kwargs: None
    return pipeline


def test_base_cfg_batches_unconditional_before_conditional() -> None:
    pipeline = _fake_runtime_pipeline("ernie-image")
    pipeline.generate_array("prompt", seed=1, steps=1, width=16, height=16)
    shape, lengths = pipeline.transformer.calls[0]
    assert shape[0] == 2
    np.testing.assert_array_equal(np.array(lengths), [1, 3])


def test_turbo_skips_cfg_and_evicts_large_components() -> None:
    pipeline = _fake_runtime_pipeline("ernie-image-turbo", evict=True)
    pipeline.generate_array("prompt", seed=1, steps=1, width=16, height=16)
    assert pipeline.transformer is None
    assert pipeline.vae is None


def test_img2img_uses_strength_to_select_denoising_steps(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (32, 16), color="navy").save(image_path)
    pipeline = _fake_runtime_pipeline("ernie-image-turbo")
    result = pipeline.edit_array(
        "make it a convertible",
        image_path,
        seed=1,
        steps=8,
        width=32,
        height=16,
        guidance=1.0,
        image_strength=0.5,
    )
    assert result.shape == (16, 32, 3)
    assert len(pipeline.transformer.calls) == 4


def test_edit_image_rounds_source_size_to_model_grid(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGBA", (1160, 890), color="navy").save(image_path)
    pixels, width, height = _load_edit_image(image_path, width=None, height=None)
    assert (width, height) == (1152, 880)
    assert pixels.shape == (1, 3, 880, 1152)


def test_edit_image_caps_large_source_to_one_megapixel(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (4032, 3024), color="navy").save(image_path)
    pixels, width, height = _load_edit_image(image_path, width=None, height=None)
    assert (width, height) == (1168, 880)
    assert pixels.shape == (1, 3, 880, 1168)


@pytest.mark.parametrize(
    "size,expected",
    [
        ((300, 200), (384, 256)),
        ((4000, 250), (2048, 128)),
    ],
)
def test_edit_auto_size_preserves_aspect_ratio(
    tmp_path: Path,
    size: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", size, color="navy").save(image_path)
    _, width, height = _load_edit_image(image_path, width=None, height=None)
    assert (width, height) == expected


def test_img2img_rejects_decoder_only_converted_vae() -> None:
    with pytest.raises(ValueError, match="VAE encoder weights"):
        _require_vae_encoder_weights(
            {"decoder.conv_in.weight": mx.zeros((1,))},
            target_shapes={
                "encoder.conv_in.weight": (2, 3, 3, 3),
                "quant_conv.weight": (4, 1, 1, 4),
                "decoder.conv_in.weight": (2, 3, 3, 4),
            },
        )


def test_conversion_detects_decoder_only_vae_index(tmp_path: Path) -> None:
    vae = tmp_path / "vae"
    vae.mkdir()
    index = vae / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "decoder.conv_in.weight": "model.safetensors",
                    "post_quant_conv.weight": "model.safetensors",
                }
            }
        )
    )
    assert not _vae_checkpoint_has_encoder(tmp_path)
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "encoder.conv_in.weight": "model.safetensors",
                    "quant_conv.weight": "model.safetensors",
                }
            }
        )
    )
    assert _vae_checkpoint_has_encoder(tmp_path)


def test_prompt_enhancement_auto_detects_optional_components(tmp_path: Path) -> None:
    pipeline = _fake_runtime_pipeline("ernie-image-turbo")
    pipeline.model_path = tmp_path
    assert not pipeline._should_enhance_prompt()
    (tmp_path / "pe").mkdir()
    (tmp_path / "pe" / "model.safetensors").write_bytes(b"x")
    (tmp_path / "pe_tokenizer").mkdir()
    (tmp_path / "pe_tokenizer" / "tokenizer.json").write_text("{}")
    pipeline.runtime_config = ErnieImageRuntimeConfig(use_prompt_enhancer=None)
    assert pipeline._should_enhance_prompt()


def test_conversion_detection_and_layout_metadata(tmp_path: Path) -> None:
    _write_layout(tmp_path)
    assert is_ernie_image_checkpoint(tmp_path)
    assert _source_layout(tmp_path) == "mlx_nhwc"
    output = tmp_path / "output"
    for component in ("transformer", "text_encoder", "vae"):
        (output / component).mkdir(parents=True, exist_ok=True)
    _write_missing_configs(output)
    assert (
        json.loads((output / "transformer" / "config.json").read_text())["_class_name"]
        == "ErnieImageTransformer2DModel"
    )
    assert (
        json.loads((output / "model_index.json").read_text())["_class_name"]
        == "ErnieImagePipeline"
    )


def test_main_convert_routes_ernie_before_vlm_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_layout(tmp_path)
    calls = {}
    monkeypatch.setattr(
        convert_module, "get_model_path", lambda *args, **kwargs: tmp_path
    )
    monkeypatch.setattr(
        convert_module,
        "fetch_from_hub",
        lambda *args, **kwargs: pytest.fail("VLM loader should not be called"),
    )

    def fake_convert(model_path, output_path, **kwargs):
        calls.update({"model_path": model_path, "output_path": output_path, **kwargs})
        return Path(output_path)

    monkeypatch.setattr(ernie_convert, "convert_ernie_image", fake_convert)
    output = tmp_path.parent / "converted"
    result = convert_module.convert(
        "baidu/ERNIE-Image-Turbo",
        str(output),
        quantize=True,
        q_mode="mxfp8",
        q_group_size=None,
        q_bits=None,
    )
    assert result == output
    assert calls["model_path"] == tmp_path
    assert calls["q_mode"] == "mxfp8"
