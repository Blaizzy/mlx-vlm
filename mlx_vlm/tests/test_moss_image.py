"""MOSS-Image numerical, checkpoint, and public image API contracts (no downloads)."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from numpy.testing import assert_allclose, assert_array_equal

from mlx_vlm.generate import ImageSamplingDefaults, resolve_image_defaults
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
    load_image_generation_model,
)
from mlx_vlm.models.flux2.vae import Flux2VAE
from mlx_vlm.models.moss_image.config import MOSSImageTransformerConfig
from mlx_vlm.models.moss_image.convert import convert_moss_image, model_card
from mlx_vlm.models.moss_image.model import MOSSImageGenerationModel
from mlx_vlm.models.moss_image.pipeline import MOSSImagePipeline
from mlx_vlm.models.moss_image.scheduler import MOSSImageScheduler
from mlx_vlm.models.moss_image.transformer import MOSSImageTransformer, rotary_positions
from mlx_vlm.models.moss_image.weights import (
    load_text_encoder,
    load_transformer,
    load_vae,
    sanitize_text_encoder_weights,
)
from mlx_vlm.models.qwen3_5.config import TextConfig
from mlx_vlm.models.qwen3_5.language import Qwen3_5Model


def tiny_transformer():
    return MOSSImageTransformer(
        MOSSImageTransformerConfig(
            dim=32,
            ffn_dim=48,
            num_heads=2,
            num_layers=2,
            text_dim=16,
            freq_dim=8,
            latent_channels=4,
        )
    )


def test_metadata_discovery_and_loading(tmp_path, monkeypatch):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "MOSSImagePipeline"})
    )
    image_module = importlib.import_module("mlx_vlm.generate.image")
    monkeypatch.setattr(
        image_module,
        "get_model_path",
        Mock(side_effect=AssertionError("Unexpected download")),
    )
    assert image_generation_model_class(str(tmp_path)) is MOSSImageGenerationModel
    assert is_image_generation_model(str(tmp_path))
    assert not MOSSImageGenerationModel.supports_model("org/unrelated")
    for model in (
        str(tmp_path),
        "OpenMOSS-Team/MOSS-Image-Base",
        "OpenMOSS-Team/MOSS-Image-SFT",
        "moss-image",
    ):
        assert image_generation_model_class(model) is MOSSImageGenerationModel
        assert resolve_image_defaults(model) == ImageSamplingDefaults(50, 3.0)
    module = importlib.import_module("mlx_vlm.models.moss_image.model")
    pipeline = Mock()
    monkeypatch.setattr(module, "MOSSImagePipeline", pipeline)
    loaded = load_image_generation_model(str(tmp_path))
    assert loaded.pipeline is pipeline.return_value
    pipeline.assert_called_once_with(tmp_path, evict_text_encoder=True)


@pytest.mark.parametrize("variant", ["Base", "SFT"])
def test_hub_checkpoint_loading(variant, tmp_path, monkeypatch):
    image_module = importlib.import_module("mlx_vlm.generate.image")
    model_module = importlib.import_module("mlx_vlm.models.moss_image.model")
    download = Mock(return_value=tmp_path)
    monkeypatch.setattr(image_module, "get_model_path", download)
    pipeline = Mock()
    monkeypatch.setattr(model_module, "MOSSImagePipeline", pipeline)
    model_id = f"OpenMOSS-Team/MOSS-Image-{variant}"
    loaded = load_image_generation_model(model_id, revision="pinned-revision")
    assert loaded.model_id == model_id
    assert loaded.default_sampling == ImageSamplingDefaults(50, 3.0)
    assert is_image_generation_model(model_id)
    pipeline.assert_called_once_with(tmp_path, evict_text_encoder=True)
    assert download.call_args.args == (model_id,)
    assert download.call_args.kwargs["revision"] == "pinned-revision"


@pytest.mark.parametrize(
    "source_id", [None, "OpenMOSS-Team/MOSS-Image-Base", "OpenMOSS-Team/MOSS-Image-SFT"]
)
def test_model_card_preserves_checkpoint_identity(source_id):
    card = model_card(source_id)
    if source_id is None:
        assert "base_model:" not in card
        assert "MOSS-Image-SFT" not in card
        assert "local MOSS-Image checkpoint" in card
    else:
        assert f"base_model: {source_id}\n" in card
        assert f"https://huggingface.co/{source_id}" in card


def test_wrapper_defaults_and_generation_options():
    pipeline = SimpleNamespace(
        generate_array=Mock(return_value=mx.zeros((16, 32, 3), dtype=mx.uint8)),
        count_prompt_tokens=lambda _: 7,
        model_path=Path("/model"),
    )
    model = MOSSImageGenerationModel(pipeline, "moss-image")
    result = model.generate(
        ImageGenerationRequest(
            "cat",
            width=32,
            height=16,
            seed=5,
            extra={"uncond_skip_layers": [], "negative_prompt": "blurry"},
        )
    )
    assert (result.steps, result.guidance, result.prompt_tokens) == (50, 3.0, 7)
    assert result.to_pil().size == (32, 16)
    pipeline.generate_array.assert_called_once_with(
        "cat",
        seed=5,
        steps=50,
        width=32,
        height=16,
        guidance=3.0,
        uncond_skip_layers=[],
        negative_prompt="blurry",
    )


def test_rope_centering_text_offset_and_float32():
    rope = np.array(rotary_positions(2, 3, 2, 16))
    image = [[0, 2, h - 0.5, w - 1] for h in range(2) for w in range(3)]
    text = [[0, t, 0, 0] for t in range(2)]
    phases = np.array(image + text, dtype=np.float32)[..., None] * np.array(
        [1, 0.01], dtype=np.float32
    )
    expected = np.concatenate([np.sin(phases), np.cos(phases)], axis=-2).reshape(
        1, 8, 16
    )
    assert rope.dtype == np.float32
    assert_allclose(rope, expected, atol=1e-7)


def test_transformer_padding_is_removed_before_text_refinement():
    mx.random.seed(12)
    model = tiny_transformer()
    image = mx.random.normal((2, 4, 2, 3))
    text = mx.random.normal((2, 4, 16))
    mask = mx.array([[1, 0, 1, 0], [0, 1, 1, 1]])
    output = model(image, mx.array([500, 250]), text, mask)
    assert output.shape == image.shape
    for i, indices in enumerate(([0, 2], [1, 2, 3])):
        single = model(
            image[i : i + 1], [500, 250][i], text[i : i + 1, mx.array(indices)]
        )
        assert_allclose(np.array(output[i : i + 1]), np.array(single), atol=2e-5)
    altered = mx.where(mask[..., None], text, mx.full(text.shape, 1e6))
    assert_allclose(
        np.array(output), np.array(model(image, [500, 250], altered, mask)), atol=1e-6
    )
    assert not np.allclose(
        np.array(output),
        np.array(model(image, [500, 250], text, mask, skip_layers=[0])),
    )
    with pytest.raises(ValueError, match="at least one"):
        model(image, 1, text, mx.zeros(mask.shape))
    with pytest.raises(ValueError, match="integer block"):
        model(image, 1, text, skip_layers=[2])


def test_scheduler_snaps_to_training_grid_and_updates_in_float32():
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(50)
    timesteps = np.array(scheduler.timesteps)
    assert len(timesteps) == 50
    assert timesteps[0] == 1000
    assert timesteps[-1] == 24
    assert_allclose(timesteps, np.round(timesteps), atol=1e-4)
    assert np.all(np.diff(timesteps) < 0)
    output = scheduler.step(
        mx.ones((1,), dtype=mx.bfloat16), 0, mx.ones((1,), dtype=mx.bfloat16)
    )
    assert output.dtype == mx.float32
    assert_allclose(np.array(output), [timesteps[1] / 1000], atol=1e-7)
    scheduler.set_timesteps(2000)
    assert np.all(np.diff(np.array(scheduler.timesteps)) < 0)
    assert len(scheduler.timesteps) <= 1000
    scheduler.set_timesteps(1)
    assert_array_equal(np.array(scheduler.sigmas), [1, 0])


def test_text_encoder_upstream_layout_and_offset_norms():
    weights = sanitize_text_encoder_weights(
        {
            "visual.patch_embed.weight": mx.ones((2, 2)),
            "language_model.layers.0.linear_attn.conv1d.weight": mx.ones((16, 1, 4)),
            "language_model.layers.0.input_layernorm.weight": mx.zeros(8),
            "language_model.layers.0.linear_attn.norm.weight": mx.full((8,), 3),
            "language_model.layers.3.self_attn.q_norm.weight": mx.zeros(8),
            "language_model.norm.weight": mx.zeros(8),
        }
    )
    assert len(weights) == 5
    assert weights["layers.0.linear_attn.conv1d.weight"].shape == (16, 4, 1)
    for key in (
        "layers.0.input_layernorm.weight",
        "layers.3.self_attn.q_norm.weight",
        "norm.weight",
    ):
        assert_array_equal(np.array(weights[key]), np.ones(8))
    assert_array_equal(
        np.array(weights["layers.0.linear_attn.norm.weight"]), np.full(8, 3)
    )


def tiny_text_encoder():
    return Qwen3_5Model(
        TextConfig(
            model_type="qwen3_5_text",
            hidden_size=64,
            intermediate_size=128,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            num_hidden_layers=4,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=128,
            head_dim=16,
            rope_parameters={
                "type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        )
    )


def test_unpadded_prompt_encoding_without_a_kv_cache():
    pipeline = object.__new__(MOSSImagePipeline)
    tokenizer = Mock(return_value={"input_ids": np.array([[2, 3, 4, 5]])})
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    pipeline.tokenizer = tokenizer
    pipeline.text_encoder = tiny_text_encoder()
    pipeline.text_encoder.eval()
    hidden = pipeline._encode_prompt("cat", 32)
    assert hidden.shape == (1, 4, 64)
    assert bool(mx.all(mx.isfinite(hidden)).item())
    tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "cat"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    tokenizer.assert_called_once_with(
        "formatted prompt",
        truncation=True,
        max_length=32,
        return_tensors="np",
    )


def test_qwen_gdn_normalization_uses_l2_epsilon():
    attention = tiny_text_encoder().layers[0].linear_attn
    q = mx.arange(128, dtype=mx.float32).reshape(1, 1, 1, 128) * 1e-5
    k = q + 1e-4
    actual_q, actual_k = attention._normalize_qk(q, k)
    q, k = np.array(q), np.array(k)
    expected_q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(128)
    expected_k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    assert_allclose(np.array(actual_q), expected_q, atol=1e-7)
    assert_allclose(np.array(actual_k), expected_k, atol=1e-7)


@pytest.mark.parametrize("dtype", [None, "bfloat16"])
@pytest.mark.parametrize(
    "source_id", [None, "OpenMOSS-Team/MOSS-Image-Base", "OpenMOSS-Team/MOSS-Image-SFT"]
)
def test_original_to_mlx_checkpoint_round_trip(tmp_path, dtype, source_id):
    source, output = tmp_path / "source", tmp_path / "mlx"
    source.mkdir()
    (source / "model_index.json").write_text(
        json.dumps({"_class_name": "MOSSImagePipeline"})
    )
    for name in ("tokenizer", "scheduler"):
        (source / name).mkdir()
    (source / "tokenizer/tokenizer.json").write_text("{}")
    (source / "scheduler/scheduler_config.json").write_text("{}")
    (source / "scheduler/upstream_only.py").write_text("# not used at runtime")
    (source / "LICENSE").write_text("upstream license")
    text, transformer = tiny_text_encoder(), tiny_transformer()
    vae = Flux2VAE(decoder_block_out_channels=(32, 32))
    models = {"text_encoder": text, "transformer": transformer, "vae": vae}
    configs = {
        "text_encoder": {
            "text_config": asdict(text.args),
            "vision_config": {"unused": True},
        },
        "transformer": asdict(transformer.config),
        "vae": {"decoder_block_out_channels": [32, 32]},
    }
    for name, model in models.items():
        directory = source / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(configs[name]))
        weights = {}
        for key, value in tree_flatten(model.parameters()):
            if name == "text_encoder":
                if key == "norm.weight" or key.endswith(
                    (
                        ".input_layernorm.weight",
                        ".post_attention_layernorm.weight",
                        ".q_norm.weight",
                        ".k_norm.weight",
                    )
                ):
                    value = value - 1
                if key.endswith("conv1d.weight"):
                    value = value.moveaxis(1, 2)
                key = "language_model." + key
            elif name == "transformer":
                key = key.replace("time_embedding.layers.", "time_embedding.")
            else:
                key = key.replace(".to_out.", ".to_out.0.")
                if value.ndim == 4:
                    value = value.transpose(0, 3, 1, 2)
            weights[key] = value
        mx.save_safetensors(str(directory / "model.safetensors"), weights)
    options = {} if source_id is None else {"source_id": source_id}
    assert convert_moss_image(source, output, dtype=dtype, **options) == output
    assert (
        json.loads((output / "model_index.json").read_text()).get("source_model")
        == source_id
    )
    card = (output / "README.md").read_text()
    if source_id is None:
        assert "base_model:" not in card
        assert "MOSS-Image-SFT" not in card
    else:
        assert f"base_model: {source_id}\n" in card
    for name, loader in (
        ("text_encoder", load_text_encoder),
        ("transformer", load_transformer),
        ("vae", load_vae),
    ):
        loaded = dict(tree_flatten(loader(output).parameters()))
        expected = dict(tree_flatten(models[name].parameters()))
        assert loaded.keys() == expected.keys()
        for key, value in expected.items():
            if (
                dtype is not None
                and name != "vae"
                and not key.endswith((".A_log", ".dt_bias"))
            ):
                value = value.astype(getattr(mx, dtype))
            assert_allclose(
                np.array(loaded[key].astype(mx.float32)),
                np.array(value.astype(mx.float32)),
                atol=1e-7,
            )
        if dtype is not None and name != "vae":
            assert (
                json.loads((output / name / "config.json").read_text())["dtype"]
                == dtype
            )
    assert not (output / "scheduler/upstream_only.py").exists()
    assert "vision_config" not in json.loads(
        (output / "text_encoder/config.json").read_text()
    )
    assert (output / "LICENSE").read_text() == "upstream license"
    assert "generate_image" in (output / "README.md").read_text()
    reconverted = tmp_path / "reconverted"
    convert_moss_image(output, reconverted)
    assert (reconverted / "README.md").read_text() == card
    with pytest.raises(FileExistsError):
        convert_moss_image(source, output)


def fake_pipeline(monkeypatch):
    pipeline = object.__new__(MOSSImagePipeline)
    pipeline.model_path = Path("/unused")
    pipeline.evict_text_encoder = False
    pipeline.transformer_config = {"num_layers": 42, "latent_channels": 4}
    pipeline.scheduler_config = {}
    pipeline.text_encoder = object()
    pipeline._encode_prompt = lambda prompt, length: mx.full(
        (1, 1, 16), 1 if prompt else -1
    )
    calls = []

    def predict(latents, time, context, *, skip_layers=None):
        calls.append((float(context[0, 0, 0].item()), skip_layers))
        value = 2 if float(context[0, 0, 0].item()) > 0 else -1
        return mx.full(
            latents.shape, value - (1 if skip_layers else 0), dtype=mx.bfloat16
        )

    pipeline.transformer = predict
    captured = []

    def decode(latents):
        captured.append(latents)
        return mx.zeros(
            (latents.shape[0], 3, latents.shape[2] * 16, latents.shape[3] * 16)
        )

    pipeline.vae = SimpleNamespace(decode_packed_latents=decode)
    return pipeline, calls, captured


@pytest.mark.parametrize(
    "guidance,slg_scale,expected_velocity,calls_per_step",
    [
        (3, 0, 10, 2),
        (1, 0, 2, 1),
        (0, 0, -2, 2),
        (3, 2, 12, 3),
        (1, 2, 3, 2),
    ],
)
def test_pipeline_cfg_skip_guidance_and_repeatable_noise(
    monkeypatch, guidance, slg_scale, expected_velocity, calls_per_step
):
    pipeline, calls, captured = fake_pipeline(monkeypatch)
    image = pipeline.generate_array(
        "cat",
        seed=9,
        width=32,
        height=16,
        steps=1,
        guidance=guidance,
        slg_layers=[24],
        slg_scale=slg_scale,
        num_images=2,
    )
    noise = mx.random.normal((2, 4, 1, 2), key=mx.random.key(9))
    assert_allclose(
        np.array(captured[0]), np.array(noise - expected_velocity), atol=1e-6
    )
    assert len(calls) == calls_per_step
    if guidance != 1:
        assert calls[1] == (-1, frozenset({22}))
    assert image.shape == (2, 16, 32, 3)
    assert image.dtype == mx.uint8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 17},
        {"height": 0},
        {"steps": 0},
        {"guidance": float("nan")},
        {"num_images": 0},
        {"max_sequence_length": 4097},
        {"uncond_skip_layers": [42]},
        {"slg_scale": -1},
        {"slg_start": 0.8, "slg_stop": 0.2},
    ],
)
def test_invalid_inputs_fail_before_loading(monkeypatch, kwargs):
    pipeline, _, _ = fake_pipeline(monkeypatch)
    with pytest.raises(ValueError):
        pipeline.generate_array("cat", **kwargs)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_shared_flux2_vae_decoder_preserves_precision(dtype):
    vae = Flux2VAE(decoder_block_out_channels=(32, 32))
    vae.set_dtype(dtype)
    output = vae.decode_packed_latents(mx.zeros((1, 128, 1, 1), dtype=dtype))
    assert output.dtype == dtype
    assert output.shape == (1, 3, 4, 4)
