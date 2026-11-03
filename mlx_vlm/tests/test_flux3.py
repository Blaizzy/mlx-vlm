"""Flux3 architecture, native-package discovery, and image runtime contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

from mlx_vlm.generate.edit_image import ImageEditRequest, image_edit_model_class
from mlx_vlm.generate.image import (
    ImageGenerationRequest,
    image_generation_model_class,
    is_image_generation_model,
)
from mlx_vlm.generate.image_defaults import resolve_image_defaults
from mlx_vlm.models.flux3.config import Flux3Config, validate_dimensions
from mlx_vlm.models.flux3.model import Flux3ImageEditModel, Flux3ImageGenerationModel
from mlx_vlm.models.flux3.pipeline import (
    Flux3Image,
    image_ids,
    prepare_reference,
    timesteps,
)
from mlx_vlm.models.flux3.text_encoder import Flux3TextEncoder
from mlx_vlm.models.flux3.transformer import (
    Block,
    Flux3Transformer,
    QKNorm,
    timestep_embedding,
)
from mlx_vlm.models.qwen3_vl.config import TextConfig


def tiny_config(**kwargs):
    return Flux3Config(
        hidden=32,
        heads=2,
        depth=1,
        depth_single=1,
        axes_dim=(4, 4, 4, 4),
        in_channels={"image": 8, "image_cond": 8, "text": 16},
        **kwargs,
    )


def native_metadata(**kwargs):
    return {
        "model": "flux-3-image-klein",
        "dit": {
            "in_channels": {"image": 128, "image_cond": 128, "text": 20480},
            **kwargs,
        },
    }


@pytest.mark.parametrize("task", ["generate", "edit"])
def test_alias_defaults_do_not_load_or_download(monkeypatch, task):
    monkeypatch.setattr(mx, "load", Mock(side_effect=AssertionError("weight IO")))
    from mlx_vlm import utils

    monkeypatch.setattr(
        utils, "snapshot_download", Mock(side_effect=AssertionError("network IO"))
    )
    assert resolve_image_defaults("flux3-klein", task).steps == 16
    base = resolve_image_defaults("black-forest-labs/flux-3-image-klein-base", task)
    assert (base.steps, base.guidance) == (50, 3.5)


def test_native_metadata_discovers_renamed_generation_and_edit(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(native_metadata(flow_map=True)))
    # A component Qwen config must not route this package to the VL loader.
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder/config.json").write_text('{"model_type":"qwen3_vl"}')
    assert is_image_generation_model(str(tmp_path))
    assert image_generation_model_class(str(tmp_path)) is Flux3ImageGenerationModel
    assert image_edit_model_class(str(tmp_path)) is Flux3ImageEditModel
    assert resolve_image_defaults(str(tmp_path)).steps == 16


def test_flux3_download_excludes_quantized_variants(monkeypatch, tmp_path):
    from mlx_vlm.generate.image import _resolve_image_model_path
    from mlx_vlm.models.flux3 import download

    fetch = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(download, "snapshot_download", fetch)
    monkeypatch.setattr(download, "validate_model_layout", lambda path: Path(path))
    assert _resolve_image_model_path("flux3-klein", revision="pinned") == tmp_path
    assert fetch.call_args.args == ("black-forest-labs/flux-3-image-klein",)
    assert fetch.call_args.kwargs["revision"] == "pinned"
    assert "*.safetensors" not in fetch.call_args.kwargs["allow_patterns"]
    assert "model.safetensors" in fetch.call_args.kwargs["allow_patterns"]


def test_config_reads_native_sampler_and_rejects_cuda_quantization():
    data = native_metadata(flow_map=False)
    data["sampler"] = {"steps": 50, "guidance": 3.5, "alpha": 6.93}
    config = Flux3Config.from_dict(data)
    assert (config.flow_map, config.default_steps, config.default_guidance) == (
        False,
        50,
        3.5,
    )
    with pytest.raises(ValueError, match="BF16"):
        Flux3Config.from_dict({**data, "quantization": {"format": "nvfp4"}})


def test_schedule_matches_flow_map_reference():
    t = np.linspace(1, 0, 17, dtype=np.float32)
    assert_allclose(np.array(timesteps(16)), 6.93 * t / (1 + 5.93 * t), atol=1e-7)
    assert_allclose(
        np.array(timestep_embedding(mx.array([0.0])))[0], [1.0] * 128 + [0.0] * 128
    )
    with pytest.raises(ValueError):
        timesteps(0)


def test_bfloat16_timestep_rounds_before_float32_phases():
    t = mx.array([0.7], dtype=mx.bfloat16)
    # BF16 0.7 * 1000 rounds to 700 before conversion to float32.
    freqs = np.exp(-np.log(10000) * np.arange(128, dtype=np.float32) / 128)
    expected = mx.array(
        np.concatenate([np.cos(700 * freqs), np.sin(700 * freqs)])[None]
    ).astype(mx.bfloat16)
    assert_allclose(
        np.array(timestep_embedding(t).astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=0.004,
        rtol=0.004,
    )


def test_image_ids_use_half_spatial_coordinates_and_negative_reference_axis():
    expected = [[[0, 0, 0, -3], [0, 0, 0.5, -3], [0, 0.5, 0, -3], [0, 0.5, 0.5, -3]]]
    assert_array_equal(np.array(image_ids(32, 32, reference=2)), expected)


@pytest.mark.parametrize("flow_map", [False, True])
@pytest.mark.parametrize("reference_count", [0, 3])
def test_transformer_batched_samples_stay_independent(flow_map, reference_count):
    mx.random.seed(42)
    model = Flux3Transformer(tiny_config(flow_map=flow_map))
    data = {
        m: mx.random.normal((2, n, c))
        for m, n, c in [
            ("image", 4, 8),
            ("image_cond", reference_count, 8),
            ("text", 5, 16),
        ]
    }
    ids = {m: mx.zeros((2, x.shape[1], 4)) for m, x in data.items()}
    target = 0.3 if flow_map else None
    batch = model(data, ids, 0.7, target_timestep=target)
    individual = mx.concatenate(
        [
            model(
                {k: v[i : i + 1] for k, v in data.items()},
                {k: v[i : i + 1] for k, v in ids.items()},
                0.7,
                target_timestep=target,
            )
            for i in range(2)
        ]
    )
    assert_allclose(np.array(batch), np.array(individual), atol=1e-6, rtol=1e-5)
    assert batch.shape == (2, 4, 8)
    with pytest.raises(ValueError, match="target_timestep"):
        model(data, ids, 0.7, target_timestep=None if flow_map else 0.3)


def test_parallel_attention_mlp_block_matches_numpy():
    mx.random.seed(9)
    config = tiny_config()
    block = Block(config)
    x = np.random.default_rng(3).normal(size=(1, 3, 32)).astype(np.float32)
    shift, scale, gate = [
        np.random.default_rng(i).normal(size=(1, 1, 32)).astype(np.float32) * 0.1
        for i in range(3)
    ]
    weights = {k: np.array(v) for k, v in tree_flatten(block.parameters())}

    def linear(v, key):
        return v @ weights[key + ".weight"].T

    h = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)
    h = h * (1 + scale) + shift
    q, k, v = [
        linear(h, name).reshape(1, 3, 2, 16).transpose(0, 2, 1, 3)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    q = q / np.sqrt((q * q).mean(-1, keepdims=True) + 1e-6)
    k = k / np.sqrt((k * k).mean(-1, keepdims=True) + 1e-6)
    scores = (q @ k.transpose(0, 1, 3, 2)) / 4
    probs = np.exp(scores - scores.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    attn = (probs @ v).transpose(0, 2, 1, 3).reshape(1, 3, 32)
    a, b = np.split(linear(h, "mlp_in"), 2, axis=-1)
    expected = x + gate * (
        linear(attn, "attn_out") + linear(a / (1 + np.exp(-a)) * b, "mlp_out")
    )
    freqs = (mx.ones((1, 3, 8)), mx.zeros((1, 3, 8)))
    actual = block(mx.array(x), freqs, tuple(mx.array(v) for v in (shift, scale, gate)))
    assert_allclose(np.array(actual), expected, atol=5e-7, rtol=1e-5)


def test_qk_norm_rotates_adjacent_pairs():
    q = mx.array([[[[1.0, 2.0, 3.0, 4.0]]]])
    out, _ = QKNorm(4)(q, q, (mx.zeros((1, 1, 2)), mx.ones((1, 1, 2))))
    expected = np.array([-2.0, 1.0, -4.0, 3.0]) / np.sqrt(7.5 + 1e-6)
    assert_allclose(np.array(out).reshape(-1), expected, atol=1e-6)


def test_text_features_preserve_padding_and_hidden_state_layer_semantics():
    config = TextConfig(
        model_type="qwen3_vl_text",
        num_hidden_layers=3,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        vocab_size=32,
        rope_theta=10000,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [2, 1, 1]},
    )
    encoder = Flux3TextEncoder(config)
    tokens = mx.array([[1, 2, 3, 0, 0]])
    mask = mx.array([[1, 1, 1, 0, 0]])
    padded = encoder.get_prompt_embeds(tokens, mask, (1, 3))
    plain = encoder.get_prompt_embeds(tokens[:, :3], mask[:, :3], (1, 3))
    assert padded.shape == (1, 5, 32)
    assert_allclose(
        np.array(padded[:, :3].astype(mx.float32)),
        np.array(plain.astype(mx.float32)),
        atol=0.02,
        rtol=0.02,
    )
    # Padding hidden states are retained, as in the reference encoder.
    assert np.any(np.array(padded[:, 3:].astype(mx.float32)) != 0)


@pytest.mark.parametrize("multi", [False, True])
def test_reference_preparation_preserves_aspect_ratio_and_grid(multi):
    image = Image.new("RGB", (503, 277))
    out, crop = prepare_reference(image, multi=multi)
    assert out.size == ((496, 272) if multi else (512, 288))
    assert crop == (None if multi else (4, 5, 507, 282))


def test_wrappers_forward_model_defaults_and_edit_reference_paths():
    call = Mock(return_value=mx.zeros((64, 80, 3), dtype=mx.uint8))
    pipeline = SimpleNamespace(
        config=Flux3Config(),
        generate_array=call,
        edit_array=call,
        model_path=Path("/unused"),
        tokenizer=SimpleNamespace(count_tokens=lambda _: 10),
    )
    generation = Flux3ImageGenerationModel(pipeline, "flux3")
    result = generation.generate(ImageGenerationRequest("caption", width=80, height=64))
    assert (result.family, result.steps, result.width, result.height) == (
        "flux3",
        16,
        80,
        64,
    )
    edit = Flux3ImageEditModel(pipeline, "flux3")
    result = edit.edit(ImageEditRequest("caption", ("ref.png",)))
    assert result.metadata["reference_count"] == 1
    assert call.call_args.args == ("caption", ("ref.png",))
    assert call.call_args.kwargs["guidance"] == 1


@pytest.mark.parametrize("width,height", [(63, 64), (64, 65), (4112, 64)])
def test_invalid_dimensions(width, height):
    with pytest.raises(ValueError):
        validate_dimensions(width, height)


def test_edit_requires_one_to_ten_references():
    pipeline = Flux3Image.__new__(Flux3Image)
    for paths in ([], ["ref.png"] * 11):
        with pytest.raises(ValueError, match="1 and 10"):
            pipeline.edit_array("caption", paths)


def test_default_edit_crops_reference_padding_but_explicit_size_preserves_canvas(
    tmp_path,
):
    path = tmp_path / "reference.png"
    Image.new("RGB", (503, 277)).save(path)
    pipeline = Flux3Image.__new__(Flux3Image)
    pipeline.config = Flux3Config()
    pipeline.evict_transformer = False
    pipeline.upsample_prompt = False
    pipeline.prompt_expansion_model = "unused"
    pipeline._encode_prompt = Mock(
        return_value=(mx.zeros((1, 1, 20480)), mx.zeros((1, 1, 4)))
    )
    pipeline.transformer = lambda data, ids, t, **kw: mx.zeros_like(data["image"])
    pipeline.vae = SimpleNamespace(
        encode=lambda image: mx.zeros(
            (1, image.shape[1] // 16 * (image.shape[2] // 16), 128)
        ),
        decode=lambda tokens, height, width: mx.zeros((1, height, width, 3)),
    )
    assert pipeline.edit_array("caption", [path], steps=1).shape == (277, 503, 3)
    assert pipeline.edit_array(
        "caption", [path], steps=1, width=512, height=288
    ).shape == (288, 512, 3)
