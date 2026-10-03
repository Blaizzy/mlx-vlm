"""Video model components, generation, dispatch, conversion, and muxing."""

import importlib
import json
import math
import shutil
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image

from mlx_vlm.generate import (
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReference,
    is_video_generation_model,
    video_generation_model_class,
)
from mlx_vlm.tests.test_image_generation_models import _check_download, _write_files

h3 = importlib.import_module("mlx_vlm.models.minimax_h3")
h3_model = importlib.import_module("mlx_vlm.models.minimax_h3.model")
processing = importlib.import_module("mlx_vlm.models.minimax_h3.processing")
qwen = importlib.import_module("mlx_vlm.models.qwen3_vl")
qwen_language = importlib.import_module("mlx_vlm.models.qwen3_vl.language")
VIDEO_CASES = json.loads(
    Path(__file__).with_name("video_generation_cases.json").read_text()
)
REFERENCES = VIDEO_CASES["references"]


def _assert_reference(name, actual, *, rtol=1e-7, atol=0):
    assert_allclose(
        actual,
        np.array(REFERENCES[name], dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


def _config(name):
    case = VIDEO_CASES["configs"][name]
    return getattr(h3, case["class"]).from_dict(case["values"])


def _qwen_config(layers=51):
    return deepcopy(VIDEO_CASES["qwen_model"]) | {
        "text_config": _qwen_text_fields(layers, "rope_type")
    }


# Video model components


class _SyntheticTokenizer:
    special_ids = {
        "<|vision_start|>": 900,
        "<|image_pad|>": 901,
        "<|video_pad|>": 902,
        "<|vision_end|>": 903,
    }

    def __call__(self, value, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [1000 + ord(character) for character in value]}

    def convert_tokens_to_ids(self, token):
        return self.special_ids[token]


class _SyntheticConditioner:
    def __init__(self, workflow="t2va"):
        self.workflow = workflow

    def _encode(self, prompt, references):
        assert (
            prompt
            == {
                "t2va": "synthetic",
                "fl2va": "synthetic-fl",
                "ref2va": "synthetic-ref",
            }[self.workflow]
        )
        assert len(references or []) == (self.workflow != "t2va")
        return h3.MiniMaxH3ConditioningOutput(
            hidden_states=mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) * 0.01,
            token_tags=mx.array(
                [1, 0 if self.workflow == "ref2va" else 1], dtype=mx.int32
            ),
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
        )

    def encode_fl2va(self, prompt, images=None):
        return self._encode(prompt, images)

    def encode_ref2va(self, prompt, references):
        return self._encode(prompt, references)


class _SyntheticRefPipeline(h3.MiniMaxH3Pipeline):
    def _prepare_references(self, references, num_frames):
        del references
        image = (mx.arange(64 * 64 * 3) % 256).astype(mx.uint8).reshape(64, 64, 3)
        return [h3.MiniMaxH3PreparedReference(kind="image", image=image)], num_frames


def _load_canonical_synthetic_weights(model, *, video=False, text=False):
    weights = []
    for offset, (key, parameter) in enumerate(sorted(tree_flatten(model.parameters()))):
        shape = parameter.shape
        source_shape = (
            (shape[0], shape[-1], *shape[1:-1])
            if video and parameter.ndim == 5
            else shape
        )
        values = ((mx.arange(math.prod(shape)) % 29).astype(mx.float32) - 14.0) * 0.005
        values = (values + ((offset % 7) - 3) * 0.001).reshape(source_shape)
        if (
            ("norm" in key and key.endswith("weight"))
            if video
            else key.endswith("norm.weight") or (text and "layernorm.weight" in key)
        ):
            values = 1.0 + values * 0.1
        if video and parameter.ndim == 5:
            values = values.transpose(0, 2, 3, 4, 1)
        weights.append((key, values))
    model.load_weights(weights, strict=True)


def _canonical_audio_vae_source_weights(
    model: h3.MiniMaxH3AudioVAE,
) -> dict[str, mx.array]:
    source_shapes = {}
    parameters = dict(tree_flatten(model.parameters()))
    for key, parameter in parameters.items():
        weight_norm = key.endswith(".weight") and (
            key.startswith("encoder.block.")
            or key.startswith("decoder.conv_pre.")
            or key.startswith("decoder.ups.")
            or key.startswith("decoder.resblocks.")
            or key.startswith("decoder.conv_post.")
        )
        if weight_norm:
            prefix = key[: -len("weight")]
            if key.startswith("decoder.ups."):
                source_shapes[f"{prefix}weight_v"] = (
                    parameter.shape[-1],
                    parameter.shape[0],
                    parameter.shape[1],
                )
                source_shapes[f"{prefix}weight_g"] = (parameter.shape[-1], 1, 1)
            else:
                source_shapes[f"{prefix}weight_v"] = (
                    parameter.shape[0],
                    parameter.shape[-1],
                    parameter.shape[1],
                )
                source_shapes[f"{prefix}weight_g"] = (parameter.shape[0], 1, 1)
        else:
            shape = parameter.shape
            if parameter.ndim == 3 and key.endswith("weight"):
                shape = (parameter.shape[0], parameter.shape[-1], parameter.shape[1])
            source_shapes[key] = shape

    source = {}
    for offset, (key, shape) in enumerate(sorted(source_shapes.items())):
        if key.endswith("filter"):
            values = parameters[key]
        else:
            values = (
                (mx.arange(math.prod(shape)) % 29).astype(mx.float32) - 14.0
            ) * 0.004
            values = values + ((offset % 7) - 3) * 0.0007
            values = values.reshape(shape)
            if "norm" in key and key.endswith("weight"):
                values = 1.0 + values * 0.1
            if key.endswith("weight_g"):
                values = 0.2 + mx.abs(values)
            if key.endswith(".alpha") and values.ndim == 3:
                values = 1.0 + values * 0.1
        source[key] = values
    return source


def _load_canonical_audio_vae_weights(model: h3.MiniMaxH3AudioVAE) -> None:
    source = _canonical_audio_vae_source_weights(model)
    model.load_weights(sorted(model.sanitize(source).items()), strict=True)


def test_geometry_and_patch_roundtrip():
    assert h3.resolve_canvas_size(16, 9) == (768, 1344)
    assert h3.resolve_canvas_size(9, 16) == (1344, 768)
    assert h3.align_num_frames(120) == 124
    assert h3.video_latent_num_frames(124) == 37
    assert h3.audio_latent_num_frames(124) == 207
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        h3.resolve_canvas_size(5, 1)

    latents = mx.arange(1 * 2 * 3 * 4 * 6).reshape(1, 2, 3, 4, 6)
    rows = h3.patchify_video_latents(latents, (1, 2, 2))
    restored = h3.unpatchify_video_tokens(rows, 3, 4, 6, 2, (1, 2, 2))
    assert rows.shape == (18, 8)
    assert mx.array_equal(latents, restored).item()


def test_fl2va_packing_matches_diffusers_golden():
    layout = h3.build_packed_sequence(
        mx.array([1, 1]),
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=(1, 1, 1),
    )
    expected_positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 16.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 16.0],
        [2.0, 16.0, 0.0],
        [2.0, 16.0, 16.0],
    ]
    assert layout.position_ids.tolist() == expected_positions
    assert layout.token_tags.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]
    assert layout.video_indices.tolist() == [4, 5, 6, 7]
    assert layout.audio_indices.tolist() == [2, 3]
    assert layout.text_indices.tolist() == [0, 1]

    timesteps, timestep_indices = h3.build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    assert timesteps.tolist() == [0.25, 0.5]
    assert timestep_indices.tolist() == [0, 0, 1, 1, 0, 0, 0, 0]


def test_ref2va_mixed_packing_matches_diffusers_golden():
    references = [
        h3.MiniMaxH3PreparedReference(
            kind="image", num_latent_frames=1, latent_height=4, latent_width=6
        ),
        h3.MiniMaxH3PreparedReference(
            kind="video",
            has_audio=True,
            num_latent_frames=3,
            latent_height=4,
            latent_width=4,
            num_audio_latents=2,
        ),
        h3.MiniMaxH3PreparedReference(
            kind="audio", has_audio=True, num_audio_latents=3
        ),
    ]
    layout = h3.build_ref2va_packed_sequence(
        mx.array([1, 0, 1, 1, 0]),
        references,
        num_latent_frames=2,
        latent_height=6,
        latent_width=8,
        num_audio_latents=4,
        patch_size=(1, 2, 2),
    )
    assert layout.sequence_length == 65
    assert layout.num_condition_video_rows == 18
    assert layout.num_condition_audio_rows == 10
    assert layout.video_indices.tolist() == [
        *range(5, 11),
        *range(15, 27),
        *range(41, 65),
    ]
    assert layout.audio_indices.tolist() == [*range(11, 15), *range(27, 41)]
    positions = layout.position_ids.tolist()
    assert positions[5][0] == 5.0
    assert positions[11][0] == 6.0
    assert positions[15][0] == 6.0
    assert positions[27][0] == 21.0
    assert positions[33][0] == 24.0
    assert positions[41][0] == 24.0


def test_scheduler_matches_diffusers_golden():
    scheduler = h3.MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(5)
    assert_allclose(
        scheduler.sigmas.tolist(),
        [1.0, 0.9729729891, 0.9230769277, 0.8000000119, 0.0],
        rtol=0.0,
        atol=5e-10,
    )
    assert_allclose(
        scheduler.timesteps.tolist(),
        [0.0, 0.0270270109, 0.0769230723, 0.1999999881],
        rtol=0.0,
        atol=5e-10,
    )

    sample = mx.array([[0.1, -0.2], [0.3, 0.4]], dtype=mx.float32)
    velocity = mx.array([[0.5, 0.6], [-0.7, 0.8]], dtype=mx.float32)
    output = scheduler.step(velocity, scheduler.timesteps[0], sample)
    assert_allclose(
        output,
        np.array([[0.11351351, -0.1837838], [0.2810811, 0.42162162]], dtype=np.float32),
        rtol=0.0,
        atol=1e-8,
    )


def test_mlx_lanczos_matches_pillow_reference():
    pixels = np.random.default_rng(7).integers(0, 256, size=(13, 15, 3), dtype=np.uint8)
    for height, width in ((7, 9), (21, 26), (13, 23)):
        expected = np.asarray(
            Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
        )
        actual = processing.resize_lanczos(mx.array(pixels), height, width)
        assert_array_equal(actual, expected)

    stretched = processing.prepare_keyframe_image(mx.array(pixels), 8, 10, stretch=True)
    covered = processing.prepare_keyframe_image(mx.array(pixels), 8, 10, stretch=False)
    assert stretched.shape == (8, 10, 3)
    assert covered.shape == (8, 10, 3)


def test_mlx_reference_frame_and_audio_processing():
    frames = mx.arange(10, dtype=mx.uint8).reshape(10, 1, 1, 1)
    frames = mx.repeat(frames, 3, axis=-1)
    upsampled = processing.resample_reference_frames(frames, 10.0)
    downsampled = processing.resample_reference_frames(frames, 30.0)
    assert upsampled[:, 0, 0, 0].tolist() == REFERENCES["upsampled_frame_indices"]
    assert downsampled[:, 0, 0, 0].tolist() == [0, 1, 3, 4, 5, 6, 8, 9]

    sampled, timestamps = processing.sample_reference_video_frames(
        mx.repeat(frames, 4, axis=0)
    )
    assert sampled[:, 0, 0, 0].tolist() == [0, 3, 6, 9]
    assert timestamps == [0.25, 1.25]

    waveform = mx.arange(17, dtype=mx.float32)[None] / 17.0
    resampled = processing.prepare_reference_waveform(waveform, 16000, 32000, 1.0)
    assert resampled.shape == (2, 34)
    assert_allclose(
        resampled[0, :5],
        [0.00016416, 0.02440057, 0.05869471, 0.09065638, 0.11777476],
        rtol=2e-4,
        atol=2e-6,
    )
    assert_array_equal(resampled[0], resampled[1])

    normalized = processing.normalize_visual_vae_pixels(
        mx.zeros((2, 3, 4, 3), mx.uint8)
    )
    assert normalized.shape == (1, 3, 2, 3, 4)
    assert_allclose(
        normalized[0, :, 0, 0, 0],
        -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]),
        rtol=1e-6,
    )

    decoded_video, decoded_fps = processing.decode_video(
        np.zeros((3, 3, 4, 5), dtype=np.uint8)
    )
    assert decoded_video.shape == (3, 4, 5, 3)
    assert decoded_fps is None
    decoded_audio, decoded_rate = processing.decode_audio(
        np.zeros((17, 2), dtype=np.float32)
    )
    assert decoded_audio.shape == (2, 17)
    assert decoded_rate is None


def test_video_soundtrack_decode_moves_pcm_directly_into_mlx(monkeypatch):
    pcm = np.array([[0.25, -0.5], [0.75, 0.125]], dtype="<f4")

    def fake_run(command, **kwargs):
        del kwargs
        if command[0].endswith("ffprobe"):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"streams": [{"sample_rate": "48000", "channels": 2}]}
                ),
            )
        assert command[0].endswith("ffmpeg")
        return SimpleNamespace(returncode=0, stdout=pcm.tobytes(), stderr=b"")

    monkeypatch.setattr(processing.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr(processing.subprocess, "run", fake_run)
    waveform, sample_rate = processing.decode_video_soundtrack("reference.mp4")
    assert sample_rate == 48000
    assert_array_equal(waveform, pcm.T)
    assert processing.decode_video_soundtrack(mx.zeros((2, 4, 4, 3), mx.uint8)) is None


def test_qwen_processors_match_transformers_synthetic_golden():
    image = (
        np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 256
    ).astype(np.uint8)
    video = (
        np.arange(2 * 64 * 64 * 3, dtype=np.uint32).reshape(2, 64, 64, 3) % 256
    ).astype(np.uint8)
    image_patches, image_grid = processing.process_qwen_images([mx.array(image)])
    video_patches, video_grid = processing.process_qwen_videos([mx.array(video)])

    assert image_patches.shape == (256, 1536)
    assert video_patches.shape == (16, 1536)
    assert image_grid.tolist() == [[1, 16, 16]]
    assert video_grid.tolist() == [[1, 4, 4]]
    _assert_reference(
        "processor_image",
        image_patches[0, :8],
    )
    _assert_reference(
        "processor_video",
        video_patches[0, :24],
    )


def test_prompt_presentations_match_diffusers_ordering():
    tokenizer = _SyntheticTokenizer()
    fl_ids, fl_tags = h3.build_fl2va_presentation(tokenizer, "go", [2])
    label_length = len("<Picture 1>: ")
    assert fl_ids[label_length : label_length + 4] == [900, 901, 901, 903]
    assert fl_tags[label_length : label_length + 4] == [0, 0, 0, 0]
    assert h3.create_mm_token_type_ids(fl_ids, tokenizer)[
        label_length : label_length + 4
    ] == [0, 1, 1, 0]

    references = [
        h3.MiniMaxH3PreparedReference(kind="image"),
        h3.MiniMaxH3PreparedReference(
            kind="video", has_audio=True, block_timestamps=[0.25, 1.25]
        ),
        h3.MiniMaxH3PreparedReference(kind="audio", has_audio=True),
    ]
    ids, tags = h3.build_ref2va_presentation(
        tokenizer,
        "prompt",
        references,
        image_token_counts=[1],
        video_block_token_counts=[2],
    )
    text = "".join(chr(token - 1000) for token, tag in zip(ids, tags) if tag == 1)
    assert text == (
        "<Picture 1>: <Audio 1>: <Video 1>: <0.2 seconds><1.2 seconds><Audio 2>: prompt"
    )
    assert tags.count(0) == 3 + 4 + 4
    assert h3.trim_reference_num_frames(5) == 22
    assert h3.trim_reference_num_frames(39) == 39


def _qwen_text_fields(layers, rope_key):
    return dict(
        VIDEO_CASES["qwen"],
        num_hidden_layers=layers,
        rope_scaling={rope_key: "default", "mrope_section": [1, 1, 1]},
    )


def test_qwen_layer_two_matches_transformers_synthetic_golden():
    config = qwen.TextConfig(**_qwen_text_fields(3, "type"))
    model = qwen_language.Qwen3VLModel(config)
    _load_canonical_synthetic_weights(model, text=True)

    with mx.stream(mx.cpu):
        hidden_states = model(
            mx.array([[1, 2, 3, 4]], dtype=mx.int32),
            stop_after_layer=2,
            apply_final_norm=False,
        )
        mx.eval(hidden_states)
    _assert_reference("conditioner", hidden_states, rtol=1e-06, atol=2e-08)


def test_qwen_mixed_image_video_deepstack_keeps_token_order():
    config = _qwen_config(layers=1)
    model = qwen.Model(qwen.ModelConfig.from_dict(config))
    image_layers = [mx.array([[1.0, 1.5], [2.0, 2.5]])]
    video_layers = [mx.array([[3.0, 3.5]])]
    merged = model._merge_deepstack_features(
        mx.array([[29, 30, 29]], dtype=mx.int32), image_layers, video_layers
    )
    assert_array_equal(merged, [[[1.0, 1.5], [3.0, 3.5], [2.0, 2.5]]])


def _transformer_inputs(config):
    layout = h3.build_packed_sequence(
        mx.array([1, 1]),
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=config.patch_size,
    )
    timesteps, timestep_indices = h3.build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    return (
        (mx.arange(4, dtype=mx.float32).reshape(1, 4, 1) - 1.5) * 0.1,
        (mx.arange(4, dtype=mx.float32).reshape(1, 2, 2) - 1.5) * 0.1,
        (mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) - 4.5) * 0.05,
        timesteps,
        timestep_indices,
        layout.token_tags,
        layout.position_ids,
        layout.video_indices,
        layout.audio_indices,
        layout.text_indices,
    )


def test_tiny_transformer_matches_diffusers_synthetic_golden():
    config = _config("transformer")
    model = h3.MiniMaxH3Transformer(config)
    _load_canonical_synthetic_weights(model)
    args = _transformer_inputs(config)

    with mx.stream(mx.cpu):
        output = model(*args)
        mx.eval(output.sample, output.audio_sample)

    _assert_reference("transformer_video", output.sample, rtol=1e-06, atol=1e-07)
    _assert_reference("transformer_audio", output.audio_sample, rtol=1e-06, atol=1e-07)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_tiny_transformer_adaln_cache_is_bitwise_identical_after_drop(dtype):
    config = _config("transformer")
    model = h3.MiniMaxH3Transformer(config)
    _load_canonical_synthetic_weights(model)
    model.set_dtype(dtype)
    args = _transformer_inputs(config)

    with mx.stream(mx.cpu):
        live = model(*args)
        mx.eval(live.sample, live.audio_sample)
        cache = model.build_adaln_cache(args[3])
        cached = model(*args, cache)
        mx.eval(cached.sample, cached.audio_sample)
        other = h3.MiniMaxH3Transformer(config)
        _load_canonical_synthetic_weights(other)
        other.set_dtype(dtype)
        with pytest.raises(ValueError, match="different transformer"):
            other(*args, cache)
        freed = model.drop_adaln_weights()
        dropped = model(*args, cache)
        mx.eval(dropped.sample, dropped.audio_sample)

    assert freed > 0
    assert cache.nbytes > 0
    assert not model.adaln_weights_available
    remaining_keys = {key for key, _ in tree_flatten(model.parameters())}
    assert "transformer_blocks.0.adaln_proj.linear.weight" not in remaining_keys
    assert "norm_out.linear.weight" not in remaining_keys
    assert mx.array_equal(live.sample, cached.sample).item()
    assert mx.array_equal(live.audio_sample, cached.audio_sample).item()
    assert mx.array_equal(live.sample, dropped.sample).item()
    assert mx.array_equal(live.audio_sample, dropped.audio_sample).item()
    with pytest.raises(RuntimeError, match="AdaLN projection weights were dropped"):
        model(*args)


@pytest.mark.parametrize("case", VIDEO_CASES["vae"], ids=lambda case: case["id"])
def test_vae_matches_diffusers_synthetic_golden(case):
    config = VIDEO_CASES["configs"][case["config"]]
    model = getattr(h3, config["class"].removesuffix("Config"))(_config(case["config"]))
    video = case["id"] == "video"
    if video:
        _load_canonical_synthetic_weights(model, video=True)
        model.disable_tiling()
    else:
        _load_canonical_audio_vae_weights(model)
    pixels = (
        (mx.arange(math.prod(case["shape"]), dtype=mx.float32) % case["modulus"])
        - case["offset"]
    ).reshape(case["shape"]) * case["scale"]
    # Keep reference comparisons on CPU: Metal matmul uses lower-precision fast math.
    with mx.stream(mx.cpu):
        posterior = model.encode(pixels)
        outputs = {"latents": posterior.mode()}
        outputs["decoded"] = model.decode(outputs["latents"]).sample
        if not video:
            outputs["logs"] = posterior.logs
        mx.eval(*outputs.values())
    for name, shape in case["output_shapes"].items():
        assert outputs[name].shape == tuple(shape)
    for name, reference in case["references"].items():
        _assert_reference(
            reference["name"],
            outputs[name].flatten()[:8],
            rtol=reference["rtol"],
            atol=reference["atol"],
        )
    if video:
        model.enable_tiling(8, 8, 4, 4)
        assert model._split_tiles(12, 8, 4) == ([0, 4], [8, 8], [4])


def _tiny_pipeline_modules():
    transformer = h3.MiniMaxH3Transformer(_config("transformer"))
    _load_canonical_synthetic_weights(transformer)
    return (
        transformer,
        h3.MiniMaxH3VideoVAE(_config("pipeline_video_vae")),
        h3.MiniMaxH3AudioVAE(_config("pipeline_audio_vae")),
    )


def test_video_path_automatically_contributes_its_soundtrack(monkeypatch):
    transformer, video_vae, audio_vae = _tiny_pipeline_modules()
    pipeline = h3.MiniMaxH3Pipeline(
        transformer=transformer,
        conditioner=_SyntheticConditioner("ref2va"),
        video_vae=video_vae,
        audio_vae=audio_vae,
        partition="ref2va",
    )
    frames = mx.zeros((5, 8, 8, 3), mx.uint8)
    soundtrack = mx.zeros((2, 100), mx.float32)
    monkeypatch.setattr(
        h3.pipeline, "decode_video_soundtrack", lambda path: (soundtrack, 20)
    )
    monkeypatch.setattr(h3.pipeline, "decode_video", lambda path: (frames, 24.0))
    monkeypatch.setattr(
        h3.pipeline,
        "prepare_reference_frames",
        lambda values, num_frames: values[:num_frames],
    )
    monkeypatch.setattr(
        h3.pipeline,
        "prepare_reference_waveform",
        lambda waveform, *args: waveform,
    )

    prepared, num_frames = pipeline._prepare_references(
        [h3.MiniMaxH3Reference(video="reference.mp4")], None
    )
    assert num_frames == 124
    assert len(prepared) == 1
    assert prepared[0].kind == "video"
    assert prepared[0].has_audio
    assert prepared[0].waveform.shape == (2, 100)


def _workflow_pipeline(workflow="t2va", progress_callback=None):
    transformer, video_vae, audio_vae = _tiny_pipeline_modules()
    if workflow == "ref2va":
        transformer = h3.MiniMaxH3Transformer(
            replace(_config("transformer"), patch_size=(1, 2, 2))
        )
        _load_canonical_synthetic_weights(transformer)
    factory = _SyntheticRefPipeline if workflow == "ref2va" else h3.MiniMaxH3Pipeline
    pipeline = factory(
        transformer=transformer,
        conditioner=_SyntheticConditioner(workflow),
        video_vae=video_vae,
        audio_vae=audio_vae,
        **({"partition": "ref2va"} if workflow == "ref2va" else {}),
    )
    width = 32 if workflow == "t2va" else 64
    media = {
        "fl2va": {"image": mx.zeros((64, 64, 3), mx.uint8)},
        "ref2va": {"references": [object()]},
    }.get(workflow, {})
    request = h3.MiniMaxH3GenerationRequest(
        prompt={
            "t2va": "synthetic",
            "fl2va": "synthetic-fl",
            "ref2va": "synthetic-ref",
        }[workflow],
        height=width,
        width=width,
        num_frames=124,
        num_inference_steps=2 if workflow == "t2va" else 3,
        output_type="latent",
        latents=mx.zeros(
            (1, 1, h3.video_latent_num_frames(124), width // 32, width // 32),
            mx.float32,
        ),
        audio_latents=mx.zeros((2, 2, h3.audio_latent_num_frames(124)), mx.float32),
        progress_callback=progress_callback,
        **media,
    )
    return pipeline, request


def test_tiny_t2va_pipeline_runs_joint_denoise_to_latents():
    num_video_latents = h3.video_latent_num_frames(124)
    num_audio_latents = h3.audio_latent_num_frames(124)
    progress_events = []
    pipeline, request = _workflow_pipeline(
        progress_callback=lambda *event: progress_events.append(event)
    )
    with mx.stream(mx.cpu):
        output = pipeline.generate(request)
        mx.eval(output.video, output.audio)
    assert output.video.shape == (1, 1, num_video_latents, 1, 1)
    assert output.audio.shape == (2, 2, num_audio_latents)
    assert output.metadata["partition"] == "fl2va"
    assert [event[:3] for event in progress_events] == [
        ("prepare", 0, 2),
        ("cache_adaln", 0, 2),
        ("cache_adaln", 1, 2),
        ("cache_adaln", 2, 2),
        ("denoise", 0, 1),
        ("denoise", 1, 1),
        ("decode", 1, 1),
        ("decoded", 1, 1),
    ]
    assert all(event[3] == 124 for event in progress_events)
    assert output.metadata["adaln_cached"]
    assert output.metadata["adaln_weights_dropped"]
    assert output.metadata["adaln_cache_bytes"] > 0
    assert output.metadata["adaln_weights_freed_bytes"] > 0
    assert mx.all(mx.isfinite(output.video)).item()
    assert mx.all(mx.isfinite(output.audio)).item()

    with mx.stream(mx.cpu):
        repeated = pipeline.generate(replace(request, progress_callback=None))
        mx.eval(repeated.video, repeated.audio)
    assert mx.array_equal(output.video, repeated.video).item()
    assert mx.array_equal(output.audio, repeated.audio).item()

    with pytest.raises(RuntimeError, match="reload the pipeline"):
        pipeline.generate(
            replace(request, num_inference_steps=3, progress_callback=None)
        )


@pytest.mark.parametrize("workflow", ["fl2va", "ref2va"])
def test_conditioned_pipeline_cached_trajectory_is_bitwise_identical(workflow):
    pipeline, request = _workflow_pipeline(workflow)
    with mx.stream(mx.cpu):
        live = pipeline.generate(
            replace(request, cache_adaln=False, drop_adaln_weights=False)
        )
        mx.eval(live.video, live.audio)
        output = pipeline.generate(request)
        mx.eval(output.video, output.audio)
    assert output.video.shape == (1, 1, h3.video_latent_num_frames(124), 2, 2)
    assert output.audio.shape == (2, 2, h3.audio_latent_num_frames(124))
    assert output.metadata["partition"] == workflow
    assert mx.array_equal(live.video, output.video).item()
    assert mx.array_equal(live.audio, output.audio).item()
    assert output.metadata["adaln_weights_dropped"]
    assert mx.all(mx.isfinite(output.video)).item()
    assert mx.all(mx.isfinite(output.audio)).item()


def _canonical_parameter_weights(model) -> dict[str, mx.array]:
    weights = {}
    for offset, (key, parameter) in enumerate(sorted(tree_flatten(model.parameters()))):
        values = (
            (mx.arange(math.prod(parameter.shape)) % 31).astype(mx.float32) - 15.0
        ) * 0.003
        values = values.reshape(parameter.shape) + (offset % 5) * 0.0002
        if "norm" in key and key.endswith("weight"):
            values = 1.0 + values * 0.1
        weights[key] = values
    return weights


def _write_official_component(path, config, weights):
    name = "diffusion_pytorch_model.safetensors"
    index = dict(
        metadata=dict(total_size=sum(value.nbytes for value in weights.values())),
        weight_map={key: name for key in sorted(weights)},
    )
    _write_files(path, metadata={"config.json": config, name + ".index.json": index})
    mx.save_safetensors(str(path / name), dict(sorted(weights.items())))


def _tiny_qwen_source():
    config = _qwen_config()
    model = qwen.Model(qwen.ModelConfig.from_dict(config))
    official = {}
    for key, value in _canonical_parameter_weights(model).items():
        if key.startswith("vision_tower."):
            official_key = key.replace("vision_tower", "model.visual", 1)
            if key == "vision_tower.patch_embed.proj.weight":
                value = value.transpose(0, 4, 1, 2, 3)
        elif key.startswith("language_model.model."):
            official_key = key.replace(
                "language_model.model", "model.language_model", 1
            )
        elif key == "language_model.lm_head.weight":
            official_key = "lm_head.weight"
        else:
            raise AssertionError(f"unmapped Qwen test tensor: {key}")
        official[official_key] = value
    return config, official


def _write_tiny_tokenizer(path):
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "<|vision_start|>": 1,
                "<|image_pad|>": 2,
                "<|video_pad|>": 3,
                "<|vision_end|>": 4,
            },
            unk_token="<unk>",
        )
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    fast.save_pretrained(path)


def _write_tiny_official_h3(root):
    transformer_config = _config("transformer")
    for name in ("transformer", "transformer_ref"):
        transformer = h3.MiniMaxH3Transformer(transformer_config)
        _write_official_component(
            root / name,
            asdict(transformer_config),
            _canonical_parameter_weights(transformer),
        )

    video_config = _config("video_vae")
    video_vae = h3.MiniMaxH3VideoVAE(video_config)
    video_weights = {}
    for key, value in _canonical_parameter_weights(video_vae).items():
        video_weights[key] = (
            value.transpose(0, 4, 1, 2, 3) if value.ndim == 5 else value
        )
    _write_official_component(root / "vae", asdict(video_config), video_weights)

    audio_config = _config("audio_vae")
    audio_vae = h3.MiniMaxH3AudioVAE(audio_config)
    _write_official_component(
        root / "audio_vae",
        asdict(audio_config),
        _canonical_audio_vae_source_weights(audio_vae),
    )

    qwen_config, qwen_weights = _tiny_qwen_source()
    _write_official_component(root / "text_encoder", qwen_config, qwen_weights)
    _write_tiny_tokenizer(root / "tokenizer")
    (root / "LICENSE").write_text("synthetic license fixture\n")


@pytest.mark.parametrize("case", VIDEO_CASES["downloads"], ids=lambda case: case["id"])
def test_video_download_contract(case, tmp_path, monkeypatch):
    _check_download(tmp_path, monkeypatch, case)


def test_official_layout_conversion_and_strict_reload(tmp_path, monkeypatch):
    source = tmp_path / "official"
    _write_tiny_official_h3(source)

    t2_plan = h3.download_plan("t2va")
    fl_plan = h3.download_plan("fl2va")
    ref_plan = h3.download_plan("ref2va")
    assert t2_plan.revision == "b3c7290e66afdf293bef3b9077b7a266ef421f34"
    assert h3.download_plan("t2va", repo_id="test-org/minimax-h3").revision is None
    assert t2_plan.partition == "fl2va"
    assert t2_plan.components == fl_plan.components
    assert t2_plan.patterns == fl_plan.patterns
    assert "transformer" in fl_plan.components
    assert "transformer_ref" not in fl_plan.components
    assert "transformer_ref" in ref_plan.components
    assert "transformer" not in ref_plan.components
    assert "model_index.json" in fl_plan.patterns
    assert "modular_model_index.json" in fl_plan.patterns
    assert not any(
        pattern.startswith(("FL2VA/", "Ref2VA/"))
        for pattern in (*fl_plan.patterns, *ref_plan.patterns)
    )
    with pytest.raises(ValueError, match="workflow is required"):
        h3.download_plan()
    with pytest.raises(ValueError, match="workflow must be"):
        h3.download_plan("unknown")

    download_calls = []

    def fake_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(h3.download, "snapshot_download", fake_snapshot_download)
    remote_pipeline = h3.load_pipeline(
        "test-org/minimax-h3", workflow="t2va", text_only=True, revision="test-revision"
    )
    assert remote_pipeline.partition == "fl2va"
    assert download_calls[0]["revision"] == "test-revision"
    assert "transformer/**" in download_calls[0]["allow_patterns"]
    assert "transformer_ref/**" not in download_calls[0]["allow_patterns"]

    with pytest.raises(ValueError, match="uses the 'fl2va' partition"):
        h3.load_pipeline(source, workflow="t2va", partition="ref2va")

    official_pipeline = h3.load_pipeline(source, partition="fl2va", text_only=True)
    assert official_pipeline.partition == "fl2va"

    dry_run = h3.convert_minimax_h3(
        source, tmp_path / "unused", partition="fl2va", text_only=True, dry_run=True
    )
    assert dry_run.dry_run
    assert dry_run.source_bytes > 0
    assert dry_run.converted_bytes > 0
    assert dry_run.tensor_counts["conditioner"] > 0
    assert not dry_run.destination.exists()

    fl_path = tmp_path / "fl"
    fl_report = h3.convert_minimax_h3(
        source, fl_path, partition="fl2va", text_only=True
    )
    assert not (fl_path / "transformer_ref").exists()
    fl_manifest = json.loads((fl_path / "h3_manifest.json").read_text())
    assert fl_manifest["partition"] == "fl2va"
    assert fl_manifest["text_only"] is True
    conditioner_weights = mx.load(str(fl_path / "conditioner/model.safetensors"))
    assert not any(key.startswith("vision_tower.") for key in conditioner_weights)
    assert not any("layers.50." in key for key in conditioner_weights)
    assert "language_model.model.norm.weight" not in conditioner_weights
    assert "language_model.lm_head.weight" not in conditioner_weights
    fl_pipeline = h3.load_pipeline(fl_path)
    assert fl_pipeline.partition == "fl2va"
    assert not fl_pipeline.conditioner.has_vision

    fl_copy = tmp_path / "fl-copy"
    h3.convert_minimax_h3(source, fl_copy, partition="fl2va", text_only=True)
    fl_copy_manifest = json.loads((fl_copy / "h3_manifest.json").read_text())
    assert fl_manifest["sha256"] == fl_copy_manifest["sha256"]
    assert fl_report.tensor_counts == fl_copy_manifest["tensor_counts"]

    ref_path = tmp_path / "ref"
    h3.convert_minimax_h3(source, ref_path, partition="ref2va")
    ref_pipeline = h3.load_pipeline(ref_path)
    assert ref_pipeline.partition == "ref2va"
    assert ref_pipeline.conditioner.has_vision
    with pytest.raises(ValueError, match="not 'fl2va'"):
        h3.load_pipeline(ref_path, partition="fl2va")


# Video generation and output handling

video_module = importlib.import_module("mlx_vlm.generate.video_generation")


def _result(path=None, **overrides):
    values = dict(
        frames=mx.zeros((2, 32, 64, 3), mx.uint8),
        audio=mx.zeros((2, 2667), mx.float32),
        fps=24.0,
        sampling_rate=32000,
        seed=7,
        width=64,
        height=32,
        num_frames=2,
        steps=2,
        model="MiniMaxAI/MiniMax-H3",
        family="minimax_h3",
        workflow="ref2va",
        path=path,
    )
    return VideoGenerationResult(**(values | overrides))


def test_video_model_discovery_uses_h3_metadata(tmp_path):
    (tmp_path / "h3_manifest.json").write_text(
        json.dumps({"format": "mlx-vlm-minimax-h3", "partition": "fl2va"})
    )

    assert (
        video_generation_model_class(str(tmp_path))
        is h3_model.MiniMaxH3VideoGenerationModel
    )
    assert is_video_generation_model("MiniMaxAI/MiniMax-H3")
    assert not is_video_generation_model("example/not-a-video-model")


def test_h3_video_adapter_maps_ordered_references_and_outputs():
    progress = Mock()
    generate = Mock(
        return_value=SimpleNamespace(
            video=mx.full((1, 2, 32, 64, 3), 0.5, mx.float32),
            audio=mx.zeros((1, 2, 2667), mx.float32),
            fps=24,
            sampling_rate=32000,
            metadata={"width": 64, "height": 32, "num_frames": 2},
        )
    )
    model = h3_model.MiniMaxH3VideoGenerationModel(
        pipeline=SimpleNamespace(partition="ref2va", generate=generate),
        model_id="synthetic-h3",
        workflow="ref2va",
    )
    result = model.generate(
        VideoGenerationRequest(
            prompt="synthetic",
            seed=7,
            steps=2,
            num_frames=2,
            progress_callback=progress,
            references=(
                VideoReference("image", "first.png"),
                VideoReference("video", "motion.mp4"),
                VideoReference("audio", "sound.wav"),
            ),
        )
    )

    assert [reference.kind for reference in generate.call_args.args[0].references] == [
        "image",
        "video",
        "audio",
    ]
    assert result.frames.shape == (2, 32, 64, 3)
    assert result.frames.dtype == mx.uint8
    assert result.audio.shape == (2, 2667)
    assert result.workflow == "ref2va"
    assert generate.call_args.args[0].progress_callback is progress


def test_generate_video_string_request_routes_extra_kwargs():
    requests = []
    progress_events = []
    model = SimpleNamespace(
        generate=lambda request: (requests.append(request), _result())[1]
    )

    result = video_module.generate_video(
        model,
        "synthetic",
        seed=7,
        num_frames=2,
        progress_callback=lambda *event: progress_events.append(event),
        custom_option="value",
    )

    assert result.seed == 7
    assert requests[0].num_frames == 2
    assert requests[0].extra == {"custom_option": "value"}
    assert progress_events == [("complete", 30, 30, 2)]


def test_video_progress_bar_reports_effective_frames_per_second(monkeypatch):
    bar = Mock(total=2, n=0)

    def update(amount):
        bar.n += amount

    bar.update.side_effect = update
    clock = iter([10.0, 10.0, 11.0, 12.0, 14.0])
    monkeypatch.setattr(video_module, "tqdm", lambda **kwargs: bar)
    monkeypatch.setattr(video_module.time, "perf_counter", lambda: next(clock))

    with video_module._VideoProgressBar(
        steps=2, num_frames=4, disable=False
    ) as progress:
        progress("prepare", 0, 2, 4)
        progress("cache_adaln", 1, 2, 4)
        progress("denoise", 1, 2, 4)
        progress("complete", 2, 2, 4)

    assert bar.n == 2
    assert bar.set_postfix.call_args.args == ({"frames/s": "1.00"},)
    descriptions = [call.args[0] for call in bar.set_description_str.call_args_list]
    assert "Caching AdaLN" in descriptions
    assert "Denoising video" in descriptions
    bar.close.assert_called_once_with()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required for the container smoke test",
)
def test_video_result_muxes_video_and_audio(tmp_path):
    frame_count = 6
    fps = 6
    sampling_rate = 32000
    frames = np.zeros((frame_count, 32, 64, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(frame_count, dtype=np.uint8)[:, None, None] * 30
    timeline = np.arange(sampling_rate, dtype=np.float32) / sampling_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * timeline)
    audio = np.stack([tone, tone])
    result = _result(
        frames=mx.array(frames),
        audio=mx.array(audio),
        fps=fps,
        sampling_rate=sampling_rate,
        num_frames=frame_count,
        model="synthetic",
        family="synthetic",
        workflow="t2va",
    )

    output_path = result.save(tmp_path / "muxed.mp4")
    probe = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {
        stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]
    }
    assert output_path.stat().st_size > 0
    assert stream_types == {"video", "audio"}
