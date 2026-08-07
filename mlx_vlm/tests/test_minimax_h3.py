import json
import math
from dataclasses import asdict, replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from PIL import Image

import mlx_vlm.models.minimax_h3.download as h3_download_module
import mlx_vlm.models.minimax_h3.pipeline as h3_pipeline_module
import mlx_vlm.models.minimax_h3.processing as h3_processing_module
from mlx_vlm.models.minimax_h3 import (
    MiniMaxH3AudioVAE,
    MiniMaxH3AudioVAEConfig,
    MiniMaxH3ConditioningOutput,
    MiniMaxH3GenerationRequest,
    MiniMaxH3Pipeline,
    MiniMaxH3PreparedReference,
    MiniMaxH3Reference,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer,
    MiniMaxH3TransformerConfig,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEConfig,
    align_num_frames,
    audio_latent_num_frames,
    build_fl2va_presentation,
    build_packed_sequence,
    build_ref2va_packed_sequence,
    build_ref2va_presentation,
    build_row_timesteps,
    convert_minimax_h3,
    create_mm_token_type_ids,
    download_plan,
    load_pipeline,
    patchify_video_latents,
    resolve_canvas_size,
    trim_reference_num_frames,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from mlx_vlm.models.minimax_h3.processing import (
    decode_audio,
    decode_video,
    decode_video_soundtrack,
    normalize_visual_vae_pixels,
    prepare_keyframe_image,
    prepare_reference_waveform,
    process_qwen_images,
    process_qwen_videos,
    resample_reference_frames,
    resize_lanczos,
    sample_reference_video_frames,
)
from mlx_vlm.models.qwen3_vl.config import ModelConfig as Qwen3VLModelConfig
from mlx_vlm.models.qwen3_vl.config import TextConfig as Qwen3VLTextConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLForConditionalGeneration


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
    def encode_fl2va(self, prompt, images=None):
        assert prompt == "synthetic"
        assert not images
        return MiniMaxH3ConditioningOutput(
            hidden_states=mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) * 0.01,
            token_tags=mx.array([1, 1], dtype=mx.int32),
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
        )


class _SyntheticFLConditioner:
    def encode_fl2va(self, prompt, images=None):
        assert prompt == "synthetic-fl"
        assert len(images) == 1
        return MiniMaxH3ConditioningOutput(
            hidden_states=mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) * 0.01,
            token_tags=mx.array([1, 1], dtype=mx.int32),
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
        )


class _SyntheticRefConditioner:
    def encode_ref2va(self, prompt, references):
        assert prompt == "synthetic-ref"
        assert len(references) == 1
        return MiniMaxH3ConditioningOutput(
            hidden_states=mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) * 0.01,
            token_tags=mx.array([1, 0], dtype=mx.int32),
            input_ids=mx.array([[1, 2]], dtype=mx.int32),
        )


class _SyntheticRefPipeline(MiniMaxH3Pipeline):
    def _prepare_references(self, references, num_frames):
        del references
        image = (mx.arange(64 * 64 * 3) % 256).astype(mx.uint8).reshape(64, 64, 3)
        return [MiniMaxH3PreparedReference(kind="image", image=image)], num_frames


def _tiny_transformer_config() -> MiniMaxH3TransformerConfig:
    return MiniMaxH3TransformerConfig(
        num_attention_heads=1,
        attention_head_dim=6,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=10,
        in_channels=1,
        audio_in_channels=2,
        patch_size=(1, 1, 1),
        text_dim=5,
        freq_dim=4,
        time_embed_hidden_dim=6,
        time_embed_dim=4,
        rope_freq_dim=1,
    )


def _load_canonical_synthetic_weights(model: MiniMaxH3Transformer) -> None:
    weights = []
    for offset, (key, parameter) in enumerate(sorted(tree_flatten(model.parameters()))):
        size = math.prod(parameter.shape)
        values = ((mx.arange(size) % 29).astype(mx.float32) - 14.0) * 0.005
        values = values + ((offset % 7) - 3) * 0.001
        values = values.reshape(parameter.shape)
        if key.endswith("norm.weight"):
            values = 1.0 + values * 0.1
        weights.append((key, values))
    model.load_weights(weights, strict=True)


def _tiny_video_vae_config() -> MiniMaxH3VideoVAEConfig:
    return MiniMaxH3VideoVAEConfig(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(2, 2),
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        clip_length=17,
        token_drop=3,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )


def _load_canonical_video_vae_weights(model: MiniMaxH3VideoVAE) -> None:
    weights = []
    for offset, (key, parameter) in enumerate(sorted(tree_flatten(model.parameters()))):
        shape = parameter.shape
        torch_shape = (
            (shape[0], shape[-1], shape[1], shape[2], shape[3])
            if parameter.ndim == 5
            else shape
        )
        size = math.prod(shape)
        values = ((mx.arange(size) % 29).astype(mx.float32) - 14.0) * 0.005
        values = values + ((offset % 7) - 3) * 0.001
        values = values.reshape(torch_shape)
        if "norm" in key and key.endswith("weight"):
            values = 1.0 + values * 0.1
        if parameter.ndim == 5:
            values = values.transpose(0, 2, 3, 4, 1)
        weights.append((key, values))
    model.load_weights(weights, strict=True)


def _tiny_audio_vae_config() -> MiniMaxH3AudioVAEConfig:
    return MiniMaxH3AudioVAEConfig(
        encoder_dim=4,
        encoder_rates=(2, 2),
        latent_dim=32,
        latent_channels=8,
        num_attention_heads=2,
        decoder_dim=16,
        decoder_rates=(2, 2),
        decoder_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        sampling_rate=32000,
        latents_mean=(0.0,) * 8,
        latents_std=(1.0,) * 8,
    )


def _canonical_audio_vae_source_weights(
    model: MiniMaxH3AudioVAE,
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


def _load_canonical_audio_vae_weights(model: MiniMaxH3AudioVAE) -> None:
    source = _canonical_audio_vae_source_weights(model)
    model.load_weights(sorted(model.sanitize(source).items()), strict=True)


def test_geometry_and_patch_roundtrip():
    assert resolve_canvas_size(16, 9) == (768, 1344)
    assert resolve_canvas_size(9, 16) == (1344, 768)
    assert align_num_frames(120) == 124
    assert video_latent_num_frames(124) == 37
    assert audio_latent_num_frames(124) == 207
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_canvas_size(5, 1)

    latents = mx.arange(1 * 2 * 3 * 4 * 6).reshape(1, 2, 3, 4, 6)
    rows = patchify_video_latents(latents, (1, 2, 2))
    restored = unpatchify_video_tokens(rows, 3, 4, 6, 2, (1, 2, 2))
    assert rows.shape == (18, 8)
    assert mx.array_equal(latents, restored).item()


def test_fl2va_packing_matches_diffusers_golden():
    layout = build_packed_sequence(
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

    timesteps, timestep_indices = build_row_timesteps(
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
        MiniMaxH3PreparedReference(
            kind="image",
            num_latent_frames=1,
            latent_height=4,
            latent_width=6,
        ),
        MiniMaxH3PreparedReference(
            kind="video",
            has_audio=True,
            num_latent_frames=3,
            latent_height=4,
            latent_width=4,
            num_audio_latents=2,
        ),
        MiniMaxH3PreparedReference(
            kind="audio",
            has_audio=True,
            num_audio_latents=3,
        ),
    ]
    layout = build_ref2va_packed_sequence(
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
    assert layout.audio_indices.tolist() == [
        *range(11, 15),
        *range(27, 41),
    ]
    positions = layout.position_ids.tolist()
    assert positions[5][0] == 5.0
    assert positions[11][0] == 6.0
    assert positions[15][0] == 6.0
    assert positions[27][0] == 21.0
    assert positions[33][0] == 24.0
    assert positions[41][0] == 24.0


def test_scheduler_matches_diffusers_golden():
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(5)
    np.testing.assert_allclose(
        scheduler.sigmas.tolist(),
        [1.0, 0.9729729891, 0.9230769277, 0.8000000119, 0.0],
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        scheduler.timesteps.tolist(),
        [0.0, 0.0270270109, 0.0769230723, 0.1999999881],
        rtol=0.0,
        atol=5e-10,
    )

    sample = mx.array([[0.1, -0.2], [0.3, 0.4]], dtype=mx.float32)
    velocity = mx.array([[0.5, 0.6], [-0.7, 0.8]], dtype=mx.float32)
    output = scheduler.step(velocity, scheduler.timesteps[0], sample)
    np.testing.assert_allclose(
        np.array(output),
        np.array(
            [[0.11351351, -0.1837838], [0.2810811, 0.42162162]],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-8,
    )


def test_mlx_lanczos_matches_pillow_reference():
    pixels = np.random.default_rng(7).integers(
        0,
        256,
        size=(13, 15, 3),
        dtype=np.uint8,
    )
    for height, width in ((7, 9), (21, 26), (13, 23)):
        expected = np.asarray(
            Image.fromarray(pixels).resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
        )
        actual = resize_lanczos(mx.array(pixels), height, width)
        np.testing.assert_array_equal(np.array(actual), expected)

    stretched = prepare_keyframe_image(
        mx.array(pixels),
        8,
        10,
        stretch=True,
    )
    covered = prepare_keyframe_image(
        mx.array(pixels),
        8,
        10,
        stretch=False,
    )
    assert stretched.shape == (8, 10, 3)
    assert covered.shape == (8, 10, 3)


def test_mlx_reference_frame_and_audio_processing():
    frames = mx.arange(10, dtype=mx.uint8).reshape(10, 1, 1, 1)
    frames = mx.repeat(frames, 3, axis=-1)
    upsampled = resample_reference_frames(frames, 10.0)
    downsampled = resample_reference_frames(frames, 30.0)
    assert upsampled[:, 0, 0, 0].tolist() == [
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        6,
        7,
        7,
        8,
        8,
        8,
        9,
        9,
    ]
    assert downsampled[:, 0, 0, 0].tolist() == [0, 1, 3, 4, 5, 6, 8, 9]

    sampled, timestamps = sample_reference_video_frames(mx.repeat(frames, 4, axis=0))
    assert sampled[:, 0, 0, 0].tolist() == [0, 3, 6, 9]
    assert timestamps == [0.25, 1.25]

    waveform = mx.arange(17, dtype=mx.float32)[None] / 17.0
    resampled = prepare_reference_waveform(waveform, 16000, 32000, 1.0)
    assert resampled.shape == (2, 34)
    np.testing.assert_allclose(
        np.array(resampled[0, :5]),
        [0.00016416, 0.02440057, 0.05869471, 0.09065638, 0.11777476],
        rtol=2e-4,
        atol=2e-6,
    )
    np.testing.assert_array_equal(np.array(resampled[0]), np.array(resampled[1]))

    normalized = normalize_visual_vae_pixels(mx.zeros((2, 3, 4, 3), mx.uint8))
    assert normalized.shape == (1, 3, 2, 3, 4)
    np.testing.assert_allclose(
        np.array(normalized[0, :, 0, 0, 0]),
        -np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]),
        rtol=1e-6,
    )

    decoded_video, decoded_fps = decode_video(np.zeros((3, 3, 4, 5), dtype=np.uint8))
    assert decoded_video.shape == (3, 4, 5, 3)
    assert decoded_fps is None
    decoded_audio, decoded_rate = decode_audio(np.zeros((17, 2), dtype=np.float32))
    assert decoded_audio.shape == (2, 17)
    assert decoded_rate is None


def test_video_soundtrack_decode_moves_pcm_directly_into_mlx(monkeypatch):
    pcm = np.array([[0.25, -0.5], [0.75, 0.125]], dtype="<f4")

    def fake_run(command, **kwargs):
        del kwargs
        if command[0].endswith("ffprobe"):
            return type(
                "Probe",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {"streams": [{"sample_rate": "48000", "channels": 2}]}
                    ),
                },
            )()
        assert command[0].endswith("ffmpeg")
        return type(
            "Decode",
            (),
            {"returncode": 0, "stdout": pcm.tobytes(), "stderr": b""},
        )()

    monkeypatch.setattr(
        h3_processing_module.shutil, "which", lambda name: f"/tools/{name}"
    )
    monkeypatch.setattr(h3_processing_module.subprocess, "run", fake_run)
    waveform, sample_rate = decode_video_soundtrack("reference.mp4")
    assert sample_rate == 48000
    np.testing.assert_array_equal(np.array(waveform), pcm.T)
    assert decode_video_soundtrack(mx.zeros((2, 4, 4, 3), mx.uint8)) is None


def test_qwen_processors_match_transformers_synthetic_golden():
    image = (
        np.arange(256 * 256 * 3, dtype=np.uint32).reshape(256, 256, 3) % 256
    ).astype(np.uint8)
    video = (
        np.arange(2 * 64 * 64 * 3, dtype=np.uint32).reshape(2, 64, 64, 3) % 256
    ).astype(np.uint8)
    image_patches, image_grid = process_qwen_images([mx.array(image)])
    video_patches, video_grid = process_qwen_videos([mx.array(video)])

    assert image_patches.shape == (256, 1536)
    assert video_patches.shape == (16, 1536)
    assert image_grid.tolist() == [[1, 16, 16]]
    assert video_grid.tolist() == [[1, 4, 4]]
    expected_image = np.array(
        [
            -1.0,
            -0.9764705896,
            -0.9529411793,
            -0.9294117689,
            -0.9058823586,
            -0.8823529482,
            -0.8588235378,
            -0.8352941275,
        ],
        dtype=np.float32,
    )
    expected_video = np.array(
        [
            -1.0,
            -0.9764705896,
            -0.9529411793,
            -0.9294117689,
            -0.9058823586,
            -0.8823529482,
            -0.8588235378,
            -0.8352941275,
            -0.8117647171,
            -0.7882353067,
            -0.7647058964,
            -0.7411764860,
            -0.7176470757,
            -0.6941176653,
            -0.6705882549,
            -0.6470588446,
            0.5058823824,
            0.5294117928,
            0.5529412031,
            0.5764706135,
            0.6000000238,
            0.6235294342,
            0.6470588446,
            0.6705882549,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.array(image_patches[0, :8]), expected_image)
    np.testing.assert_allclose(np.array(video_patches[0, :24]), expected_video)


def test_prompt_presentations_match_diffusers_ordering():
    tokenizer = _SyntheticTokenizer()
    fl_ids, fl_tags = build_fl2va_presentation(tokenizer, "go", [2])
    label_length = len("<Picture 1>: ")
    assert fl_ids[label_length : label_length + 4] == [900, 901, 901, 903]
    assert fl_tags[label_length : label_length + 4] == [0, 0, 0, 0]
    assert create_mm_token_type_ids(fl_ids, tokenizer)[
        label_length : label_length + 4
    ] == [0, 1, 1, 0]

    references = [
        MiniMaxH3PreparedReference(kind="image"),
        MiniMaxH3PreparedReference(
            kind="video",
            has_audio=True,
            block_timestamps=[0.25, 1.25],
        ),
        MiniMaxH3PreparedReference(kind="audio", has_audio=True),
    ]
    ids, tags = build_ref2va_presentation(
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
    assert trim_reference_num_frames(5) == 22
    assert trim_reference_num_frames(39) == 39


def test_qwen_conditioner_layer_stop_is_pre_final_norm():
    config = Qwen3VLTextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=3,
        hidden_size=12,
        intermediate_size=16,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        head_dim=6,
        rope_theta=10000.0,
        max_position_embeddings=32,
        rope_scaling={"type": "default", "mrope_section": [1, 1, 1]},
    )
    model = Qwen3VLModel(config)
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    with mx.stream(mx.cpu):
        layer_two = model(
            input_ids,
            stop_after_layer=2,
            apply_final_norm=False,
        )
        layer_two_normalized = model(
            input_ids,
            stop_after_layer=2,
            apply_final_norm=True,
        )
        full_pre_norm = model(input_ids, apply_final_norm=False)
        full = model(input_ids)
        mx.eval(layer_two, layer_two_normalized, full_pre_norm, full)
    np.testing.assert_allclose(
        np.array(layer_two_normalized),
        np.array(model.norm(layer_two)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(full),
        np.array(model.norm(full_pre_norm)),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(np.array(layer_two), np.array(full_pre_norm))


def test_qwen_layer_two_matches_transformers_synthetic_golden():
    config = Qwen3VLTextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=3,
        hidden_size=12,
        intermediate_size=16,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        head_dim=6,
        rope_theta=10000.0,
        max_position_embeddings=32,
        rope_scaling={"type": "default", "mrope_section": [1, 1, 1]},
    )
    model = Qwen3VLModel(config)
    weights = []
    for offset, (key, parameter) in enumerate(sorted(tree_flatten(model.parameters()))):
        values = (
            (mx.arange(math.prod(parameter.shape)) % 29).astype(mx.float32) - 14.0
        ) * 0.005
        values = values + ((offset % 7) - 3) * 0.001
        values = values.reshape(parameter.shape)
        if key.endswith("norm.weight") or "layernorm.weight" in key:
            values = 1.0 + values * 0.1
        weights.append((key, values))
    model.load_weights(weights, strict=True)

    with mx.stream(mx.cpu):
        hidden_states = model(
            mx.array([[1, 2, 3, 4]], dtype=mx.int32),
            stop_after_layer=2,
            apply_final_norm=False,
        )
        mx.eval(hidden_states)
    expected = np.array(
        [
            -0.0008692872,
            0.0143544516,
            -0.0518668033,
            0.0281886049,
            -0.0888264850,
            0.0361017920,
            0.0340734236,
            -0.0342860520,
            0.0559054166,
            -0.0485159159,
            0.0682821869,
            0.0416498594,
            -0.0122771589,
            0.0475819260,
            0.0924062729,
            0.0631933957,
            0.1447156221,
            -0.0848623663,
            -0.0704205111,
            0.0460798815,
            -0.0490613580,
            -0.0370092504,
            -0.0613816194,
            -0.0795182437,
            -0.0960728675,
            -0.0168740973,
            -0.0908068568,
            -0.0215693936,
            -0.1027236059,
            -0.0146589465,
            -0.0097286785,
            -0.0250200611,
            0.0191809162,
            -0.0044598971,
            0.0220890567,
            -0.0287228636,
            -0.0046169655,
            0.0754216537,
            0.0689876825,
            0.0728251711,
            0.0522902757,
            0.0766496062,
            0.0754699782,
            -0.0322860964,
            0.1001782492,
            0.0107320528,
            -0.0313899368,
            -0.0808366984,
        ],
        dtype=np.float32,
    ).reshape(1, 4, 12)
    np.testing.assert_allclose(np.array(hidden_states), expected, rtol=1e-6, atol=2e-8)


def test_qwen_mixed_image_video_deepstack_keeps_token_order():
    config, _ = _tiny_qwen_source()
    config["text_config"]["num_hidden_layers"] = 1
    model = Qwen3VLForConditionalGeneration(Qwen3VLModelConfig.from_dict(config))
    image_layers = [mx.array([[1.0, 1.5], [2.0, 2.5]])]
    video_layers = [mx.array([[3.0, 3.5]])]
    merged = model._merge_deepstack_features(
        mx.array([[29, 30, 29]], dtype=mx.int32),
        image_layers,
        video_layers,
    )
    np.testing.assert_array_equal(
        np.array(merged),
        np.array([[[1.0, 1.5], [3.0, 3.5], [2.0, 2.5]]]),
    )


def test_tiny_transformer_matches_diffusers_synthetic_golden():
    config = _tiny_transformer_config()
    model = MiniMaxH3Transformer(config)
    _load_canonical_synthetic_weights(model)
    layout = build_packed_sequence(
        mx.array([1, 1]),
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=config.patch_size,
    )
    timesteps, timestep_indices = build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    video = (mx.arange(4, dtype=mx.float32).reshape(1, 4, 1) - 1.5) * 0.1
    audio = (mx.arange(4, dtype=mx.float32).reshape(1, 2, 2) - 1.5) * 0.1
    text = (mx.arange(10, dtype=mx.float32).reshape(1, 2, 5) - 4.5) * 0.05

    # CPU matmul provides the strict implementation-parity signal. MLX's
    # Metal float32 matmul intentionally uses lower-precision fast math.
    with mx.stream(mx.cpu):
        output = model(
            video,
            audio,
            text,
            timesteps,
            timestep_indices,
            layout.token_tags,
            layout.position_ids,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
        )
        mx.eval(output.sample, output.audio_sample)

    expected_video = np.array(
        [[[0.35690272], [0.35708505], [0.35721874], [0.35731968]]],
        dtype=np.float32,
    )
    expected_audio = np.array(
        [
            [
                [0.3702623, 0.048529133],
                [0.37101823, 0.056271546],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.array(output.sample),
        expected_video,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.array(output.audio_sample),
        expected_audio,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_tiny_transformer_adaln_cache_is_bitwise_identical_after_drop(dtype):
    config = _tiny_transformer_config()
    model = MiniMaxH3Transformer(config)
    _load_canonical_synthetic_weights(model)
    model.set_dtype(dtype)
    layout = build_packed_sequence(
        mx.array([1, 1]),
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=config.patch_size,
    )
    timesteps, timestep_indices = build_row_timesteps(
        layout,
        video_timestep=0.25,
        audio_timestep=0.5,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    args = (
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

    with mx.stream(mx.cpu):
        live = model(*args)
        mx.eval(live.sample, live.audio_sample)
        cache = model.build_adaln_cache(timesteps)
        cached = model(*args, cache)
        mx.eval(cached.sample, cached.audio_sample)
        other = MiniMaxH3Transformer(config)
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


def test_tiny_video_vae_matches_diffusers_synthetic_golden():
    model = MiniMaxH3VideoVAE(_tiny_video_vae_config())
    _load_canonical_video_vae_weights(model)
    model.disable_tiling()
    pixels = ((mx.arange(1 * 3 * 22 * 8 * 8, dtype=mx.float32) % 31) - 15.0).reshape(
        1, 3, 22, 8, 8
    ) * 0.01
    with mx.stream(mx.cpu):
        latents = model.encode(pixels).mode()
        decoded = model.decode(latents).sample
        mx.eval(latents, decoded)

    assert latents.shape == (1, 4, 7, 2, 2)
    assert decoded.shape == pixels.shape
    expected_latents = np.array(
        [
            -0.0873808935,
            -0.0868481994,
            -0.0734840408,
            -0.0774308816,
            -0.0855999961,
            -0.1233575493,
            -0.0818316489,
            -0.1153013855,
        ],
        dtype=np.float32,
    )
    expected_pixels = np.array(
        [
            0.0168855451,
            0.3830011189,
            -0.3451227248,
            0.3849773705,
            0.0160088632,
            0.3825895190,
            -0.3445619047,
            0.3845651448,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.array(latents.flatten()[:8]), expected_latents, rtol=2e-6, atol=2e-7
    )
    np.testing.assert_allclose(
        np.array(decoded.flatten()[:8]), expected_pixels, rtol=2e-6, atol=2e-7
    )

    model.enable_tiling(8, 8, 4, 4)
    assert model._split_tiles(12, 8, 4) == ([0, 4], [8, 8], [4])


def test_tiny_audio_vae_matches_diffusers_synthetic_golden():
    model = MiniMaxH3AudioVAE(_tiny_audio_vae_config())
    _load_canonical_audio_vae_weights(model)
    waveform = (
        (mx.arange(2 * 33, dtype=mx.float32).reshape(2, 1, 33) % 23) - 11.0
    ) * 0.02
    with mx.stream(mx.cpu):
        posterior = model.encode(waveform)
        latents = posterior.mode()
        decoded = model.decode(latents).sample
        mx.eval(latents, posterior.logs, decoded)

    assert latents.shape == (2, 8, 9)
    assert decoded.shape == (2, 1, 36)
    expected_mean = np.array(
        [
            -0.0414157696,
            -0.0378905796,
            -0.0380623564,
            -0.0380322821,
            -0.0377460718,
            -0.0376516134,
            -0.0376482755,
            -0.0384906232,
        ],
        dtype=np.float32,
    )
    expected_logs = np.array(
        [
            -0.0412009582,
            -0.0375841446,
            -0.0377684683,
            -0.0377356336,
            -0.0374392867,
            -0.0373374224,
            -0.0373311266,
            -0.0381926671,
        ],
        dtype=np.float32,
    )
    expected_waveform = np.array(
        [
            -0.0065163295,
            -0.0028203637,
            0.0028824992,
            0.0113496268,
            0.0075704525,
            0.0072223009,
            0.0065093203,
            0.0071694441,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.array(latents.flatten()[:8]), expected_mean, rtol=1e-6, atol=3e-8
    )
    np.testing.assert_allclose(
        np.array(posterior.logs.flatten()[:8]),
        expected_logs,
        rtol=1e-6,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        np.array(decoded.flatten()[:8]),
        expected_waveform,
        rtol=2e-6,
        atol=2e-8,
    )


def _tiny_pipeline_modules():
    transformer = MiniMaxH3Transformer(_tiny_transformer_config())
    _load_canonical_synthetic_weights(transformer)
    video_vae = MiniMaxH3VideoVAE(
        MiniMaxH3VideoVAEConfig(
            in_channels=3,
            out_channels=3,
            latent_channels=1,
            block_out_channels=(8, 8, 8, 8, 8),
            layers_per_block=1,
            spatial_downsample_factors=(2, 2, 2, 2, 2),
            temporal_downsample_factors=(2, 2, 1, 1, 1),
            norm_num_groups=8,
            decoder_num_layers=1,
            decoder_num_attention_heads=1,
            decoder_attention_head_dim=8,
            decoder_num_register_tokens=1,
            decoder_ffn_mult=1,
            latents_mean=(0.0,),
            latents_std=(1.0,),
        )
    )
    audio_vae = MiniMaxH3AudioVAE(
        MiniMaxH3AudioVAEConfig(
            encoder_dim=4,
            encoder_rates=(2, 2),
            latent_dim=8,
            latent_channels=2,
            num_attention_heads=1,
            decoder_dim=8,
            decoder_rates=(2, 2),
            decoder_kernel_sizes=(4, 4),
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1,),),
            latents_mean=(0.0, 0.0),
            latents_std=(1.0, 1.0),
        )
    )
    return transformer, video_vae, audio_vae


def test_video_path_automatically_contributes_its_soundtrack(monkeypatch):
    transformer, video_vae, audio_vae = _tiny_pipeline_modules()
    pipeline = MiniMaxH3Pipeline(
        transformer=transformer,
        conditioner=_SyntheticRefConditioner(),
        video_vae=video_vae,
        audio_vae=audio_vae,
        partition="ref2va",
    )
    frames = mx.zeros((5, 8, 8, 3), mx.uint8)
    soundtrack = mx.zeros((2, 100), mx.float32)
    monkeypatch.setattr(
        h3_pipeline_module,
        "decode_video_soundtrack",
        lambda path: (soundtrack, 20),
    )
    monkeypatch.setattr(h3_pipeline_module, "decode_video", lambda path: (frames, 24.0))
    monkeypatch.setattr(
        h3_pipeline_module,
        "prepare_reference_frames",
        lambda values, num_frames: values[:num_frames],
    )
    monkeypatch.setattr(
        h3_pipeline_module,
        "prepare_reference_waveform",
        lambda waveform, *args: waveform,
    )

    prepared, num_frames = pipeline._prepare_references(
        [MiniMaxH3Reference(video="reference.mp4")], None
    )
    assert num_frames == 124
    assert len(prepared) == 1
    assert prepared[0].kind == "video"
    assert prepared[0].has_audio
    assert prepared[0].waveform.shape == (2, 100)


def test_tiny_t2va_pipeline_runs_joint_denoise_to_latents():
    transformer, video_vae, audio_vae = _tiny_pipeline_modules()
    pipeline = MiniMaxH3Pipeline(
        transformer=transformer,
        conditioner=_SyntheticConditioner(),
        video_vae=video_vae,
        audio_vae=audio_vae,
    )
    num_video_latents = video_latent_num_frames(124)
    num_audio_latents = audio_latent_num_frames(124)
    progress_events = []
    request = MiniMaxH3GenerationRequest(
        prompt="synthetic",
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=2,
        output_type="latent",
        latents=mx.zeros((1, 1, num_video_latents, 1, 1), mx.float32),
        audio_latents=mx.zeros((2, 2, num_audio_latents), mx.float32),
        progress_callback=lambda *event: progress_events.append(event),
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


def test_tiny_fl2va_cached_trajectory_is_bitwise_identical():
    transformer, video_vae, audio_vae = _tiny_pipeline_modules()
    pipeline = MiniMaxH3Pipeline(
        transformer=transformer,
        conditioner=_SyntheticFLConditioner(),
        video_vae=video_vae,
        audio_vae=audio_vae,
    )
    num_video_latents = video_latent_num_frames(124)
    num_audio_latents = audio_latent_num_frames(124)
    request = MiniMaxH3GenerationRequest(
        prompt="synthetic-fl",
        image=mx.zeros((64, 64, 3), mx.uint8),
        height=64,
        width=64,
        num_frames=124,
        num_inference_steps=3,
        output_type="latent",
        latents=mx.zeros((1, 1, num_video_latents, 2, 2), mx.float32),
        audio_latents=mx.zeros((2, 2, num_audio_latents), mx.float32),
    )
    with mx.stream(mx.cpu):
        live = pipeline.generate(
            replace(request, cache_adaln=False, drop_adaln_weights=False)
        )
        mx.eval(live.video, live.audio)
        cached = pipeline.generate(request)
        mx.eval(cached.video, cached.audio)

    assert mx.array_equal(live.video, cached.video).item()
    assert mx.array_equal(live.audio, cached.audio).item()
    assert cached.metadata["adaln_weights_dropped"]


def test_tiny_ref2va_pipeline_runs_conditioned_joint_denoise():
    _, video_vae, audio_vae = _tiny_pipeline_modules()
    transformer = MiniMaxH3Transformer(
        replace(_tiny_transformer_config(), patch_size=(1, 2, 2))
    )
    _load_canonical_synthetic_weights(transformer)
    pipeline = _SyntheticRefPipeline(
        transformer=transformer,
        conditioner=_SyntheticRefConditioner(),
        video_vae=video_vae,
        audio_vae=audio_vae,
        partition="ref2va",
    )
    num_video_latents = video_latent_num_frames(124)
    num_audio_latents = audio_latent_num_frames(124)
    request = MiniMaxH3GenerationRequest(
        prompt="synthetic-ref",
        references=[object()],
        height=64,
        width=64,
        num_frames=124,
        num_inference_steps=3,
        output_type="latent",
        latents=mx.zeros((1, 1, num_video_latents, 2, 2), mx.float32),
        audio_latents=mx.zeros((2, 2, num_audio_latents), mx.float32),
    )
    with mx.stream(mx.cpu):
        live = pipeline.generate(
            replace(request, cache_adaln=False, drop_adaln_weights=False)
        )
        mx.eval(live.video, live.audio)
        output = pipeline.generate(request)
        mx.eval(output.video, output.audio)
    assert output.video.shape == (1, 1, num_video_latents, 2, 2)
    assert output.audio.shape == (2, 2, num_audio_latents)
    assert output.metadata["partition"] == "ref2va"
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
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    name = "diffusion_pytorch_model.safetensors"
    mx.save_safetensors(str(path / name), dict(sorted(weights.items())))
    index = {
        "metadata": {"total_size": sum(value.nbytes for value in weights.values())},
        "weight_map": {key: name for key in sorted(weights)},
    }
    (path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index)
    )


def _tiny_qwen_source():
    config = {
        "model_type": "qwen3_vl",
        "image_token_id": 29,
        "video_token_id": 30,
        "vision_start_token_id": 28,
        "vision_end_token_id": 27,
        "text_config": {
            "model_type": "qwen3_vl",
            "num_hidden_layers": 51,
            "hidden_size": 12,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "head_dim": 6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 32,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 1],
            },
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 12,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "fullatt_block_indexes": [0],
            "deepstack_visual_indexes": [],
        },
    }
    model = Qwen3VLForConditionalGeneration(Qwen3VLModelConfig.from_dict(config))
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
    transformer_config = _tiny_transformer_config()
    for name in ("transformer", "transformer_ref"):
        transformer = MiniMaxH3Transformer(transformer_config)
        _write_official_component(
            root / name,
            asdict(transformer_config),
            _canonical_parameter_weights(transformer),
        )

    video_config = _tiny_video_vae_config()
    video_vae = MiniMaxH3VideoVAE(video_config)
    video_weights = {}
    for key, value in _canonical_parameter_weights(video_vae).items():
        video_weights[key] = (
            value.transpose(0, 4, 1, 2, 3) if value.ndim == 5 else value
        )
    _write_official_component(root / "vae", asdict(video_config), video_weights)

    audio_config = _tiny_audio_vae_config()
    audio_vae = MiniMaxH3AudioVAE(audio_config)
    _write_official_component(
        root / "audio_vae",
        asdict(audio_config),
        _canonical_audio_vae_source_weights(audio_vae),
    )

    qwen_config, qwen_weights = _tiny_qwen_source()
    _write_official_component(root / "text_encoder", qwen_config, qwen_weights)
    _write_tiny_tokenizer(root / "tokenizer")
    (root / "LICENSE").write_text("synthetic license fixture\n")


def test_official_layout_conversion_and_strict_reload(tmp_path, monkeypatch):
    source = tmp_path / "official"
    _write_tiny_official_h3(source)

    t2_plan = download_plan("t2va")
    fl_plan = download_plan("fl2va")
    ref_plan = download_plan("ref2va")
    assert t2_plan.revision == "b3c7290e66afdf293bef3b9077b7a266ef421f34"
    assert download_plan("t2va", repo_id="test-org/minimax-h3").revision is None
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
        download_plan()
    with pytest.raises(ValueError, match="workflow must be"):
        download_plan("unknown")

    download_calls = []

    def fake_snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(source)

    monkeypatch.setattr(h3_download_module, "snapshot_download", fake_snapshot_download)
    remote_pipeline = load_pipeline(
        "test-org/minimax-h3",
        workflow="t2va",
        text_only=True,
        revision="test-revision",
    )
    assert remote_pipeline.partition == "fl2va"
    assert download_calls[0]["revision"] == "test-revision"
    assert "transformer/**" in download_calls[0]["allow_patterns"]
    assert "transformer_ref/**" not in download_calls[0]["allow_patterns"]

    ref_download = h3_download_module.download_model(
        workflow="ref2va",
        repo_id="test-org/minimax-h3",
        revision="test-revision",
    )
    assert ref_download == source
    assert "transformer_ref/**" in download_calls[1]["allow_patterns"]
    assert "transformer/**" not in download_calls[1]["allow_patterns"]

    with pytest.raises(ValueError, match="uses the 'fl2va' partition"):
        load_pipeline(source, workflow="t2va", partition="ref2va")

    official_pipeline = load_pipeline(
        source,
        partition="fl2va",
        text_only=True,
    )
    assert official_pipeline.partition == "fl2va"

    dry_run = convert_minimax_h3(
        source,
        tmp_path / "unused",
        partition="fl2va",
        text_only=True,
        dry_run=True,
    )
    assert dry_run.dry_run
    assert dry_run.source_bytes > 0
    assert dry_run.converted_bytes > 0
    assert dry_run.tensor_counts["conditioner"] > 0
    assert not dry_run.destination.exists()

    fl_path = tmp_path / "fl"
    fl_report = convert_minimax_h3(
        source,
        fl_path,
        partition="fl2va",
        text_only=True,
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
    fl_pipeline = load_pipeline(fl_path)
    assert fl_pipeline.partition == "fl2va"
    assert not fl_pipeline.conditioner.has_vision

    fl_copy = tmp_path / "fl-copy"
    convert_minimax_h3(
        source,
        fl_copy,
        partition="fl2va",
        text_only=True,
    )
    fl_copy_manifest = json.loads((fl_copy / "h3_manifest.json").read_text())
    assert fl_manifest["sha256"] == fl_copy_manifest["sha256"]
    assert fl_report.tensor_counts == fl_copy_manifest["tensor_counts"]

    ref_path = tmp_path / "ref"
    convert_minimax_h3(source, ref_path, partition="ref2va")
    ref_pipeline = load_pipeline(ref_path)
    assert ref_pipeline.partition == "ref2va"
    assert ref_pipeline.conditioner.has_vision
    with pytest.raises(ValueError, match="not 'fl2va'"):
        load_pipeline(ref_path, partition="fl2va")
