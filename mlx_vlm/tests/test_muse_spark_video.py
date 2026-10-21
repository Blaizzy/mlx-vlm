"""Muse Spark temporal patches, timestamp prompts, and native video inference."""

import json
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.generate.video import processor_handles_video, resolve_video_inputs
from mlx_vlm.models.muse_spark import MuseSparkVideoProcessor
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.tests.test_muse_spark import (
    _CASE,
    create_model,
    create_processor,
    hf_checkpoint,
)
from mlx_vlm.utils import VideoMetadata, prepare_inputs, resolve_video_sampling


def frames(count=3, height=8, width=12):
    return np.random.default_rng(11).integers(
        0, 256, (count, 3, height, width), dtype=np.uint8
    )


@pytest.mark.parametrize(
    "total,source_fps,fps,cap,expected",
    [
        (300, 30, 2, None, np.linspace(0, 299, 20).astype(int)),
        (300, 30, 2, 5, [0, 99, 199, 299]),
        (10, 30, 2, None, [0, 9]),
        (1, 30, 2, None, [0]),
        (300, 30, 2, 1, [0]),
        (10, None, 24, None, np.arange(10)),
    ],
)
def test_uniform_sampling(total, source_fps, fps, cap, expected):
    processor = MuseSparkVideoProcessor()
    metadata = VideoMetadata(total, source_fps, [])
    np.testing.assert_array_equal(
        processor.sample_frames(metadata, fps=fps, max_frames=cap), expected
    )


def test_release_sampling_defaults_and_overrides():
    processor = create_processor()
    sampling = resolve_video_sampling(processor, {})
    assert (sampling.fps, sampling.max_frames, sampling.frame_factor) == (2.0, 600, 1)
    override = {"fps": 1.5, "max_frames": 5, "nframes": 3}
    sampling = resolve_video_sampling(processor, override)
    assert (sampling.fps, sampling.max_frames, sampling.nframes) == (1.5, 5, 3)
    assert override == {}


def test_temporal_patch_layout_and_last_frame_padding():
    processor = create_processor().video_processor
    video = frames()
    outputs = processor(video, fps=4)
    assert outputs["video_grid_thw"].tolist() == [[2, 4, 6]]
    # Independent reference: visit each group, spatial patch, time, and channel.
    pixels = video.astype(np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    pixels = np.concatenate([pixels, pixels[-1:]])
    expected = np.stack(
        [
            pixels[t : t + 2, :, row : row + 2, col : col + 2].flatten()
            for t in (0, 2)
            for row in range(0, 8, 2)
            for col in range(0, 12, 2)
        ]
    )
    np.testing.assert_array_equal(outputs["pixel_values_videos"], expected)
    assert outputs["timestamps"] == [[0.0, 0.5]]
    channels_last = processor(video.transpose(0, 2, 3, 1), fps=4)
    pil_frames = processor(
        [Image.fromarray(f.transpose(1, 2, 0)) for f in video], fps=4
    )
    for other in (channels_last, pil_frames):
        np.testing.assert_array_equal(other["pixel_values_videos"], expected)


def test_video_resize_obeys_spatial_token_budget():
    processor = MuseSparkVideoProcessor(
        patch_size=2, temporal_patch_size=2, merge_size=2, max_video_frame_tokens=6
    )
    result = processor(frames(1, height=40, width=80))
    temporal, height, width = result["video_grid_thw"][0]
    assert temporal == 1 and height % 2 == width % 2 == 0
    assert height * width // 4 <= 6


def test_source_timestamps_and_multiple_clip_prompt_expansion():
    processor = create_processor()
    videos = [frames(), frames(1, height=4, width=8)]
    metadata = [
        VideoMetadata(120, 30, [0, 30, 90]),
        {"timestamps": [7.25]},
    ]
    # Inspect the actual string passed to the tokenizer, before WordLevel
    # maps the timestamp text to unknown tokens in this tiny test vocabulary.
    with patch.object(processor, "_expand", wraps=processor._expand) as expand:
        outputs = processor(
            ["first <|video|>", "second <|video|>"],
            videos=videos,
            video_metadata=metadata,
            fps=[2, 1],
        )
    assert expand.call_args.args[2] == [
        "<|vid_start|>Time: 0.00s"
        + "<|video|>" * 6
        + "<|vid_frame_separator|>Time: 3.00s"
        + "<|video|>" * 6
        + "<|vid_end|>",
        "<|vid_start|>Time: 7.25s" + "<|video|>" * 2 + "<|vid_end|>",
    ]
    assert outputs["video_grid_thw"].tolist() == [[2, 4, 6], [1, 2, 4]]
    assert int((outputs["input_ids"][0] == 253).sum()) == 12
    assert int((outputs["input_ids"][1] == 253).sum()) == 2
    assert "timestamps" not in outputs
    # Metadata with missing source fps can use the supplied rate.
    output = processor.video_processor(
        videos[0], video_metadata=[VideoMetadata(3, None, [0, 1, 2])], fps=2
    )
    assert output["timestamps"] == [[0, 1]]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"videos": []}, "no clips"),
        ({"videos": np.zeros((0, 3, 8, 8), np.uint8)}, "nonempty"),
        ({"videos": np.zeros((2, 3, 8, 8), np.float32)}, "uint8"),
        ({"video_metadata": []}, "one metadata"),
        ({"fps": [1, 2]}, "one metadata"),
        ({"fps": 0}, "fps"),
        ({"video_metadata": [{"timestamps": [0]}]}, "frame count"),
        ({"video_metadata": [{"timestamps": [0, 2, 1]}]}, "ordered"),
        ({"video_metadata": [{"timestamps": [0, float("nan"), 2]}]}, "finite"),
    ],
)
def test_invalid_video_inputs(kwargs, message):
    inputs = {"videos": frames(), **kwargs}
    with pytest.raises(ValueError, match=message):
        create_processor().video_processor(**inputs)


def test_temporal_groups_and_clips_attend_independently():
    model = create_model(vision=True)
    processor = create_processor().video_processor
    video = frames(4, height=12, width=16)
    # Identical temporal slots must reproduce the still-image vision path.
    video[1] = video[0]
    inputs = processor(video)
    features = model.encode_video(
        mx.array(inputs["pixel_values_videos"]), mx.array(inputs["video_grid_thw"])
    )
    image = mx.array((video[0].astype(np.float32) / 255 - 0.5) / 0.5)
    assert features.shape == (24, 64)
    assert mx.allclose(features[:12], model.encode_image(image), atol=1e-5)
    # A second temporal slot affects its group, with no attention leakage to
    # another temporal group or clip (even through full-attention layers).
    video[1] = 255 - video[1]
    changed = processor([video, frames(1, height=4, width=8)])
    changed_features = model.encode_video(
        mx.array(changed["pixel_values_videos"]), mx.array(changed["video_grid_thw"])
    )
    assert changed_features.shape == (26, 64)
    assert not mx.allclose(changed_features[:12], features[:12], atol=1e-4)
    assert mx.allclose(changed_features[12:24], features[12:], atol=1e-5)


@pytest.mark.parametrize(
    "grid,message",
    [
        ([[1, 2, 2]], "patch count"),
        ([[1, 3, 16]], "divisible"),
        ([[0, 4, 6]], "positive integer"),
        ([[2, 4.5, 6]], "positive integer"),
        ([2, 4, 6], "positive integer"),
    ],
)
def test_invalid_packed_video_grids(grid, message):
    model = create_model(vision=True)
    with pytest.raises(ValueError, match=message):
        model.encode_video(mx.zeros((48, 24)), grid)


def test_video_with_image_audio_and_cached_decode():
    model = create_model(vision=True, audio=True)
    processor = create_processor()
    prompt = apply_chat_template(
        processor,
        model.config,
        "describe",
        video=["first.mp4", "second.mp4"],
        num_images=1,
        num_audios=1,
    )
    assert prompt.count("<|video|>") == 2
    assert processor_handles_video(processor)
    resolution = resolve_video_inputs(processor, ["clip.mp4"])
    assert resolution.videos == ["clip.mp4"] and not resolution.used_fallback
    inputs = prepare_inputs(
        processor,
        prompts=prompt,
        images=[np.zeros((4, 4, 3), np.uint8)],
        audio=[np.ones(5, np.float32)],
        videos=[frames(), frames(1, height=4, width=8)],
        fps=[2, 4],
    )
    ids = inputs.pop("input_ids")
    embeddings = model.get_input_embeddings(ids, **inputs).inputs_embeds
    expected_video = model.encode_video(
        inputs["pixel_values_videos"], inputs["video_grid_thw"]
    )

    def features_for(token_id):
        indices = mx.array(np.flatnonzero(np.asarray(ids[0]) == token_id))
        return embeddings[0, indices]

    assert mx.allclose(features_for(253), expected_video, atol=1e-5)
    assert mx.allclose(
        features_for(255), model.encode_image(inputs["pixel_values"]), atol=1e-5
    )
    assert mx.allclose(
        features_for(254),
        model.encode_audio(inputs["input_values"], inputs["input_values_mask"]),
        atol=1e-5,
    )
    full = model(ids, inputs_embeds=embeddings).logits
    cache = model.make_cache()
    chunks = [
        model(ids[:, :7], inputs_embeds=embeddings[:, :7], cache=cache).logits,
        model(ids[:, 7:-1], inputs_embeds=embeddings[:, 7:-1], cache=cache).logits,
        model(ids[:, -1:], cache=cache).logits,
    ]
    assert mx.all(mx.isfinite(full))
    assert mx.allclose(mx.concatenate(chunks, axis=1), full, atol=3e-3, rtol=3e-3)
    with pytest.raises(ValueError, match="tokens and features"):
        model.get_input_embeddings(
            mx.array([[253]]),
            pixel_values_videos=inputs["pixel_values_videos"],
            video_grid_thw=inputs["video_grid_thw"],
        )
    with pytest.raises(ValueError, match="vision encoder"):
        create_model().encode_video(inputs["pixel_values_videos"], [[2, 4, 6]])
    with pytest.raises(ValueError, match="video_grid_thw"):
        model.encode_video(inputs["pixel_values_videos"], None)


def test_saved_checkpoint_video_file_generation(tmp_path):
    import cv2

    from mlx_vlm import generate, load

    video_path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 4, (16, 16))
    assert writer.isOpened()
    for frame in frames(8, height=16, width=16):
        writer.write(frame.transpose(1, 2, 0))
    writer.release()

    model = create_model(vision=True)
    processor = create_processor()
    processor.video_processor.num_frames = 10
    processor.video_processor.frame_timestamp_format = "Time: {timestamp:.3f}s"
    processor.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(_CASE["config"]))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), hf_checkpoint(model))
    loaded, processor = load(tmp_path)
    assert processor.video_processor.num_frames == 10
    assert processor.video_processor.max_video_frame_tokens == 16
    assert processor.video_processor.temporal_patch_size == 2
    assert processor.video_processor.frame_timestamp_format == "Time: {timestamp:.3f}s"
    prompt = apply_chat_template(
        processor, loaded.config, "describe", video=video_path, num_images=0
    )
    with patch.object(processor, "_expand", wraps=processor._expand) as expand:
        inputs = prepare_inputs(processor, videos=[video_path], prompts=prompt, fps=2)
    assert inputs["video_grid_thw"].tolist() == [[2, 8, 8]]
    # Sampled indices [0, 2, 4, 7] at source fps=4 -> group starts 0, 1 sec.
    assert "Time: 1.000s" in expand.call_args.args[2][0]
    assert "Time: 1.750s" not in expand.call_args.args[2][0]
    # An explicit odd frame count is retained and padded by the processor.
    odd = prepare_inputs(processor, videos=[video_path], prompts=prompt, nframes=3)
    assert odd["video_grid_thw"].tolist() == [[2, 8, 8]]
    result = generate(
        loaded,
        processor,
        prompt,
        video=[video_path],
        max_tokens=2,
        verbose=False,
    )
    assert result.generation_tokens == 2
