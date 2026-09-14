"""Mage-VL frame sampling, prompt ordering, and image-path video regressions."""

import json
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoProcessor, PreTrainedTokenizerFast

from mlx_vlm.generate.video import processor_handles_video
from mlx_vlm.models.mage_vl.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.mage_vl.mage_vl import Model
from mlx_vlm.models.mage_vl.processing_mage_vl import (
    IMAGE_PAD,
    VIDEO_PAD,
    VISION_END,
    VISION_START,
    MageVLProcessor,
)
from mlx_vlm.models.mage_vl.vision import build_cu_seqlens
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor
from mlx_vlm.prompt_utils import apply_chat_template, get_message_json
from mlx_vlm.utils import VideoMetadata, prepare_inputs, resolve_video_sampling

VIDEO_BLOCK = VISION_START + VIDEO_PAD + VISION_END
IMAGE_BLOCK = VISION_START + IMAGE_PAD + VISION_END
CHAT_TEMPLATE = (
    "{% for message in messages %}{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
)


class RecordingTokenizer(PreTrainedTokenizerFast):
    def __call__(self, text, **kwargs):
        self.last_text = text
        return super().__call__(text, **kwargs)


@pytest.fixture
def processor():
    tokens = ["[UNK]", "[PAD]", IMAGE_PAD, VIDEO_PAD, VISION_START, VISION_END]
    tokenizer = RecordingTokenizer(
        tokenizer_object=Tokenizer(
            WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]")
        ),
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=tokens[2:],
        chat_template=CHAT_TEMPLATE,
    )
    image_processor = Qwen3VLImageProcessor(
        patch_size=16,
        temporal_patch_size=1,
        merge_size=2,
        min_pixels=1024,
        max_pixels=8192,
    )
    return MageVLProcessor(image_processor=image_processor, tokenizer=tokenizer)


def frames(count=3, width=32, value=0):
    return np.full((count, 3, 32, width), value, dtype=np.uint8)


def image_counts(output):
    return (np.array(output["input_ids"]) == 2).sum(axis=1).tolist()


def test_image_batch_keeps_distinct_counts(processor):
    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))]
    output = processor(text=[IMAGE_BLOCK, IMAGE_BLOCK], images=images)
    assert image_counts(output) == [1, 2]
    assert "patch_positions" not in output
    expected = processor.image_processor(images)
    np.testing.assert_array_equal(output["pixel_values"], expected["pixel_values"])


def test_video_uses_reference_timestamps_positions_and_frame_attention(processor):
    metadata = VideoMetadata(total_num_frames=61, fps=30, frames_indices=[0, 15, 60])
    output = processor(text=VIDEO_BLOCK, videos=[frames()], video_metadata=[metadata])
    assert processor.tokenizer.last_text == [
        "".join(f"<{sec:.1f} seconds>{IMAGE_BLOCK}" for sec in [0, 0.5, 2])
    ]
    assert image_counts(output) == [3]
    assert "pixel_values_videos" not in output
    assert "video_grid_thw" not in output
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
    positions = np.array(output["patch_positions"]).reshape(3, 4, 3)
    np.testing.assert_array_equal(positions[:, :, 0], [[0] * 4, [15] * 4, [60] * 4])
    np.testing.assert_array_equal(positions[1, :, 1:], [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert build_cu_seqlens(output["image_grid_thw"].tolist(), 12, 4) == [0, 4, 8, 12]


def test_mixed_media_and_batch_follow_prompt_order(processor):
    stills = [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (64, 32), "blue")]
    clips = [frames(2, value=10), frames(3, width=64, value=20)]
    output = processor(
        text=[VIDEO_BLOCK + IMAGE_BLOCK, IMAGE_BLOCK + VIDEO_BLOCK],
        images=stills,
        videos=clips,
        fps=[2, 1],
    )
    expected = processor.image_processor([*clips[0], stills[0], stills[1], *clips[1]])
    np.testing.assert_array_equal(output["pixel_values"], expected["pixel_values"])
    np.testing.assert_array_equal(output["image_grid_thw"], expected["image_grid_thw"])
    assert image_counts(output) == [3, 8]
    assert VIDEO_PAD not in "".join(processor.tokenizer.last_text)
    expected_times = np.repeat([0, 1, 0, 0, 0, 1, 2], [4, 4, 4, 8, 8, 8, 8])
    np.testing.assert_array_equal(
        np.array(output["patch_positions"])[:, 0], expected_times
    )


@pytest.mark.parametrize("layout", ["tchw", "thwc", "pil", "batched_pil"])
def test_predecoded_formats_agree(processor, layout):
    video = frames(3, width=64, value=30)
    expected = processor(text=VIDEO_BLOCK, videos=[video])
    supplied = video
    if layout == "thwc":
        supplied = video.transpose(0, 2, 3, 1)
    elif "pil" in layout:
        supplied = [Image.fromarray(frame.transpose(1, 2, 0)) for frame in video]
        if layout == "batched_pil":
            supplied = [supplied]
    output = processor(text=VIDEO_BLOCK, videos=supplied)
    for key in ("pixel_values", "image_grid_thw", "patch_positions", "input_ids"):
        np.testing.assert_array_equal(output[key], expected[key])


@pytest.mark.parametrize(
    "prompt,videos",
    [(VIDEO_BLOCK * 2, [frames()]), (VIDEO_BLOCK, [frames(), frames()])],
)
def test_video_placeholder_mismatch_is_rejected(processor, prompt, videos):
    with pytest.raises(ValueError, match="placeholder"):
        processor(text=prompt, videos=videos)


def test_bad_metadata_is_rejected(processor):
    with pytest.raises(ValueError, match="one video_metadata"):
        processor(text=VIDEO_BLOCK, videos=[frames()], video_metadata=[])
    with pytest.raises(ValueError, match="frame count"):
        processor(
            text=VIDEO_BLOCK,
            videos=[frames()],
            video_metadata=[
                VideoMetadata(total_num_frames=10, fps=30, frames_indices=[0, 9])
            ],
        )
    with pytest.raises(ValueError, match="positive and finite"):
        processor(text=VIDEO_BLOCK, videos=[frames()], fps=0)


def test_prompt_utils_keeps_stills_with_video(processor):
    message = get_message_json(
        "mage_vl", "Describe both", num_images=1, video=["clip.mp4"]
    )
    assert [part["type"] for part in message["content"]] == ["image", "video", "text"]
    prompt = apply_chat_template(
        processor,
        {"model_type": "mage_vl"},
        "Describe both",
        num_images=1,
        video=["clip.mp4"],
    )
    assert prompt == IMAGE_BLOCK + VIDEO_BLOCK + "Describe both"
    assert processor_handles_video(processor)


def test_prepare_inputs_preserves_exact_source_metadata(processor):
    source = VideoMetadata(total_num_frames=240, fps=24, frames_indices=[0, 120, 239])
    with patch("mlx_vlm.utils.load_video", return_value=(frames(), source)) as decode:
        output = prepare_inputs(
            processor, videos=["clip.mp4"], prompts=VIDEO_BLOCK, nframes=3
        )
    sampling = decode.call_args.args[1]
    assert sampling.nframes == 3 and sampling.frame_factor == 1
    assert processor.tokenizer.last_text == [
        "".join(f"<{sec:.1f} seconds>{IMAGE_BLOCK}" for sec in [0, 5, 239 / 24])
    ]
    assert np.array(output["patch_positions"])[-1, 0] == 239


def test_prepare_inputs_accepts_supplied_metadata(processor):
    metadata = {"total_num_frames": 60, "fps": 10, "frames_indices": [0, 20, 59]}
    output = prepare_inputs(
        processor, videos=[frames()], prompts=VIDEO_BLOCK, video_metadata=[metadata]
    )
    assert "<5.9 seconds>" in processor.tokenizer.last_text[0]
    assert np.array(output["patch_positions"])[-1, 0] == 59


def test_shared_decoder_supports_odd_frame_counts(processor, tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
    if not writer.isOpened():
        pytest.skip("mp4v encoder unavailable")
    for value in (0, 100, 200):
        writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    writer.release()
    output = prepare_inputs(processor, videos=[path], prompts=VIDEO_BLOCK, nframes=3)
    assert output["image_grid_thw"].shape == (3, 3)
    assert np.array(output["patch_positions"])[-1, 0] == 2


def test_numpy_processor_loads_through_auto_processor(processor, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mage_vl"}))
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "patch_size": 16,
                "temporal_patch_size": 1,
                "merge_size": 2,
                "min_pixels": 1024,
                "max_pixels": 8192,
            }
        )
    )
    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=processor.tokenizer
    ):
        loaded = AutoProcessor.from_pretrained(tmp_path)
    assert isinstance(loaded, MageVLProcessor)
    assert processor_handles_video(loaded)
    sampling = resolve_video_sampling(loaded, {})
    assert (sampling.min_frames, sampling.frame_factor, sampling.max_frames) == (
        1,
        1,
        32,
    )
    assert image_counts(loaded(text=VIDEO_BLOCK, videos=[frames()])) == [3]


def test_mixed_video_pixels_affect_only_their_visual_embeddings(processor):
    config = ModelConfig(
        text_config=TextConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=32,
        ),
        vision_config=VisionConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            out_hidden_size=64,
            text_hidden_size=64,
        ),
        image_token_id=2,
        video_token_id=3,
    )
    model = Model(config)
    kwargs = {
        "text": IMAGE_BLOCK + VIDEO_BLOCK,
        "images": [Image.new("RGB", (32, 32), "red")],
    }
    first = processor(**kwargs, videos=[frames(2, value=0)])
    second = processor(**kwargs, videos=[frames(2, value=255)])
    a = model.get_input_embeddings(**first).inputs_embeds
    b = model.get_input_embeddings(**second).inputs_embeds
    mx.eval(a, b)
    visual_indices = np.flatnonzero(np.array(first["input_ids"])[0] == 2)
    np.testing.assert_array_equal(
        np.array(a)[0, visual_indices[0]], np.array(b)[0, visual_indices[0]]
    )
    assert not np.allclose(
        np.array(a)[0, visual_indices[1:]], np.array(b)[0, visual_indices[1:]]
    )
    assert mx.all(mx.isfinite(a)).item() and mx.all(mx.isfinite(b)).item()
