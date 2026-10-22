"""LFM3 cache, rotary, checkpoint-layout, and multimodal contracts."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_vlm.models.lfm3 import Model, ModelConfig
from mlx_vlm.models.lfm3.language import RotaryEmbedding
from mlx_vlm.models.lfm3_vl import Model as VlModel
from mlx_vlm.models.lfm3_vl import ModelConfig as VlConfig
from mlx_vlm.models.lfm3_vl import VisionConfig
from mlx_vlm.models.lfm3_vl.processing_lfm3_vl import (
    Lfm3VlProcessor,
    Lfm3VlVideoProcessor,
)
from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor


def text_config(**kwargs):
    config = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        layer_types=["conv", "full_attention", "sliding_attention"],
        sliding_window=4,
    )
    return ModelConfig(**(config | kwargs))


def vl_config():
    return VlConfig(
        text_config=text_config(),
        vision_config=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            patch_size=2,
            layer_types=["conv", "full_attention"],
        ),
        projector_hidden_size=48,
        image_token_id=60,
        video_token_id=61,
    )


@pytest.mark.parametrize("chunks", [[13], [1] * 13, [3, 2, 1, 4, 3]])
@pytest.mark.parametrize("sandwich", [True, False])
def test_cached_chunks_match_full_forward_across_window(chunks, sandwich):
    mx.random.seed(31)
    model = Model(text_config(use_sandwich_norm=sandwich))
    model.eval()
    tokens = mx.array([[i % 64 for i in range(13)]])
    expected = model(tokens).logits
    cache = model.make_cache()
    actual, offset = [], 0
    for length in chunks:
        actual.append(model(tokens[:, offset : offset + length], cache=cache).logits)
        mx.eval(actual[-1], [c.state for c in cache])
        offset += length
    np.testing.assert_allclose(
        np.array(mx.concatenate(actual, axis=1)),
        np.array(expected),
        atol=2e-5,
        rtol=2e-5,
    )
    assert cache[1].offset == 13
    assert cache[2].offset == 13


def test_partial_rope_preserves_pairs_in_both_halves():
    rope = RotaryEmbedding(8, {"rope_theta": 10000, "partial_rotary_factor": 0.5})
    x = mx.arange(16, dtype=mx.float32).reshape(1, 1, 2, 8) + 1
    y = np.asarray(rope(x, offset=9))
    np.testing.assert_array_equal(
        y[..., [2, 3, 6, 7]], np.asarray(x)[..., [2, 3, 6, 7]]
    )
    angles = np.arange(9, 11)[:, None] * np.array([1, 10000 ** (-2 / 8)])
    a, b = np.asarray(x)[..., :2], np.asarray(x)[..., 4:6]
    np.testing.assert_allclose(
        y[..., :2], a * np.cos(angles) - b * np.sin(angles), atol=2e-6
    )
    np.testing.assert_allclose(
        y[..., 4:6], b * np.cos(angles) + a * np.sin(angles), atol=2e-6
    )


def test_merged_batch_cache_respects_per_row_rotary_offsets():
    model = Model(text_config())
    model.eval()
    caches = [model.make_cache(), model.make_cache()]
    for tokens, cache in zip([[1, 2, 3], [4, 5, 6, 7, 8, 9]], caches):
        mx.eval(model(mx.array([tokens]), cache=cache).logits)
    batched = [type(a).merge([a, b]) for a, b in zip(*caches)]
    actual = model(mx.array([[10], [11]]), cache=batched).logits
    expected = mx.concatenate(
        [
            model(mx.array([[token]]), cache=cache).logits
            for token, cache in zip([10, 11], caches)
        ]
    )
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("visual", [False, True])
def test_hf_checkpoint_keys_and_sanitize_idempotency(visual):
    model = VlModel(vl_config()) if visual else Model(text_config())
    native = dict(tree_flatten(model.parameters()))
    hf = {}
    for name, value in native.items():
        value = (
            mx.arange(value.size, dtype=mx.float32).reshape(value.shape) / value.size
        )
        native[name] = value
        if name.endswith("conv.conv.weight"):
            value = value.transpose(0, 2, 1)
        elif name.endswith("spatial_conv.weight"):
            value = value.transpose(0, 3, 1, 2)
        if visual:
            name = name.replace("language_model.model.", "language_model.")
            name = "model." + name
        else:
            name = name.removeprefix("language_model.")
        hf[name] = value
    hf["lm_head.weight"] = mx.ones((64, 32))
    once = model.sanitize(hf)
    twice = model.sanitize(once)
    assert once.keys() == native.keys() == twice.keys()
    for name in native:
        np.testing.assert_array_equal(np.asarray(once[name]), np.asarray(native[name]))
        np.testing.assert_array_equal(np.asarray(twice[name]), np.asarray(native[name]))
    model.load_weights(list(once.items()), strict=True)


def test_visual_frame_boundaries_and_cached_features():
    model = VlModel(vl_config())
    pixels = mx.random.normal((24, 24))
    grid = mx.array([[1, 4, 4], [2, 2, 2]])
    combined = model.encode_images(pixels, image_grid_thw=grid)
    separate = mx.concatenate(
        [
            model.encode_images(pixels[:16], image_grid_thw=mx.array([[1, 4, 4]])),
            model.encode_images(pixels[16:], image_grid_thw=mx.array([[2, 2, 2]])),
        ]
    )
    np.testing.assert_allclose(np.asarray(combined), np.asarray(separate), atol=1e-5)
    ids = mx.array([[1] + [60] * 6 + [2]])
    full = model.get_input_embeddings(ids, pixels, image_grid_thw=grid).inputs_embeds
    cached = model.get_input_embeddings(
        ids, pixels, cached_image_features=combined
    ).inputs_embeds
    np.testing.assert_array_equal(np.asarray(full), np.asarray(cached))
    with pytest.raises(ValueError, match="tokens"):
        model.get_input_embeddings(mx.array([[60]]), pixels, image_grid_thw=grid)


def test_media_placeholders_are_validated_without_replacing_inserted_tokens():
    result = Lfm3VlProcessor._expand(
        ["a<|image|>b<|image|>c"], "<|image|>", ["<|image|>" * 2, "<|image|>" * 3]
    )
    assert result == ["a<|image|><|image|>b<|image|><|image|><|image|>c"]
    with pytest.raises(ValueError, match="placeholders"):
        Lfm3VlProcessor._expand(["a"], "<|image|>", ["image"])
    with pytest.raises(ValueError, match="placeholders"):
        Lfm3VlProcessor._expand(["<|image|>"], "<|image|>", [])


@pytest.mark.parametrize("visual", [False, True])
def test_quantized_checkpoint_roundtrip(tmp_path, visual):
    import json
    from dataclasses import asdict

    from mlx_vlm.utils import load_model

    if visual:
        config = vl_config()
        config.vision_config.patch_size = 4
        config.projector_hidden_size = 64
        model = VlModel(config)
    else:
        model = Model(text_config())
    nn.quantize(model, group_size=32, bits=4)
    model.eval()
    ids = mx.array([[1, 60, 2]]) if visual else mx.array([[1, 2, 3]])
    kwargs = (
        dict(
            pixel_values=mx.random.normal((4, 96)), image_grid_thw=mx.array([[1, 2, 2]])
        )
        if visual
        else {}
    )
    expected = model(ids, **kwargs).logits
    config = asdict(model.config)
    config["quantization"] = dict(group_size=32, bits=4)
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    restored = load_model(tmp_path)
    actual = restored(ids, **kwargs).logits
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def make_processor():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokens = ["<unk>", "<|image|>", "<|video|>", "<|vision_start|>", "<|vision_end|>"]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>")
        ),
        unk_token="<unk>",
        additional_special_tokens=tokens[1:],
    )
    return Lfm3VlProcessor(
        Qwen3VLImageProcessor(min_pixels=4096, max_pixels=16384),
        tokenizer,
        Lfm3VlVideoProcessor(min_pixels=4096, max_pixels=65536),
    )


def test_processor_save_preserves_resolution_override(tmp_path):
    from PIL import Image

    processor = make_processor()
    processor.image_processor.max_pixels = 4096
    image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    expected = processor(text="<|image|>", images=image, return_tensors="np")
    processor.save_pretrained(tmp_path)
    restored = Lfm3VlProcessor.from_pretrained(tmp_path)
    actual = restored(text="<|image|>", images=image, return_tensors="np")
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_video_timestamps_use_sampled_fps_and_repeat_the_last_odd_frame():
    processor = make_processor()
    tokenizer = processor.tokenizer

    class RecordingTokenizer:
        def __call__(self, text, **kwargs):
            self.text = text
            return tokenizer(text, **kwargs)

    processor.tokenizer = RecordingTokenizer()
    frames = np.stack(
        [np.full((64, 64, 3), value, dtype=np.uint8) for value in (0, 100, 255)]
    )
    result = processor(
        text="<|video|>", videos=[frames], fps=[2.0], return_tensors="np"
    )
    text = processor.tokenizer.text[0]
    assert "<0.2 seconds>" in text  # average of frames at 0.0 and 0.5 seconds
    assert "<1.0 seconds>" in text  # the final frame is duplicated, not extrapolated
    np.testing.assert_array_equal(result["video_grid_thw"], [[2, 4, 4]])
    np.testing.assert_allclose(result["pixel_values_videos"][-16:], 1.0)
    with pytest.raises(ValueError, match="pre-sampled"):
        processor(text="<|video|>", videos=[frames], do_sample_frames=True)
