from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm.models.glm5_next.config import ModelConfig, TextConfig, VisionConfig
from mlx_vlm.models.glm5_next.fp8 import (
    convert_glm5_next_fp8_weights,
    make_quantization_config,
)
from mlx_vlm.models.glm5_next.glm5_next import Model
from mlx_vlm.models.glm5_next.language import LanguageModel
from mlx_vlm.models.glm5_next.processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    _resize_geometry,
)
from mlx_vlm.prompt_utils import apply_chat_template


def _text_config():
    return TextConfig(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        mlp_layer_types=["dense", "sparse"],
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        indexer_types=["full", "full"],
        index_topk=4,
        index_kpool=2,
        index_head_dim=4,
        index_n_heads=2,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        hc_mult=4,
        max_position_embeddings=64,
    )


def _vision_config(depth=0):
    return VisionConfig(
        depth=depth,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        projection_intermediate_size=24,
    )


def _image_processor():
    return Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=4,
        max_image_tokens=64,
        do_normalize=False,
    )


def _video_processor():
    return Glm5NextVideoProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=4,
        max_image_tokens=64,
        do_normalize=False,
    )


class _Tokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    image_token = "<|image|>"
    image_token_id = 31
    video_token = "<|video|>"
    video_token_id = 30

    def __init__(self):
        self.last_text = None

    def convert_tokens_to_ids(self, token):
        return {
            self.image_token: self.image_token_id,
            self.video_token: self.video_token_id,
        }.get(token, 1)

    def __call__(self, text, **kwargs):
        del kwargs
        self.last_text = text
        rows = []
        for value in text:
            ids = []
            index = 0
            while index < len(value):
                if value.startswith(self.image_token, index):
                    ids.append(self.image_token_id)
                    index += len(self.image_token)
                elif value.startswith(self.video_token, index):
                    ids.append(self.video_token_id)
                    index += len(self.video_token)
                else:
                    ids.append(1)
                    index += 1
            rows.append(ids)
        return {
            "input_ids": rows,
            "attention_mask": [[1] * len(row) for row in rows],
        }

    def batch_decode(self, *args, **kwargs):
        del args, kwargs
        return []

    def decode(self, *args, **kwargs):
        del args, kwargs
        return ""

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        return ""


def test_glm5_next_config_builds_checkpoint_schedules():
    config = TextConfig(num_hidden_layers=5, index_topk=8, index_kpool=4)

    assert config.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "deepseek_sparse_attention",
        "linear_attention",
    ]
    assert config.mlp_layer_types == ["dense", "dense", "dense", "sparse", "sparse"]
    assert config.linear_num_heads == 64
    assert config.linear_lower_bound == -5.0


def test_glm5_next_preprocessing_preserves_aspect_and_pads_temporally():
    settings = {
        "patch_size": 2,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "patch_expand_factor": 1,
        "min_image_tokens": 4,
        "max_image_tokens": 64,
    }
    assert _resize_geometry(2, 5, 9, **settings) == (8, 12, 6, 12)

    image = np.full((5, 9, 3), 255, dtype=np.uint8)
    image_inputs = _image_processor()(image)
    assert image_inputs["pixel_values"].shape == (24, 24)
    np.testing.assert_array_equal(image_inputs["image_grid_thw"], [[1, 4, 6]])

    video = np.full((3, 5, 9, 3), 255, dtype=np.uint8)
    video_inputs = _video_processor()(video)
    assert video_inputs["pixel_values_videos"].shape == (48, 24)
    np.testing.assert_array_equal(video_inputs["video_grid_thw"], [[2, 4, 6]])


def test_glm5_next_processor_expands_video_frames_with_timestamps():
    tokenizer = _Tokenizer()
    processor = Glm5NextProcessor(
        image_processor=_image_processor(),
        tokenizer=tokenizer,
        video_processor=_video_processor(),
    )
    video = np.full((3, 5, 9, 3), 255, dtype=np.uint8)
    metadata = SimpleNamespace(timestamps=[0.0, 0.5, 1.0, 1.5])

    output = processor(
        text="<|video|>",
        videos=video,
        video_metadata=[metadata],
        return_mm_token_type_ids=True,
    )

    rendered = tokenizer.last_text[0]
    assert rendered.count("<|begin_of_image|>") == 2
    assert rendered.count("<|image|>") == 12
    assert "0.0 seconds" in rendered
    assert "1.0 seconds" in rendered
    assert int(np.asarray(output["mm_token_type_ids"]).sum()) == 12


def test_glm5_next_prompt_utils_formats_image_and_video_content():
    image_messages = apply_chat_template(
        None,
        {"model_type": "glm5_next"},
        "Describe the image.",
        return_messages=True,
        num_images=1,
    )
    assert image_messages[0]["content"][0] == {"type": "image"}

    video_messages = apply_chat_template(
        None,
        {"model_type": "glm5_next"},
        "Describe the video.",
        return_messages=True,
        video="clip.mp4",
        fps=2,
    )
    assert video_messages[0]["content"][0] == {
        "type": "video",
        "video": "clip.mp4",
        "max_pixels": 224 * 224,
        "fps": 2,
    }


def test_glm5_next_hybrid_decoder_cache_matches_full_forward():
    mx.random.seed(0)
    model = LanguageModel(_text_config())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

    full = model(tokens).logits
    cache = model.make_cache()
    decoded = []
    for index in range(tokens.shape[1]):
        logits = model(tokens[:, index : index + 1], cache=cache).logits
        mx.eval(logits)
        decoded.append(logits)
    decoded = mx.concatenate(decoded, axis=1)
    mx.eval(full, decoded)

    assert full.shape == (1, 5, 40)
    assert bool(mx.all(mx.isfinite(full)).item())
    assert float(mx.max(mx.abs(full - decoded)).item()) < 1e-5

    chunk_cache = model.make_cache()
    first = model(tokens[:, :2], cache=chunk_cache).logits
    second = model(tokens[:, 2:], cache=chunk_cache).logits
    chunked = mx.concatenate([first, second], axis=1)
    mx.eval(chunked)
    assert float(mx.max(mx.abs(full - chunked)).item()) < 1e-5


def test_glm5_next_multimodal_forward_and_sanitize():
    config = ModelConfig(
        text_config=_text_config(),
        vision_config=_vision_config(depth=1),
        image_token_id=31,
        video_token_id=30,
        image_start_token_id=32,
        image_end_token_id=33,
        video_start_token_id=34,
        video_end_token_id=35,
    )
    model = Model(config)
    model.eval()
    image_inputs = _image_processor()(np.full((4, 4, 3), 255, dtype=np.uint8))
    input_ids = mx.array([[31] * 4], dtype=mx.int32)
    output = model(
        input_ids,
        pixel_values=mx.array(image_inputs["pixel_values"]),
        image_grid_thw=mx.array(image_inputs["image_grid_thw"]),
    )
    mx.eval(output.logits)
    assert output.logits.shape == (1, 4, 40)
    assert bool(mx.all(mx.isfinite(output.logits)).item())

    raw = {
        "model.visual.patch_embed.proj.weight": mx.zeros((8, 3, 2, 2, 2)),
        "model.visual.downsample.weight": mx.zeros((16, 8, 2, 2)),
        "model.language_model.layers.0.hc_attn_fn": mx.zeros((24, 64)),
        "model.language_model.layers.0.self_attn.q_conv1d.weight": mx.zeros((8, 1, 2)),
        "model.language_model.layers.1.self_attn.kv_b_proj.weight": mx.zeros((16, 4)),
        "model.language_model.layers.2.input_layernorm.weight": mx.zeros((16,)),
    }
    for expert in range(2):
        for name, shape in (
            ("gate_proj", (8, 16)),
            ("up_proj", (8, 16)),
            ("down_proj", (16, 8)),
        ):
            raw[f"model.language_model.layers.1.mlp.experts.{expert}.{name}.weight"] = (
                mx.zeros(shape)
            )

    sanitized = model.sanitize(raw)
    assert sanitized["vision_tower.patch_embed.proj.weight"].shape == (8, 2, 2, 2, 3)
    assert sanitized["vision_tower.downsample.weight"].shape == (16, 2, 2, 8)
    assert sanitized[
        "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"
    ].shape == (2, 8, 16)
    assert "language_model.model.layers.2.input_layernorm.weight" not in sanitized
    assert set(model.sanitize(sanitized)) == set(sanitized)


def test_glm5_next_fp8_conversion_uses_native_mxfp8():
    weights = {
        "linear.weight": mx.zeros((32, 32), dtype=mx.uint8),
        "linear.weight_scale_inv": mx.ones((1, 1), dtype=mx.bfloat16),
    }
    converted = convert_glm5_next_fp8_weights(weights)
    mx.eval(converted)

    assert converted["linear.weight"].shape == (32, 8)
    assert converted["linear.scales"].shape == (32, 1)
    assert make_quantization_config(
        {
            "model_type": "glm5_next",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            },
        }
    ) == {"group_size": 32, "bits": 8, "mode": "mxfp8"}
