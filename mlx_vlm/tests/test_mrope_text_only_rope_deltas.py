"""Rope deltas must be relative to each row's unpadded length."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import LanguageModel
from mlx_vlm.tests.test_models import build_config, model_case


def _language_model():
    args = TextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=1,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10000.0,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [2, 3, 3]},
    )
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
    )
    return LanguageModel(args, config)


def test_text_only_rope_deltas_ignore_left_padding():
    lm = _language_model()
    # Row 0 is left-padded by 3 tokens, row 1 has no padding.
    ids = mx.ones((2, 5), dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])

    _, deltas = lm.get_rope_index(ids, attention_mask=mask)

    # BatchKVCache offsets already exclude padding, so decode position is
    # offset + delta == number of real tokens seen so far.
    assert deltas.tolist() == [[0], [0]]


MODULES = [
    "mlx_vlm.models.glm4v.language",
    "mlx_vlm.models.glm4v_moe.language",
    "mlx_vlm.models.glm_ocr.language",
    "mlx_vlm.models.minicpmv4_6.language",
    "mlx_vlm.models.paddleocr_vl.language",
    "mlx_vlm.models.qwen2_vl.language",
    "mlx_vlm.models.qwen2_5_vl.language",
    "mlx_vlm.models.qwen3_5.language",
    "mlx_vlm.models.qwen3_5_text.language",
    "mlx_vlm.models.qwen3_vl_moe.language",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_sibling_text_only_rope_deltas_ignore_left_padding(module_name):
    module = importlib.import_module(module_name)
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
        vision_start_token_id=100,
    )
    # Row 0 is left-padded by 3 tokens, row 1 has no padding.
    ids = mx.ones((2, 5), dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])

    _, deltas = lm.get_rope_index(ids, attention_mask=mask)

    # BatchKVCache offsets already exclude padding, so decode position is
    # offset + delta == number of real tokens seen so far.
    assert deltas.tolist() == [[0], [0]]


def _minicpmv47_model():
    module = importlib.import_module("mlx_vlm.models.minicpmv4_7")
    case = model_case("minicpmv4_7")
    return module.Model(build_config(module, case["config"]))


def test_minicpmv47_text_only_rope_deltas_ignore_left_padding():
    model = _minicpmv47_model()
    ids = mx.array([[0, 0, 0, 3, 4], [3, 4, 5, 6, 7]], dtype=mx.int32)
    mask = mx.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])

    features = model.get_input_embeddings(ids, mask=mask)

    assert features.rope_deltas.tolist() == [[0], [0]]


def test_minicpmv47_image_row_rope_delta_ignores_left_padding():
    model = _minicpmv47_model()
    ids = mx.array([[0, 0, 1, 50, 48, 48, 48, 48, 51, 2]], dtype=mx.int32)
    mask = mx.array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    media = {
        "pixel_values": [[mx.random.normal((3, 2, 128))]],
        "tgt_sizes": [np.array([[8, 8]])],
        "downsample_mode": "16x",
    }

    features = model.get_input_embeddings(ids, mask=mask, **media)

    assert features.rope_deltas.tolist() == [[-3]]
