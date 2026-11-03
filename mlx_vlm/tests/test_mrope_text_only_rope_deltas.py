"""Rope deltas must be relative to each row's unpadded length."""

import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.models.cache import BatchKVCache, KVCache
from mlx_vlm.tests.test_models import build_config, model_case


def _language_model(model_type="qwen3_vl"):
    module = importlib.import_module(f"mlx_vlm.models.{model_type}")
    args = module.TextConfig.from_dict(
        dict(
            model_type=model_type,
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
    )
    config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
    )
    return module.LanguageModel(args, config)


@pytest.mark.parametrize("length,width", [(2, 5), (1, 2), (1, 5)])
def test_text_only_rope_deltas_ignore_left_padding(length, width):
    lm = _language_model()
    ids = mx.ones((2, width), dtype=mx.int32)
    mask = mx.array([[0] * (width - length) + [1] * length, [1] * width])

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
@pytest.mark.parametrize("length,width", [(2, 5), (1, 2), (1, 5)])
def test_sibling_text_only_rope_deltas_ignore_left_padding(module_name, length, width):
    module = importlib.import_module(module_name)
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
        vision_start_token_id=100,
    )
    ids = mx.ones((2, width), dtype=mx.int32)
    mask = mx.array([[0] * (width - length) + [1] * length, [1] * width])

    _, deltas = lm.get_rope_index(ids, attention_mask=mask)

    # BatchKVCache offsets already exclude padding, so decode position is
    # offset + delta == number of real tokens seen so far.
    assert deltas.tolist() == [[0], [0]]


@pytest.mark.parametrize("padding", [1, 4])
@pytest.mark.parametrize("model_type", ["qwen2_vl", "qwen2_5_vl", "qwen3_vl"])
def test_padded_one_token_decode_matches_standalone(model_type, padding):
    mx.random.seed(17)
    lm = _language_model(model_type)
    cache = [BatchKVCache([padding, 0]) for _ in lm.layers]
    ids = mx.array([[0] * padding + [3], [4] * padding + [5]])
    mask = mx.array([[0] * padding + [1], [1] * (padding + 1)])
    lm(ids, mask=mask, cache=cache)
    actual = lm(mx.array([[2], [2]]), cache=cache).logits[0, -1]

    lm._position_ids = lm._rope_deltas = None
    cache = [KVCache() for _ in lm.layers]
    lm(mx.array([[3]]), cache=cache)
    expected = lm(mx.array([[2]]), cache=cache).logits[0, -1]

    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def _minicpmv47_model():
    module = importlib.import_module("mlx_vlm.models.minicpmv4_7")
    case = model_case("minicpmv4_7")
    return module.Model(build_config(module, case["config"]))


@pytest.mark.parametrize("length,width", [(2, 5), (1, 2), (1, 5)])
def test_minicpmv47_text_only_rope_deltas_ignore_left_padding(length, width):
    model = _minicpmv47_model()
    ids = mx.array([[0] * (width - length) + [3] * length, [4] * width])
    mask = mx.array([[0] * (width - length) + [1] * length, [1] * width])

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
