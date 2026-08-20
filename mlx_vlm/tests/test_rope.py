import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

QWEN_STYLE_MODULES = [
    "mlx_vlm.models.glm4v.language",
    "mlx_vlm.models.glm4v_moe.language",
    "mlx_vlm.models.paddleocr_vl.language",
    "mlx_vlm.models.qwen2_vl.language",
    "mlx_vlm.models.qwen2_5_vl.language",
    "mlx_vlm.models.qwen3_5.language",
    "mlx_vlm.models.qwen3_omni_moe.language",
    "mlx_vlm.models.qwen3_vl.language",
    "mlx_vlm.models.qwen3_vl_moe.language",
]


def _mrope_config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
        vision_start_token_id=100,
    )


@pytest.mark.parametrize("module_name", QWEN_STYLE_MODULES)
def test_mrope_rope_index_handles_fully_masked_rows(module_name):
    module = importlib.import_module(module_name)
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = _mrope_config()

    input_ids = mx.array(
        [
            [0, 0, 0, 0],
            [10, 100, 101, 11],
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=mx.int32,
    )
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (3, 2, 4)
    assert rope_deltas.shape == (2, 1)
    assert rope_deltas.tolist()[0] == [0]


def test_glm_ocr_rope_index_handles_fully_masked_rows():
    module = importlib.import_module("mlx_vlm.models.glm_ocr.language")
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=101,
        video_token_id=102,
    )

    input_ids = mx.array(
        [
            [0, 0, 0, 0],
            [10, 101, 11, 12],
        ],
        dtype=mx.int32,
    )
    attention_mask = mx.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=mx.int32,
    )
    image_grid_thw = mx.array([[1, 2, 2]], dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (3, 2, 4)
    assert rope_deltas.shape == (2, 1)
    assert rope_deltas.tolist()[0] == [0]


def test_ernie_mrope_rope_index_handles_empty_rows():
    module = importlib.import_module("mlx_vlm.models.ernie4_5_moe_vl.language")
    lm = module.LanguageModel.__new__(module.LanguageModel)
    lm.config = _mrope_config()

    input_ids = mx.zeros((2, 0), dtype=mx.int32)
    image_grid_thw = mx.zeros((0, 3), dtype=mx.int32)

    position_ids, rope_deltas = lm.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
    )
    mx.eval(position_ids, rope_deltas)

    assert position_ids.shape == (2, 0, 3)
    assert rope_deltas.shape == (2,)
    assert rope_deltas.tolist() == [0, 0]
