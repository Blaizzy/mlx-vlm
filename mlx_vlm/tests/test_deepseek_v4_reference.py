"""Optional real-checkpoint parity gates for DeepSeek-V4 Flash Vision.

Set ``DEEPSEEK_V4_VISION_MLX_PATH`` to the converted checkpoint and
``DEEPSEEK_V4_VISION_FIXTURE`` to an ``npz`` exported by the official PyTorch
reference. The fixture contains one fixed prompt/image and its processor,
aligned-vision, and first-token-logit outputs.
"""

import os
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_vlm.utils import load

MODEL_PATH = os.environ.get("DEEPSEEK_V4_VISION_MLX_PATH")
FIXTURE_PATH = os.environ.get("DEEPSEEK_V4_VISION_FIXTURE")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not FIXTURE_PATH,
    reason=(
        "set DEEPSEEK_V4_VISION_MLX_PATH and DEEPSEEK_V4_VISION_FIXTURE "
        "to run official-reference parity"
    ),
)


@lru_cache(maxsize=1)
def _fixture():
    return np.load(Path(FIXTURE_PATH), allow_pickle=False)


@lru_cache(maxsize=1)
def _model_and_processor():
    return load(str(Path(MODEL_PATH)), lazy=True)


def _processed():
    fixture = _fixture()
    _, processor = _model_and_processor()
    image = Image.fromarray(fixture["image_rgb"].astype(np.uint8), mode="RGB")
    return processor(
        str(fixture["prompt"].item()),
        images=[image],
        add_special_tokens=False,
        return_tensors="mlx",
    )


def test_official_processor_token_and_layout_parity():
    fixture = _fixture()
    processed = _processed()
    for key in (
        "input_ids",
        "image_grid_hw",
        "image_sample_indices",
        "image_offsets",
        "image_types",
        "image_type_offsets",
        "image_permutations",
    ):
        assert np.array_equal(np.asarray(processed[key]), fixture[key]), key
    # The official processor rounds normalized patches to BF16 immediately;
    # MLX keeps FP32 values until ``encode_image`` performs the same cast.
    # Compare the values at that shared vision-input stage.
    official_stage_pixels = processed["pixel_values"].astype(mx.bfloat16)
    assert np.allclose(
        np.asarray(official_stage_pixels.astype(mx.float32)),
        fixture["pixel_values"],
        atol=0,
        rtol=0,
    )


def test_official_aligned_vision_feature_parity():
    fixture = _fixture()
    model, _ = _model_and_processor()
    processed = _processed()
    features = model.encode_images(
        processed["pixel_values"],
        image_grid_hw=processed["image_grid_hw"],
        image_permutations=processed["image_permutations"],
    )[0]
    mx.eval(features)

    assert np.allclose(
        np.asarray(features.astype(mx.float32)),
        fixture["aligned_vision_features"],
        atol=float(fixture.get("vision_atol", 2e-2)),
        rtol=float(fixture.get("vision_rtol", 2e-2)),
    )


def test_official_first_token_logits_parity():
    fixture = _fixture()
    model, _ = _model_and_processor()
    processed = _processed()
    output = model(
        processed["input_ids"],
        pixel_values=processed["pixel_values"],
        **{key: value for key, value in processed.items() if key.startswith("image_")},
    )
    logits = output.logits[0, -1].astype(mx.float32)
    mx.eval(logits)
    expected = fixture["first_token_logits"]

    assert int(mx.argmax(logits).item()) == int(np.argmax(expected))
    assert np.allclose(
        np.asarray(logits),
        expected,
        atol=float(fixture.get("logits_atol", 1e-1)),
        rtol=float(fixture.get("logits_rtol", 2e-2)),
    )
