import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx_vlm.models.paddleocr_vl.config import VisionConfig
from mlx_vlm.models.paddleocr_vl.vision import (
    Attention,
    PaddleOCRProjector,
    VisionModel,
)


def _assert_allclose(actual, expected, atol=1e-5, rtol=1e-5):
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        np.array(actual),
        np.array(expected),
        atol=atol,
        rtol=rtol,
    )


def _tiny_vision_model():
    return VisionModel(
        VisionConfig(
            model_type="paddleocr_vl",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=4,
            patch_size=2,
            spatial_merge_size=1,
        )
    )


class TestPaddleOCRVisionFastPaths(unittest.TestCase):
    def test_projector_splits_same_size_images_by_cumulative_boundaries(self):
        projector = PaddleOCRProjector(dim=4, context_dim=3, spatial_merge_size=1)
        hidden_states = mx.random.uniform(shape=(8, 4))
        grid_thw = mx.array([[1, 2, 2], [1, 2, 2]], dtype=mx.int32)

        output = projector(hidden_states, grid_thw)
        mx.eval(output)

        self.assertEqual(output.shape, (8, 3))

    def test_attention_uses_no_mask_for_single_segment(self):
        attention = Attention(dim=8, num_heads=2)
        hidden_states = mx.random.uniform(shape=(4, 8))
        cu_seqlens = mx.array([0, 4], dtype=mx.int32)
        rotary_pos_emb = mx.zeros((4, 2))
        seen = {}

        def sdpa(q, k, v, scale=None, mask=None):
            seen["mask"] = mask
            return mx.zeros_like(q)

        with patch.object(mx.fast, "scaled_dot_product_attention", side_effect=sdpa):
            output = attention(hidden_states, cu_seqlens, rotary_pos_emb)
            mx.eval(output)

        self.assertIsNone(seen["mask"])

    def test_attention_uses_additive_block_mask_for_multiple_segments(self):
        attention = Attention(dim=8, num_heads=2)
        hidden_states = mx.random.uniform(shape=(4, 8))
        cu_seqlens = mx.array([0, 2, 4], dtype=mx.int32)
        rotary_pos_emb = mx.zeros((4, 2))
        seen = {}

        def sdpa(q, k, v, scale=None, mask=None):
            seen["mask"] = mask
            return mx.zeros_like(q)

        with patch.object(mx.fast, "scaled_dot_product_attention", side_effect=sdpa):
            output = attention(hidden_states, cu_seqlens, rotary_pos_emb)
            mx.eval(output)

        mask = seen["mask"]
        self.assertIsNotNone(mask)
        mx.eval(mask)
        mask = np.array(mask)
        self.assertEqual(mask.shape, (1, 4, 4))
        self.assertEqual(mask[0, 0, 1], 0)
        self.assertTrue(np.isneginf(mask[0, 0, 2]))

    def test_same_grid_batch_fast_path_matches_packed_fallback(self):
        model = _tiny_vision_model()
        pixel_values = mx.random.uniform(shape=(2, 4, 3, 2, 2))
        grid_thw = mx.array([[1, 2, 2], [1, 2, 2]], dtype=mx.int32)

        self.assertTrue(model._use_same_grid_batch_path(pixel_values, grid_thw, False))

        fast = model(pixel_values, grid_thw, output_hidden_states=False)
        packed = pixel_values.reshape(1, 8, 3, 2, 2)
        expected = model(packed, grid_thw, output_hidden_states=False)

        self.assertEqual(fast.shape, (8, 1024))
        _assert_allclose(fast, expected)

    def test_non_same_grid_batch_input_uses_packed_fallback(self):
        model = _tiny_vision_model()
        pixel_values = mx.random.uniform(shape=(2, 4, 3, 2, 2))
        grid_thw = mx.array([[1, 2, 2], [1, 1, 4]], dtype=mx.int32)

        self.assertFalse(model._use_same_grid_batch_path(pixel_values, grid_thw, False))

        output = model(pixel_values, grid_thw, output_hidden_states=False)
        packed = pixel_values.reshape(1, 8, 3, 2, 2)
        expected = model(packed, grid_thw, output_hidden_states=False)

        self.assertEqual(output.shape, (8, 1024))
        _assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
