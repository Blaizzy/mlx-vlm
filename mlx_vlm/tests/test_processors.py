"""Tests for custom processor implementations."""

import unittest

import numpy as np
from PIL import Image


class TestErnie4_5VLProcessor(unittest.TestCase):
    """Test ERNIE 4.5 VL processor components."""

    def test_helper_functions(self):
        """Test helper functions for ERNIE 4.5 VL processor."""
        from mlx_vlm.models.ernie4_5_moe_vl.processor import (
            ceil_by_factor,
            floor_by_factor,
            round_by_factor,
            smart_resize,
        )

        # round_by_factor
        self.assertEqual(round_by_factor(100, 28), 112)
        self.assertEqual(round_by_factor(56, 28), 56)
        self.assertEqual(round_by_factor(42, 28), 56)

        # ceil_by_factor
        self.assertEqual(ceil_by_factor(100, 28), 112)
        self.assertEqual(ceil_by_factor(56, 28), 56)
        self.assertEqual(ceil_by_factor(57, 28), 84)

        # floor_by_factor
        self.assertEqual(floor_by_factor(100, 28), 84)
        self.assertEqual(floor_by_factor(56, 28), 56)
        self.assertEqual(floor_by_factor(55, 28), 28)

        # smart_resize maintains factor
        h, w = smart_resize(224, 224, factor=28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

        # smart_resize respects min_pixels
        h, w = smart_resize(10, 10, factor=28, min_pixels=56 * 56)
        self.assertGreaterEqual(h * w, 56 * 56)

        # smart_resize respects max_pixels
        h, w = smart_resize(10000, 10000, factor=28, max_pixels=28 * 28 * 1280)
        self.assertLessEqual(h * w, 28 * 28 * 1280)

    def test_image_processor(self):
        """Test ImageProcessor for ERNIE 4.5 VL model."""
        from mlx_vlm.models.ernie4_5_moe_vl.processor import ImageProcessor

        processor = ImageProcessor()

        # Initialization
        self.assertEqual(processor.patch_size, 14)
        self.assertEqual(processor.merge_size, 2)
        self.assertEqual(processor.factor, 28)

        # get_smart_resize
        (resized_h, resized_w), (grid_h, grid_w) = processor.get_smart_resize(224, 224)
        self.assertEqual(resized_h % 28, 0)
        self.assertEqual(resized_w % 28, 0)
        self.assertEqual(grid_h, resized_h // 14)
        self.assertEqual(grid_w, resized_w // 14)

        # preprocess single image
        image = Image.new("RGB", (224, 224), color="red")
        result = processor.preprocess(image)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        self.assertEqual(result["image_grid_thw"].shape[0], 1)
        self.assertEqual(result["image_grid_thw"][0, 0], 1)

        # preprocess multiple images
        images = [
            Image.new("RGB", (224, 224), color="red"),
            Image.new("RGB", (448, 448), color="blue"),
        ]
        result = processor.preprocess(images)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)
        self.assertEqual(result["image_grid_thw"].shape[0], 2)

        # extract patches shape
        img_array = np.random.rand(3, 224, 224).astype(np.float32)
        grid_h, grid_w = 16, 16
        patches = processor._extract_patches(img_array, grid_h, grid_w)
        expected_num_patches = (grid_h // 2) * (grid_w // 2) * 4
        expected_patch_dim = 3 * 14 * 14
        self.assertEqual(patches.shape, (expected_num_patches, expected_patch_dim))

        # callable interface
        image = Image.new("RGB", (224, 224), color="red")
        result = processor(images=image)
        self.assertIn("pixel_values", result)
        self.assertIn("image_grid_thw", result)

    def test_processor_class_attributes(self):
        """Test Ernie4_5_VLProcessor class attributes."""
        from mlx_vlm.models.ernie4_5_moe_vl.processor import Ernie4_5_VLProcessor

        self.assertEqual(Ernie4_5_VLProcessor.IMG_START, "<|IMAGE_START|>")
        self.assertEqual(Ernie4_5_VLProcessor.IMG_END, "<|IMAGE_END|>")
        self.assertEqual(
            Ernie4_5_VLProcessor.IMAGE_PLACEHOLDER, "<|IMAGE_PLACEHOLDER|>"
        )


if __name__ == "__main__":
    unittest.main()
