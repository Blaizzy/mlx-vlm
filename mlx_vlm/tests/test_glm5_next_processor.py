"""GLM-5.3-Flash processor: token-budget resize and one-shot image-slot expansion.

Does not load Hugging Face configs or tokenizers.
"""

import unittest
from pathlib import Path

import numpy as np

from mlx_vlm.models.glm5_next.processing import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    llm_image_tokens,
    smart_resize,
)
from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat


class _CaptureTokenizer:
    model_input_names = ["input_ids", "attention_mask"]
    image_token = "<|image|>"
    video_token = "<|video|>"

    def __init__(self):
        self.last_text = None

    def convert_tokens_to_ids(self, tok):
        return 154854 if tok == "<|image|>" else 0

    def __call__(self, text, padding=False, return_token_type_ids=False, **kwargs):
        self.last_text = text
        texts = [text] if isinstance(text, str) else text
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }


class TestGlm5NextProcessor(unittest.TestCase):
    def test_prompt_utils_lists_glm5_next(self):
        self.assertEqual(MODEL_CONFIG["glm5_next"], MessageFormat.LIST_WITH_IMAGE_FIRST)

    def test_smart_resize_448_is_256_tokens(self):
        h, w = smart_resize(
            2, 448, 448, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000
        )
        self.assertEqual((h, w), (448, 448))
        self.assertEqual(llm_image_tokens(448, 448), 256)
        self.assertEqual((448 // 14) * (448 // 14) // 4, 256)

    def test_smart_resize_354_is_169_tokens(self):
        self.assertEqual(llm_image_tokens(354, 354), 169)
        h, w = smart_resize(
            2, 354, 354, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000
        )
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)
        self.assertEqual((h // 14) * (w // 14) // 4, 169)

    def test_glm_ocr_pixel_budget_is_wrong_for_glm53_token_limits(self):
        from mlx_vlm.models.glm_ocr.processing import smart_resize as glm_ocr_resize

        ours = smart_resize(
            2, 448, 448, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000
        )
        theirs = glm_ocr_resize(
            2, 448, 448, temporal_factor=2, factor=28, min_pixels=16, max_pixels=8000
        )
        self.assertEqual(ours, (448, 448))
        self.assertNotEqual(theirs, (448, 448))

    def test_image_processor_448_grid(self):
        processor = Glm5NextImageProcessor()
        out = processor(np.zeros((3, 448, 448), dtype=np.uint8))
        self.assertEqual(out["image_grid_thw"].tolist(), [[1, 32, 32]])
        self.assertEqual(out["pixel_values"].shape, (1024, 3 * 2 * 14 * 14))

    def test_image_slot_expansion_is_not_reconsumed(self):
        tokenizer = _CaptureTokenizer()
        image_processor = Glm5NextImageProcessor()
        processor = Glm5NextProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
        slot = "<|begin_of_image|><|image|><|end_of_image|>"
        out = processor(
            images=np.zeros((3, 448, 448), dtype=np.uint8),
            text=slot,
        )
        self.assertIn("input_ids", out)
        expanded = tokenizer.last_text[0]
        self.assertEqual(expanded.count("<|image|>"), 256)
        self.assertTrue(expanded.startswith("<|begin_of_image|>"))
        self.assertTrue(expanded.endswith("<|end_of_image|>"))
        self.assertEqual(
            expanded,
            "<|begin_of_image|>" + ("<|image|>" * 256) + "<|end_of_image|>",
        )




# ---------------------------------------------------------------------------
# Numeric parity with transformers' Glm5NextImageProcessor (weight-free).
# ---------------------------------------------------------------------------
_GLM53_IMAGE_PROCESSOR_KWARGS = {
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "min_image_tokens": 16,
    "max_image_tokens": 8000,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
_EXAMPLE_IMAGES = Path(__file__).resolve().parents[2] / "examples" / "images"


def _hf_glm5_next_image_processor():
    try:
        from transformers.models.glm5_next.image_processing_glm5_next import (
            Glm5NextImageProcessor as HFProc,
        )
    except Exception:  # pragma: no cover - older transformers
        return None
    return HFProc(**_GLM53_IMAGE_PROCESSOR_KWARGS)


class TestParityWithTransformers(unittest.TestCase):
    """pixel_values / image_grid_thw must match transformers on real images."""

    def setUp(self):
        self.hf = _hf_glm5_next_image_processor()
        if self.hf is None:
            self.skipTest("transformers Glm5NextImageProcessor not importable")
        from PIL import Image

        self.images = []
        for name in ("cats.jpg", "graph.png"):
            path = _EXAMPLE_IMAGES / name
            if path.exists():
                self.images.append((name, Image.open(path).convert("RGB")))
        if not self.images:
            self.skipTest("examples/images not available")
        self.ours = Glm5NextImageProcessor(**_GLM53_IMAGE_PROCESSOR_KWARGS)

    def test_grid_and_pixels_match(self):
        merge = _GLM53_IMAGE_PROCESSOR_KWARGS["merge_size"]
        for name, img in self.images:
            with self.subTest(image=name):
                ref = self.hf(images=[img], return_tensors="np")
                out = self.ours(images=[img])
                ref_grid = np.asarray(ref["image_grid_thw"]).reshape(-1, 3)
                our_grid = np.asarray(out["image_grid_thw"]).reshape(-1, 3)
                np.testing.assert_array_equal(our_grid, ref_grid)
                ref_px = np.asarray(ref["pixel_values"], dtype=np.float32)
                our_px = np.asarray(out["pixel_values"], dtype=np.float32)
                self.assertEqual(our_px.shape, ref_px.shape)
                self.assertLess(float(np.max(np.abs(our_px - ref_px))), 1e-5)
                # Patch count from the size helper equals the grid volume; the
                # prompt expander divides it by merge_size**2.
                self.assertEqual(
                    self.ours.get_number_of_image_patches(img.height, img.width),
                    int(np.prod(our_grid[0])),
                )
                self.assertEqual(int(np.prod(our_grid[0])) % (merge * merge), 0)


if __name__ == "__main__":
    unittest.main()
