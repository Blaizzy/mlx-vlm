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

    def test_vlm_chat_template_emits_image_slots(self):
        jinja = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "glm5_next"
            / "chat_template_vlm.jinja"
        )
        text = jinja.read_text()
        self.assertIn("<|begin_of_image|><|image|><|end_of_image|>", text)
        self.assertTrue(jinja.exists())


if __name__ == "__main__":
    unittest.main()
