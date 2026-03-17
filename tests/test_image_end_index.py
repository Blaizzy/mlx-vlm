"""
Unit tests for qwen3_5 image token boundary computations.

Covers two functions:
- image_end_index: end of the last image block (used by get_input_embeddings)
- new_img_start: start of the first *new* image block at partial_depth
  (used by get_partial_input_embeddings for partial image KV cache)
"""

import unittest

import mlx.core as mx

IMAGE_TOKEN = 151655  # Qwen3.5 image_token_index
VIDEO_TOKEN = 151656  # Qwen3.5 video_token_index
TEXT_TOKEN = 1234


def compute_image_end_index(input_ids, image_token_index, video_token_index):
    """Extracted from qwen3_5.get_input_embeddings — the logic under test."""
    flat_ids = input_ids.reshape(-1)
    image_mask = (flat_ids == image_token_index) | (flat_ids == video_token_index)
    has_image = bool(image_mask.any().item())
    if not has_image:
        return None
    last_pos = int((mx.arange(flat_ids.shape[0]) * image_mask).max().item())
    return last_pos + 1


def compute_new_img_start(
    input_ids, image_token_index, video_token_index, partial_depth
):
    """Extracted from qwen3_5.get_partial_input_embeddings — the logic under test."""
    flat = input_ids[0].tolist()
    block, in_block, new_img_start = 0, False, len(flat)
    for i, tok in enumerate(flat):
        is_vis = tok == image_token_index or tok == video_token_index
        if is_vis and not in_block:
            in_block = True
            if block == partial_depth:
                new_img_start = i
                break
        elif not is_vis and in_block:
            in_block = False
            block += 1
    return new_img_start


class TestImageEndIndex(unittest.TestCase):
    """Tests for image_end_index computation in get_input_embeddings."""

    def test_image_tokens_at_start(self):
        # [img, img, img, text, text]  → image_end_index = 3
        ids = mx.array(
            [[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, TEXT_TOKEN]]
        )
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 3)

    def test_image_tokens_in_middle(self):
        # [text, img, img, text, text]  → image_end_index = 3
        ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 3)

    def test_image_tokens_at_end(self):
        # [text, text, img, img]  → image_end_index = 4
        ids = mx.array([[TEXT_TOKEN, TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 4)

    def test_single_image_token(self):
        # [text, img, text]  → image_end_index = 2
        ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 2)

    def test_video_tokens(self):
        # [text, vid, vid, vid, text]  → image_end_index = 4
        ids = mx.array(
            [[TEXT_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN]]
        )
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 4)

    def test_mixed_image_and_video_tokens(self):
        # [img, img, vid, vid, text]  → image_end_index = 4
        ids = mx.array(
            [[IMAGE_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN]]
        )
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 4)

    def test_no_image_tokens_returns_none(self):
        ids = mx.array([[TEXT_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
        self.assertIsNone(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN))

    def test_realistic_prompt_layout(self):
        system = [TEXT_TOKEN] * 10
        image = [IMAGE_TOKEN] * 256  # typical image patch count
        user = [TEXT_TOKEN] * 20
        ids = mx.array([system + image + user])
        # last image token is at index 10 + 256 - 1 = 265
        self.assertEqual(compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN), 266)


class TestNewImgStart(unittest.TestCase):
    """Tests for new_img_start computation in get_partial_input_embeddings."""

    def test_partial_depth_0_single_image(self):
        # [text, img, img, text]  partial_depth=0 → new_img_start=1 (first image block)
        ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 0), 1)

    def test_partial_depth_1_two_images(self):
        # [img×3, text×2, img×2, text]  partial_depth=1 → new_img_start=5 (second block)
        ids = mx.array(
            [[IMAGE_TOKEN] * 3 + [TEXT_TOKEN] * 2 + [IMAGE_TOKEN] * 2 + [TEXT_TOKEN]]
        )
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1), 5)

    def test_partial_depth_2_three_images(self):
        # [img×2, text, img×2, text, img×3]  partial_depth=2 → new_img_start=6 (third block)
        ids = mx.array(
            [
                [IMAGE_TOKEN] * 2
                + [TEXT_TOKEN]
                + [IMAGE_TOKEN] * 2
                + [TEXT_TOKEN]
                + [IMAGE_TOKEN] * 3
            ]
        )
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 2), 6)

    def test_partial_depth_equals_last_block(self):
        # partial_depth points to the very last block
        ids = mx.array(
            [[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]]
        )
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1), 3)

    def test_partial_depth_out_of_range_returns_len(self):
        # partial_depth beyond number of blocks → returns len(flat) (no new block found)
        ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
        result = compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 5)
        self.assertEqual(result, len(ids[0].tolist()))

    def test_partial_depth_with_video_tokens(self):
        # [vid×2, text, vid×2]  partial_depth=1 → new_img_start=3
        ids = mx.array(
            [[VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN]]
        )
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1), 3)

    def test_partial_depth_realistic_layout(self):
        system = [TEXT_TOKEN] * 5
        img1 = [IMAGE_TOKEN] * 256
        mid_text = [TEXT_TOKEN] * 10
        img2 = [IMAGE_TOKEN] * 128
        suffix = [TEXT_TOKEN] * 5
        ids = mx.array([system + img1 + mid_text + img2 + suffix])
        # partial_depth=1 → start of img2 block = 5 + 256 + 10 = 271
        self.assertEqual(compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1), 271)


if __name__ == "__main__":
    unittest.main()
