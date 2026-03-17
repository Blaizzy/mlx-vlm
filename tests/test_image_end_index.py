"""
Unit tests for qwen3_5 image token boundary computations.

Covers two functions:
- image_end_index: end of the last image block (used by get_input_embeddings)
- new_img_start: start of the first *new* image block at partial_depth
  (used by get_partial_input_embeddings for partial image KV cache)
"""

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


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


def test_image_tokens_at_start():
    # [img, img, img, text, text]  → image_end_index = 3
    ids = mx.array([[IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 3


def test_image_tokens_in_middle():
    # [text, img, img, text, text]  → image_end_index = 3
    ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 3


def test_image_tokens_at_end():
    # [text, text, img, img]  → image_end_index = 4
    ids = mx.array([[TEXT_TOKEN, TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 4


def test_single_image_token():
    # [text, img, text]  → image_end_index = 2
    ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 2


# ---------------------------------------------------------------------------
# Video tokens (same mask logic)
# ---------------------------------------------------------------------------


def test_video_tokens():
    # [text, vid, vid, vid, text]  → image_end_index = 4
    ids = mx.array([[TEXT_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 4


def test_mixed_image_and_video_tokens():
    # [img, img, vid, vid, text]  → image_end_index = 4
    ids = mx.array([[IMAGE_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 4


# ---------------------------------------------------------------------------
# No-image path
# ---------------------------------------------------------------------------


def test_no_image_tokens_returns_none():
    ids = mx.array([[TEXT_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) is None


# ---------------------------------------------------------------------------
# Realistic Qwen3.5 prompt layout
# system_tokens + image_tokens + user_tokens
# ---------------------------------------------------------------------------


def test_realistic_prompt_layout():
    system = [TEXT_TOKEN] * 10
    image = [IMAGE_TOKEN] * 256  # typical image patch count
    user = [TEXT_TOKEN] * 20
    ids = mx.array([system + image + user])
    # last image token is at index 10 + 256 - 1 = 265
    assert compute_image_end_index(ids, IMAGE_TOKEN, VIDEO_TOKEN) == 266


# ---------------------------------------------------------------------------
# new_img_start — start of the first new image block at partial_depth
# Extracted from qwen3_5.get_partial_input_embeddings — the logic under test.
# ---------------------------------------------------------------------------


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


def test_partial_depth_0_single_image():
    # [text, img, img, text]  partial_depth=0 → new_img_start=1 (first image block)
    ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 0) == 1


def test_partial_depth_1_two_images():
    # [img×3, text×2, img×2, text]  partial_depth=1 → new_img_start=5 (second block)
    ids = mx.array(
        [[IMAGE_TOKEN] * 3 + [TEXT_TOKEN] * 2 + [IMAGE_TOKEN] * 2 + [TEXT_TOKEN]]
    )
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1) == 5


def test_partial_depth_2_three_images():
    # [img×2, text, img×2, text, img×3]  partial_depth=2 → new_img_start=7 (third block)
    ids = mx.array(
        [
            [IMAGE_TOKEN] * 2
            + [TEXT_TOKEN]
            + [IMAGE_TOKEN] * 2
            + [TEXT_TOKEN]
            + [IMAGE_TOKEN] * 3
        ]
    )
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 2) == 6


def test_partial_depth_equals_last_block():
    # partial_depth points to the very last block
    ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN]])
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1) == 3


def test_partial_depth_out_of_range_returns_len():
    # partial_depth beyond number of blocks → returns len(flat) (no new block found)
    ids = mx.array([[TEXT_TOKEN, IMAGE_TOKEN, TEXT_TOKEN]])
    result = compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 5)
    assert result == len(ids[0].tolist())


def test_partial_depth_with_video_tokens():
    # [vid×2, text, vid×2]  partial_depth=1 → new_img_start=3
    ids = mx.array([[VIDEO_TOKEN, VIDEO_TOKEN, TEXT_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN]])
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1) == 3


def test_partial_depth_realistic_layout():
    system = [TEXT_TOKEN] * 5
    img1 = [IMAGE_TOKEN] * 256
    mid_text = [TEXT_TOKEN] * 10
    img2 = [IMAGE_TOKEN] * 128
    suffix = [TEXT_TOKEN] * 5
    ids = mx.array([system + img1 + mid_text + img2 + suffix])
    # partial_depth=1 → start of img2 block = 5 + 256 + 10 = 271
    assert compute_new_img_start(ids, IMAGE_TOKEN, VIDEO_TOKEN, 1) == 271


if __name__ == "__main__":
    import sys

    tests = [
        test_image_tokens_at_start,
        test_image_tokens_in_middle,
        test_image_tokens_at_end,
        test_single_image_token,
        test_video_tokens,
        test_mixed_image_and_video_tokens,
        test_no_image_tokens_returns_none,
        test_realistic_prompt_layout,
        test_partial_depth_0_single_image,
        test_partial_depth_1_two_images,
        test_partial_depth_2_three_images,
        test_partial_depth_equals_last_block,
        test_partial_depth_out_of_range_returns_len,
        test_partial_depth_with_video_tokens,
        test_partial_depth_realistic_layout,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
