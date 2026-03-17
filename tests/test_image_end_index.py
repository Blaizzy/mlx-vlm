"""
Unit tests for image_end_index computation in qwen3_5.get_input_embeddings.

Tests the core logic: given input_ids with image/video tokens at known positions,
image_end_index should point to the first non-visual token after the image block.
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
