from __future__ import annotations

from typing import Any

from .constants import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG
from .references import MiniMaxH3PreparedReference


def _tokenize(tokenizer: Any, value: str) -> list[int]:
    return tokenizer(value, add_special_tokens=False)["input_ids"]


def _vision_ids(tokenizer: Any, pad_token: str, num_tokens: int) -> list[int]:
    if num_tokens < 0:
        raise ValueError(f"vision token count must be non-negative, got {num_tokens}")
    return (
        [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
        + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
        + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
    )


def build_fl2va_presentation(
    tokenizer: Any,
    prompt: str,
    image_token_counts: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Build H3's prompt presentation for T2VA and FL2VA requests."""
    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be a single string, got {type(prompt)}")
    token_ids: list[int] = []
    token_tags: list[int] = []
    for index, num_tokens in enumerate(image_token_counts or []):
        label_ids = _tokenize(tokenizer, f"<Picture {index + 1}>: ")
        vision_ids = _vision_ids(tokenizer, "<|image_pad|>", num_tokens)
        token_ids.extend(label_ids)
        token_ids.extend(vision_ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(label_ids))
        token_tags.extend([MINIMAX_H3_VIDEO_TAG] * len(vision_ids))
    prompt_ids = _tokenize(tokenizer, prompt)
    token_ids.extend(prompt_ids)
    token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(prompt_ids))
    return token_ids, token_tags


def build_ref2va_presentation(
    tokenizer: Any,
    prompt: str,
    references: list[MiniMaxH3PreparedReference],
    image_token_counts: list[int],
    video_block_token_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Build H3's ordered, per-modality-numbered Ref2VA presentation."""
    if not isinstance(prompt, str):
        raise ValueError(f"prompt must be a single string, got {type(prompt)}")

    token_ids: list[int] = []
    token_tags: list[int] = []

    def emit_text(value: str) -> None:
        ids = _tokenize(tokenizer, value)
        token_ids.extend(ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(ids))

    def emit_vision(pad_token: str, num_tokens: int) -> None:
        ids = _vision_ids(tokenizer, pad_token, num_tokens)
        token_ids.extend(ids)
        token_tags.extend([MINIMAX_H3_VIDEO_TAG] * len(ids))

    expected_images = sum(reference.kind == "image" for reference in references)
    expected_videos = sum(reference.kind == "video" for reference in references)
    if len(image_token_counts) != expected_images:
        raise ValueError(
            f"expected {expected_images} image token counts, got {len(image_token_counts)}"
        )
    if len(video_block_token_counts) != expected_videos:
        raise ValueError(
            "expected "
            f"{expected_videos} video block token counts, got "
            f"{len(video_block_token_counts)}"
        )

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision("<|image_pad|>", image_token_counts[counts["image"] - 1])
        elif reference.kind == "video":
            counts["video"] += 1
            emit_text(f"<Video {counts['video']}>: ")
            num_tokens = video_block_token_counts[counts["video"] - 1]
            for timestamp in reference.block_timestamps:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision("<|video_pad|>", num_tokens)
    emit_text(prompt)
    return token_ids, token_tags


def create_mm_token_type_ids(
    token_ids: list[int],
    tokenizer: Any,
) -> list[int]:
    """Return Qwen3-VL modality IDs: text=0, image=1, and video=2."""
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    return [
        1 if token == image_id else 2 if token == video_id else 0 for token in token_ids
    ]


__all__ = [
    "build_fl2va_presentation",
    "build_ref2va_presentation",
    "create_mm_token_type_ids",
]
