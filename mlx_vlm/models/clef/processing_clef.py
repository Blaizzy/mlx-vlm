# Adapted from Cloudflare/clef joint_schema_model.py (Apache-2.0).
# Copyright 2026 Cloudflare. See LICENSE in this directory.
"""Clef's trained record format, with explicit overflow errors."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = (
    "<|vision_start|><|vision_start|><|video_pad|><|vision_end|><|vision_end|>"
)
MEDIA_BATCH_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)
MEDIA_TOKEN_KEYS = ("mm_token_type_ids",)
QUESTION_TYPES = {"bool": 0, "noul": 0, "choice": 1, "score": 2}


def render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def question_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = str(question["type"])
    if question_type in ("noul", "bool"):
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if question_type == "choice":
        criteria = question["criteria"]
        if isinstance(criteria, (list, tuple)):
            criteria = dict.fromkeys(criteria)
        return sorted((str(key), value) for key, value in criteria.items())
    return [(str(index), value) for index, value in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str
    prefix_length: int = 0
    media: dict[str, Any] | None = dataclass_field(
        default=None, compare=False, repr=False
    )


def _tokens(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer(text, add_special_tokens=False)
    return result["input_ids"] if isinstance(result, Mapping) else result.input_ids


def _encode_media(
    processor: Any, record: dict[str, Any], checkpoint
) -> tuple[list[int], dict[str, Any] | None]:
    images = list(record.get("images") or [])
    videos = list(record.get("videos") or [])
    if not images and not videos:
        return [], None
    if processor is None:
        raise ValueError("records with images or videos require a processor")
    import numpy as np

    from ...utils import load_image

    def image(value):
        checkpoint()
        return load_image(value) if isinstance(value, str) else value

    images = [image(value) for value in images]
    videos = [[image(frame) for frame in frames] for frames in videos]
    checkpoint()
    media_kwargs = dict(record.get("media_kwargs") or {})
    if videos and "video_metadata" not in media_kwargs:
        video_processor = processor.video_processor
        fps = media_kwargs.pop("fps", None)
        num_frames = media_kwargs.pop("num_frames", None)
        metadata = []
        for index, frames in enumerate(videos):
            total = len(frames)
            count = num_frames
            if count is None:
                count = int(total / 24 * (fps or video_processor.fps))
                count = min(
                    max(count, video_processor.min_frames),
                    video_processor.max_frames,
                    total,
                )
            indices = np.linspace(0, total - 1, count).round().astype(int)
            videos[index] = [frames[i] for i in indices]
            metadata.append({"frames_indices": indices.tolist(), "fps": 24})
        media_kwargs["video_metadata"] = metadata
    text = IMAGE_PLACEHOLDER * len(images) + VIDEO_PLACEHOLDER * len(videos) + "\n"
    encoded = processor(
        text=[text],
        images=images or None,
        videos=videos or None,
        return_tensors="np",
        **media_kwargs,
    )
    media = {key: encoded[key] for key in MEDIA_BATCH_KEYS if key in encoded}
    for key in MEDIA_TOKEN_KEYS:
        if key in encoded:
            media[key] = encoded[key][0].tolist()
    return encoded["input_ids"][0].tolist(), media


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    max_length: int = 65536,
    max_state_tokens: int | None = None,
    processor: Any | None = None,
    checkpoint=lambda: None,
) -> EncodedRecord:
    record = {
        **record,
        "questions": {
            name: {
                **question,
                "type": "noul" if question["type"] == "bool" else question["type"],
            }
            for name, question in record["questions"].items()
        },
    }
    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    questions: list[EncodedQuestion] = []
    for question_index, (question_id, question) in enumerate(
        record["questions"].items()
    ):
        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_start = len(schema_ids)
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(_tokens(tokenizer, render(instructions)))
        question_end = len(schema_ids)
        if question_end == question_start:
            raise ValueError("A question needs nonempty instructions or a nonempty ID")
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []
        for option_index, (option_id, description) in enumerate(
            question_options(question)
        ):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(_tokens(tokenizer, render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))
        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[str(question["type"])],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    media_ids, media = _encode_media(processor, record, checkpoint)
    checkpoint()
    if media is not None:
        media["token_offset"] = len(prefix_ids)
        prefix_ids = prefix_ids + media_ids
    state_ids = _tokens(tokenizer, render(record["state"]))
    if max_state_tokens is not None:
        if len(state_ids) > max_state_tokens:
            raise ValueError(f"State exceeds {max_state_tokens} tokens")
    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if fixed_length > max_length:
        raise ValueError(
            f"schema requires {fixed_length} tokens before state; maximum is {max_length}"
        )
    if fixed_length + len(state_ids) > max_length:
        raise ValueError(
            f"Input requires {fixed_length + len(state_ids)} tokens; maximum is {max_length}"
        )
    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=(
                question.question_span[0] + schema_offset,
                question.question_span[1] + schema_offset,
            ),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in questions
    )
    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    if not input_ids or not shifted_questions:
        raise ValueError("record produced no model input or questions")
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(record.get("id", "unknown")),
        prefix_length=schema_offset,
        media=media,
    )
