from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import mlx.core as mx

from .constants import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_ROPE_FRAME_RESCALE,
    MINIMAX_H3_ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_VIDEO_TAG,
)
from .packing import (
    MiniMaxH3PackedSequence,
    _frame_position_grid,
    _temporal_position_grid,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_MAX_REFERENCE_IMAGES = 9
MINIMAX_H3_MAX_REFERENCE_VIDEOS = 3
MINIMAX_H3_MAX_REFERENCE_AUDIOS = 3
MINIMAX_H3_MAX_REFERENCES = 12


@dataclass(slots=True)
class MiniMaxH3Reference:
    image: Any = None
    video: Any = None
    fps: float | None = None
    audio: Any = None
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        media = [
            name
            for name in ("image", "video", "audio")
            if getattr(self, name) is not None
        ]
        if media not in (["image"], ["video"], ["audio"], ["video", "audio"]):
            raise ValueError(
                "a MiniMaxH3Reference must carry exactly one of image, video, "
                f"or audio, plus optional video soundtrack; got {media or 'none'}"
            )
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def kind(self) -> Literal["image", "video", "audio"]:
        if self.image is not None:
            return "image"
        return "video" if self.video is not None else "audio"

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


@dataclass(slots=True)
class MiniMaxH3PreparedReference:
    kind: Literal["image", "video", "audio"]
    has_audio: bool = False
    image: Any = None
    frames: Any = None
    waveform: mx.array | None = None
    block_timestamps: list[float] = field(default_factory=list)
    num_latent_frames: int = 1
    latent_height: int = 0
    latent_width: int = 0
    num_audio_latents: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ("image", "video", "audio"):
            raise ValueError(
                f"reference kind must be image, video, or audio, got {self.kind!r}"
            )
        if self.kind == "audio" and not self.has_audio:
            raise ValueError("an audio prepared reference must set has_audio=True")

    @property
    def num_video_rows(self) -> int:
        return (
            self.num_latent_frames
            * (self.latent_height // 2)
            * (self.latent_width // 2)
        )

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS


def validate_references(references: list[MiniMaxH3Reference]) -> None:
    if not references:
        raise ValueError("Ref2VA requires at least one reference")
    if len(references) > MINIMAX_H3_MAX_REFERENCES:
        raise ValueError(
            f"Ref2VA accepts at most {MINIMAX_H3_MAX_REFERENCES} references"
        )
    counts = {
        kind: sum(reference.kind == kind for reference in references)
        for kind in ("image", "video", "audio")
    }
    limits = {
        "image": MINIMAX_H3_MAX_REFERENCE_IMAGES,
        "video": MINIMAX_H3_MAX_REFERENCE_VIDEOS,
        "audio": MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    }
    for kind, count in counts.items():
        if count > limits[kind]:
            raise ValueError(
                f"Ref2VA accepts at most {limits[kind]} {kind} references, got {count}"
            )
    if counts["audio"] and not (counts["image"] or counts["video"]):
        raise ValueError("audio cannot be the only Ref2VA reference modality")


def trim_reference_num_frames(num_frames: int) -> int:
    """Snap down to a ``17 * n + 5`` frame count the visual VAE can encode."""
    if num_frames < 1:
        raise ValueError(
            f"a reference video must have at least one frame, got {num_frames}"
        )
    return (
        max(
            1,
            (num_frames - MINIMAX_H3_LATENTS_PER_CHUNK) // MINIMAX_H3_FRAMES_PER_CHUNK,
        )
        * MINIMAX_H3_FRAMES_PER_CHUNK
        + MINIMAX_H3_LATENTS_PER_CHUNK
    )


def _reference_temporal_position_span(num_latent_frames: int) -> float:
    # This call site is sequential in the reference implementation, unlike
    # the pairwise reduction used to position FL2VA's last-frame anchor.
    return sum(
        MINIMAX_H3_ROPE_FRAME_RESCALE
        * MINIMAX_H3_ROPE_FRAMES_PER_LATENT[
            index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)
        ]
        for index in range(num_latent_frames)
    )


def _audio_positions(
    num_audio_latents: int,
    rotary_time: float,
    width_grid: mx.array,
) -> mx.array:
    time = mx.array(rotary_time, dtype=mx.float64) + mx.arange(
        num_audio_latents,
        dtype=mx.float64,
    )
    return mx.stack(
        [
            mx.tile(time, MINIMAX_H3_AUDIO_CHANNELS),
            mx.zeros(
                (num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,),
                dtype=mx.float64,
            ),
            mx.concatenate(
                [
                    mx.full(
                        (num_audio_latents,),
                        width_grid[0],
                        dtype=mx.float64,
                    ),
                    mx.full(
                        (num_audio_latents,),
                        width_grid[-1],
                        dtype=mx.float64,
                    ),
                ]
            ),
        ],
        axis=-1,
    )


def _video_positions(
    num_latent_frames: int,
    rotary_time: float,
    frame_grid: mx.array,
) -> mx.array:
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    return mx.concatenate(
        [
            mx.repeat(frame_time, frame_grid.shape[0]).reshape(-1, 1),
            mx.tile(frame_grid, (num_latent_frames, 1)),
        ],
        axis=-1,
    )


def _build_ref2va_packed_sequence(
    text_token_tags: mx.array,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    _, patch_h, patch_w = patch_size
    num_text_tokens = int(text_token_tags.size)
    _, target_width_grid, target_frame_grid = _frame_position_grid(
        latent_height,
        latent_width,
        patch_h,
        patch_w,
    )

    positions = [
        mx.stack(
            [
                mx.arange(num_text_tokens, dtype=mx.float64),
                mx.zeros((num_text_tokens,), dtype=mx.float64),
                mx.zeros((num_text_tokens,), dtype=mx.float64),
            ],
            axis=-1,
        )
    ]
    tags = [text_token_tags.astype(mx.int32)]
    video_indices = []
    audio_indices = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)

    for reference in references:
        if reference.kind == "image":
            _, _, frame_grid = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            block = mx.concatenate(
                [
                    mx.full(
                        (reference.num_video_rows, 1),
                        mx.array(rotary_time, dtype=mx.float64),
                        dtype=mx.float64,
                    ),
                    frame_grid,
                ],
                axis=-1,
            )
            positions.append(block)
            tags.append(
                mx.full(
                    (reference.num_video_rows,),
                    MINIMAX_H3_VIDEO_TAG,
                    dtype=mx.int32,
                )
            )
            video_indices.append(
                mx.arange(cursor, cursor + reference.num_video_rows, dtype=mx.int32)
            )
            cursor += reference.num_video_rows
            rotary_time += 1.0
        elif reference.kind == "audio":
            block = _audio_positions(
                reference.num_audio_latents,
                rotary_time,
                target_width_grid,
            )
            positions.append(block)
            tags.append(
                mx.full(
                    (reference.num_audio_rows,),
                    MINIMAX_H3_AUDIO_TAG,
                    dtype=mx.int32,
                )
            )
            audio_indices.append(
                mx.arange(cursor, cursor + reference.num_audio_rows, dtype=mx.int32)
            )
            cursor += reference.num_audio_rows
            rotary_time += float(reference.num_audio_latents)
        elif reference.kind == "video":
            _, width_grid, frame_grid = _frame_position_grid(
                reference.latent_height,
                reference.latent_width,
                patch_h,
                patch_w,
            )
            audio_block = _audio_positions(
                reference.num_audio_latents,
                rotary_time,
                width_grid,
            )
            positions.append(audio_block)
            tags.append(
                mx.full(
                    (reference.num_audio_rows,),
                    MINIMAX_H3_AUDIO_TAG,
                    dtype=mx.int32,
                )
            )
            audio_indices.append(
                mx.arange(cursor, cursor + reference.num_audio_rows, dtype=mx.int32)
            )
            cursor += reference.num_audio_rows

            video_block = _video_positions(
                reference.num_latent_frames,
                rotary_time,
                frame_grid,
            )
            positions.append(video_block)
            tags.append(
                mx.full(
                    (reference.num_video_rows,),
                    MINIMAX_H3_VIDEO_TAG,
                    dtype=mx.int32,
                )
            )
            video_indices.append(
                mx.arange(cursor, cursor + reference.num_video_rows, dtype=mx.int32)
            )
            cursor += reference.num_video_rows
            rotary_time += max(
                float(reference.num_audio_latents),
                _reference_temporal_position_span(reference.num_latent_frames),
            )
        else:  # pragma: no cover - dataclass validation owns this branch
            raise ValueError(f"unsupported reference kind {reference.kind!r}")

    target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    target_audio_positions = _audio_positions(
        num_audio_latents,
        rotary_time,
        target_width_grid,
    )
    positions.append(target_audio_positions)
    tags.append(
        mx.full(
            (target_audio_rows,),
            MINIMAX_H3_AUDIO_TAG,
            dtype=mx.int32,
        )
    )
    audio_indices.append(mx.arange(cursor, cursor + target_audio_rows, dtype=mx.int32))
    cursor += target_audio_rows

    target_video_positions = _video_positions(
        num_latent_frames,
        rotary_time,
        target_frame_grid,
    )
    positions.append(target_video_positions)
    tags.append(
        mx.full(
            (target_video_positions.shape[0],),
            MINIMAX_H3_VIDEO_TAG,
            dtype=mx.int32,
        )
    )
    video_indices.append(
        mx.arange(cursor, cursor + target_video_positions.shape[0], dtype=mx.int32)
    )
    cursor += target_video_positions.shape[0]

    num_condition_video_rows = sum(
        reference.num_video_rows
        for reference in references
        if reference.kind != "audio"
    )
    num_condition_audio_rows = sum(reference.num_audio_rows for reference in references)
    return MiniMaxH3PackedSequence(
        sequence_length=cursor,
        position_ids=mx.concatenate(positions, axis=0),
        token_tags=mx.concatenate(tags, axis=0),
        video_indices=mx.concatenate(video_indices),
        audio_indices=mx.concatenate(audio_indices),
        text_indices=mx.arange(num_text_tokens, dtype=mx.int32),
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
    )


def build_ref2va_packed_sequence(
    text_token_tags: mx.array,
    references: list[MiniMaxH3PreparedReference],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
) -> MiniMaxH3PackedSequence:
    with mx.stream(mx.cpu):
        layout = _build_ref2va_packed_sequence(
            text_token_tags,
            references,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
        )
        mx.eval(
            layout.position_ids,
            layout.token_tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
        )
    return layout


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"reference image size must be positive, got {width}x{height}")
    if width > 4 * height or height > 4 * width:
        raise ValueError(
            f"reference image must be within 1:4 and 4:1, got {width}x{height}"
        )
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


__all__ = [
    "MINIMAX_H3_MAX_REFERENCE_AUDIOS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_REFERENCE_VIDEOS",
    "MINIMAX_H3_MAX_REFERENCES",
    "MINIMAX_H3_QWEN_TEMPORAL_PATCH",
    "MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS",
    "MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE",
    "MiniMaxH3PreparedReference",
    "MiniMaxH3Reference",
    "build_ref2va_packed_sequence",
    "resolve_reference_image_size",
    "validate_references",
]
