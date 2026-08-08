from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from .constants import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_ROPE_FRAME_RESCALE,
    MINIMAX_H3_ROPE_FRAMES_PER_LATENT,
    MINIMAX_H3_ROPE_SPATIAL_SCALE,
    MINIMAX_H3_SHORT_EDGE,
    MINIMAX_H3_VIDEO_TAG,
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3PackedSequence:
    sequence_length: int
    position_ids: mx.array
    token_tags: mx.array
    video_indices: mx.array
    audio_indices: mx.array
    text_indices: mx.array
    num_condition_video_rows: int
    num_condition_audio_rows: int


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
) -> tuple[int, int]:
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"aspect ratio must be positive, got {aspect_width}:{aspect_height}"
        )
    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            "MiniMax-H3 supports aspect ratios from 1:4 to 4:1, "
            f"got {aspect_width}:{aspect_height} ({ratio:g})"
        )

    if ratio >= 1.0:
        width = MINIMAX_H3_SHORT_EDGE * ratio
        height = float(MINIMAX_H3_SHORT_EDGE)
    else:
        width = float(MINIMAX_H3_SHORT_EDGE)
        height = MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = math.sqrt(MINIMAX_H3_MAX_PIXELS / area)
        width *= scale
        height *= scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def align_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    remainder = num_frames % MINIMAX_H3_FRAMES_PER_CHUNK
    increment = (MINIMAX_H3_LATENTS_PER_CHUNK - remainder) % (
        MINIMAX_H3_FRAMES_PER_CHUNK
    )
    return num_frames + increment


def video_latent_num_frames(num_frames: int) -> int:
    if num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be of the form 17 * n + 5, got {num_frames}")
    return (
        num_frames - MINIMAX_H3_LATENTS_PER_CHUNK
    ) // MINIMAX_H3_FRAMES_PER_CHUNK * MINIMAX_H3_LATENTS_PER_CHUNK + 2


def audio_latent_num_frames(num_frames: int) -> int:
    return int(round(num_frames / MINIMAX_H3_FPS * MINIMAX_H3_AUDIO_LATENTS_PER_SECOND))


def patchify_video_latents(
    latents: mx.array,
    patch_size: tuple[int, int, int],
) -> mx.array:
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"latents of shape {latents.shape} are not divisible by patch {patch_size}"
        )
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w)


def unpatchify_video_tokens(
    rows: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    channels: int,
    patch_size: tuple[int, int, int],
) -> mx.array:
    patch_t, patch_h, patch_w = patch_size
    rows = rows.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    rows = rows.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return rows.reshape(
        -1,
        channels,
        num_latent_frames,
        latent_height,
        latent_width,
    )


def unpack_audio_tokens(rows: mx.array, num_audio_latents: int) -> mx.array:
    rows = rows.reshape(
        MINIMAX_H3_AUDIO_CHANNELS,
        num_audio_latents,
        rows.shape[-1],
    )
    return rows.transpose(0, 2, 1)


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> mx.array:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    count = dim // patch
    # Python float scalars are weakly typed by MLX and may be rounded through
    # float32 even when the array is float64. Materialize every scalar at the
    # reference dtype to match NumPy's float64 linspace exactly.
    left = mx.array(left, dtype=mx.float64)
    step = mx.array(ratio / count, dtype=mx.float64)
    scale = mx.array(MINIMAX_H3_ROPE_SPATIAL_SCALE, dtype=mx.float64)
    return (left + mx.arange(count, dtype=mx.float64) * step) * scale


def _temporal_position_grid(num_latent_frames: int, origin: float) -> mx.array:
    spans = mx.array(
        [
            MINIMAX_H3_ROPE_FRAME_RESCALE
            * MINIMAX_H3_ROPE_FRAMES_PER_LATENT[
                index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)
            ]
            for index in range(num_latent_frames)
        ],
        dtype=mx.float64,
    )
    origin = mx.array(origin, dtype=mx.float64)
    return origin + mx.concatenate(
        [mx.zeros((1,), dtype=mx.float64), mx.cumsum(spans[:-1])]
    )


def _temporal_position_span(num_latent_frames: int) -> float:
    spans = [
        MINIMAX_H3_ROPE_FRAME_RESCALE
        * MINIMAX_H3_ROPE_FRAMES_PER_LATENT[
            index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)
        ]
        for index in range(num_latent_frames)
    ]
    return float(mx.sum(mx.array(spans, dtype=mx.float64)).item())


def _frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int,
    patch_w: int,
) -> tuple[mx.array, mx.array, mx.array]:
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    heights, widths = mx.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = mx.stack([heights.reshape(-1), widths.reshape(-1)], axis=-1)
    return height_grid, width_grid, frame_grid


def _build_packed_sequence(
    text_token_tags: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    _, patch_h, patch_w = patch_size
    if latent_height % patch_h or latent_width % patch_w:
        raise ValueError(
            f"latent shape {(latent_height, latent_width)} is not divisible by "
            f"patch {patch_size}"
        )
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = int(text_token_tags.size)
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = (
        num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows
    )

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    text_positions = mx.stack(
        [
            mx.arange(num_text_tokens, dtype=mx.float64),
            mx.zeros((num_text_tokens,), dtype=mx.float64),
            mx.zeros((num_text_tokens,), dtype=mx.float64),
        ],
        axis=-1,
    )
    _, width_grid, frame_grid = _frame_position_grid(
        latent_height,
        latent_width,
        patch_h,
        patch_w,
    )

    condition_positions = []
    for anchor in keyframe_anchors:
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            anchor_time = (
                float(num_text_tokens)
                + _temporal_position_span(num_latent_frames)
                - MINIMAX_H3_ROPE_FRAME_RESCALE
            )
        else:
            raise ValueError(
                f"keyframe anchor must be 'first' or 'last', got {anchor!r}"
            )
        condition_positions.append(
            mx.concatenate(
                [
                    mx.full(
                        (rows_per_frame, 1),
                        mx.array(anchor_time, dtype=mx.float64),
                        dtype=mx.float64,
                    ),
                    frame_grid,
                ],
                axis=-1,
            )
        )

    audio_time = float(num_text_tokens) + mx.arange(
        num_audio_latents,
        dtype=mx.float64,
    )
    audio_positions = mx.stack(
        [
            mx.tile(audio_time, MINIMAX_H3_AUDIO_CHANNELS),
            mx.zeros((num_audio_rows,), dtype=mx.float64),
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

    video_times = _temporal_position_grid(
        num_latent_frames,
        float(num_text_tokens),
    )
    video_positions = mx.concatenate(
        [
            mx.repeat(video_times, rows_per_frame).reshape(-1, 1),
            mx.tile(frame_grid, (num_latent_frames, 1)),
        ],
        axis=-1,
    )
    position_blocks = [text_positions, *condition_positions, audio_positions]
    position_blocks.append(video_positions)
    position_ids = mx.concatenate(position_blocks, axis=0)

    video_indices = mx.concatenate(
        [
            mx.arange(condition_start, audio_start, dtype=mx.int32),
            mx.arange(video_start, sequence_length, dtype=mx.int32),
        ]
    )
    audio_indices = mx.arange(audio_start, video_start, dtype=mx.int32)
    text_indices = mx.arange(num_text_tokens, dtype=mx.int32)
    token_tags = mx.concatenate(
        [
            text_token_tags.astype(mx.int32),
            mx.full(
                (num_condition_rows,),
                MINIMAX_H3_VIDEO_TAG,
                dtype=mx.int32,
            ),
            mx.full(
                (num_audio_rows,),
                MINIMAX_H3_AUDIO_TAG,
                dtype=mx.int32,
            ),
            mx.full(
                (num_video_rows,),
                MINIMAX_H3_VIDEO_TAG,
                dtype=mx.int32,
            ),
        ]
    )
    return MiniMaxH3PackedSequence(
        sequence_length=sequence_length,
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_rows,
        num_condition_audio_rows=0,
    )


def build_packed_sequence(
    text_token_tags: mx.array,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    keyframe_anchors: tuple[str, ...] = (),
) -> MiniMaxH3PackedSequence:
    # MLX Metal does not support float64. H3's shared A/V clock requires it,
    # so construct and materialize the structural layout on MLX's CPU stream.
    with mx.stream(mx.cpu):
        layout = _build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            keyframe_anchors,
        )
        mx.eval(
            layout.position_ids,
            layout.token_tags,
            layout.video_indices,
            layout.audio_indices,
            layout.text_indices,
        )
    return layout


def build_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[mx.array, mx.array]:
    row_timesteps = mx.full(
        (layout.sequence_length,),
        video_timestep,
        dtype=mx.float32,
    )
    condition_video_indices = layout.video_indices[: layout.num_condition_video_rows]
    target_audio_indices = layout.audio_indices[layout.num_condition_audio_rows :]
    condition_audio_indices = layout.audio_indices[: layout.num_condition_audio_rows]
    row_timesteps = row_timesteps.at[condition_video_indices].add(
        condition_video_timestep - video_timestep
    )
    row_timesteps = row_timesteps.at[target_audio_indices].add(
        audio_timestep - video_timestep
    )
    row_timesteps = row_timesteps.at[condition_audio_indices].add(
        condition_audio_timestep - video_timestep
    )

    sorted_values = sorted(set(row_timesteps.tolist()))
    timesteps = mx.array(sorted_values, dtype=mx.float32)
    indices = mx.zeros((layout.sequence_length,), dtype=mx.int32)
    for index, value in enumerate(sorted_values):
        indices = mx.where(row_timesteps == value, index, indices)
    return timesteps, indices


__all__ = [
    "MiniMaxH3PackedSequence",
    "align_num_frames",
    "audio_latent_num_frames",
    "build_packed_sequence",
    "build_row_timesteps",
    "patchify_video_latents",
    "resolve_canvas_size",
    "unpack_audio_tokens",
    "unpatchify_video_tokens",
    "video_latent_num_frames",
]
