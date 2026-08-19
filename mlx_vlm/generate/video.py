"""Frames fallback for processors without native video support.

A ``videos`` parameter in a processor signature is not enough to know video is
actually processed: the generic HF processor skeleton accepts ``videos`` but
only processes them when a ``video_processor`` component exists, and silently
discards them otherwise; the model then hallucinates an answer with no visual
input at all. When that would happen, the caller samples the video into frames
and sends them as ordered images, which any image-capable model handles.
``resolve_video_inputs`` is the shared entrypoint for that generic fallback so
CLI and server callers apply the same capability check and global frame budget.

Models whose vision patches carry two temporal slots can opt into a richer
fallback by defining a ``prepare_video_frame_pairs(processor, second_frames)``
method (the capability flag): the caller then feeds adjacent-frame pairs from
``pair_adjacent_frames`` instead of independent stills, and merges the hook's
returned generation kwargs.
"""

import inspect
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VideoInputResolution:
    """Image/video inputs after resolving the generic frames fallback."""

    images: list
    videos: list
    used_fallback: bool = False
    sampled_count: int = 0
    selected_count: int = 0
    frame_fps: float | None = None


def processor_handles_video(processor) -> bool:
    """True when the processor genuinely consumes a ``videos`` kwarg: the
    parameter is declared and a ``video_processor`` component exists (check
    the component, not just the parameter, see module docstring)."""
    method = getattr(processor, "process", processor)
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        params = {}
    has_component = getattr(processor, "video_processor", None) is not None
    return "videos" in params and has_component


def sample_video_frames(videos, fps=2.0):
    """Decode ``videos`` at ``fps`` into one chronological list of PIL frames,
    returned alongside the sampling fps actually used."""
    from ..utils import load_video

    frames = []
    frame_fps = fps
    for video in videos:
        arr, sampled_fps = load_video(str(video), fps=fps)
        frame_fps = sampled_fps or frame_fps
        for frame in arr:
            frames.append(Image.fromarray(np.transpose(frame, (1, 2, 0))))
    return frames, frame_fps


def subsample_evenly(frames, max_frames):
    """Keep at most ``max_frames``, evenly spaced across the clip. Beyond a
    couple dozen near-identical frames, models start describing a static scene
    mixture instead of the sequence, so more frames costs tokens and loses the
    temporal thread."""
    if len(frames) <= max_frames:
        return frames
    idxs = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[i] for i in idxs]


def resolve_video_inputs(
    processor,
    videos=None,
    *,
    images=None,
    fps=2.0,
    max_frames=16,
) -> VideoInputResolution:
    """Resolve unsupported videos into a globally capped list of still images.

    Native-video processors receive their inputs unchanged. Otherwise all videos
    are decoded first, then subsampled once so ``max_frames`` is a request-wide
    budget rather than a per-video budget. Inputs are copied and only returned
    after the complete conversion succeeds, avoiding partially converted media.
    """
    resolved_images = list(images or [])
    resolved_videos = list(videos or [])
    if not resolved_videos or processor_handles_video(processor):
        return VideoInputResolution(
            images=resolved_images,
            videos=resolved_videos,
        )

    max_frames = max(2, int(max_frames or 16))
    frames, frame_fps = sample_video_frames(resolved_videos, fps or 2.0)
    sampled_count = len(frames)
    if not sampled_count:
        raise ValueError("Video frame fallback decoded no frames.")
    selected = subsample_evenly(frames, max_frames)
    return VideoInputResolution(
        images=resolved_images + selected,
        videos=[],
        used_fallback=True,
        sampled_count=sampled_count,
        selected_count=len(selected),
        frame_fps=frame_fps,
    )


def pair_adjacent_frames(frames, max_pairs):
    """Anchor up to ``max_pairs`` adjacent-frame pairs, evenly spaced across
    the full sampled sequence. Pairs must be ADJACENT sampled frames: small
    displacement between the two temporal slots matches how consecutive video
    frames relate, while temporally distant frames in one patch read as a
    double exposure instead of motion. Returns
    ``(anchor_indices, first_frames, second_frames)``."""
    frames = list(frames)
    if len(frames) % 2:
        frames.append(frames[-1])
    n_pairs = max(2, min(max_pairs, len(frames) // 2))
    anchors = [
        min(round(i * (len(frames) - 2) / max(n_pairs - 1, 1)), len(frames) - 2)
        for i in range(n_pairs)
    ]
    return anchors, [frames[a] for a in anchors], [frames[a + 1] for a in anchors]


def timestamped_frame_messages(user_text, system, still_count, timestamps):
    """Chat messages placing every frame in ONE user message, each labeled with
    its clip time. Bare image entities in separate messages read as a slideshow
    ("a sequence of still images"); timestamp labels make the model ground
    observations chronologically and track how positions change. The chat
    template renders interleaved text/image parts natively, so this stays
    template-faithful."""
    parts = [{"type": "image"} for _ in range(still_count)]
    parts.append(
        {
            "type": "text",
            "text": "Here is a video as a sequence of frames in chronological order.",
        }
    )
    for t in timestamps:
        parts.append({"type": "text", "text": f"frame at t={t:.1f}s:"})
        parts.append({"type": "image"})
    parts.append({"type": "text", "text": user_text})
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": parts}]
