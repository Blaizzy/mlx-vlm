"""Checkpoint boundaries for caches whose state cannot be sliced."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .apc import adjust_prefix_to_text_suffix_boundary

__all__ = ["checkpoint_schedule"]


def checkpoint_schedule(
    token_ids: Sequence[int],
    stride: int,
    media_token_ids: Iterable[int] = (),
    *,
    guard_tokens: int = 1,
    limit: Optional[int] = None,
) -> List[int]:
    """Prefix lengths at which to store a reusable checkpoint."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    media_token_ids = tuple(media_token_ids)
    highest = len(token_ids) - max(1, guard_tokens)
    if highest <= 0:
        return []

    boundaries: List[int] = []
    for candidate in range(stride, highest + 1, stride):
        safe = adjust_prefix_to_text_suffix_boundary(
            token_ids,
            candidate,
            media_token_ids,
            max_prefix_tokens=highest,
        )
        if safe <= 0:
            continue
        if boundaries and safe <= boundaries[-1]:
            continue
        boundaries.append(safe)

    if limit is not None and limit >= 0 and len(boundaries) > limit:
        boundaries = boundaries[:limit]
    return boundaries
