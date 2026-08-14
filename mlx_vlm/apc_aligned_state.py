"""Where to checkpoint a prompt so a divergent request can resume.

Recurrent and convolution state summarises every token before it, so it cannot
be sliced to an arbitrary position the way attention KV can. Reuse therefore
happens at boundaries: a request stores the whole cache at chosen prefix
lengths, and a later request sharing that prefix restores the longest boundary
it matches.

One checkpoint near the end of a prompt only helps a request that repeats it.
Boundaries spread across the prompt are what let a request that shares an
opening and then diverges start from the last common point, so the schedule
below walks a fixed stride and keeps every boundary that leaves a text-only
suffix.
"""

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
    """Prefix lengths at which to store a reusable checkpoint.

    Boundaries are multiples of ``stride`` that keep at least ``guard_tokens``
    to generate from, moved forward when needed so no media placeholder is left
    in the suffix. ``limit`` keeps the earliest boundaries when a prompt would
    otherwise produce more than the cache should hold, since a request that
    shares an opening and then diverges can only resume from an early one.
    """
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
