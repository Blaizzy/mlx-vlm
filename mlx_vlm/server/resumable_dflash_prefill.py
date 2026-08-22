"""Cache-safe continuation state for segmented DFlash prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence


@dataclass
class DFlashPrefillContinuation:
    request_id: str
    prompt_cache: Sequence[Any]
    remaining_input_ids: Any
    remaining_embeds: Any
    prompt_kwargs: Dict[str, Any]
    cached_tokens: int
    full_input_ids: Sequence[int]
    processed_uncached: int = 0
    segment_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_tokens(self) -> int:
        return int(self.remaining_input_ids.shape[1])

    @property
    def ready_for_final(self) -> bool:
        return self.remaining_tokens == 1

    def next_segment_tokens(self, limit: int) -> int:
        if limit < 1:
            raise ValueError("limit must be positive")
        return max(0, min(int(limit), self.remaining_tokens - 1))

    def commit_segment(
        self,
        tokens: int,
        *,
        next_ids: Any,
        next_embeds: Any,
        next_kwargs: Dict[str, Any],
    ) -> None:
        if tokens < 1 or tokens >= self.remaining_tokens:
            raise ValueError("segment must leave one token for the final forward")
        self.remaining_input_ids = next_ids
        self.remaining_embeds = next_embeds
        self.prompt_kwargs = next_kwargs
        self.processed_uncached += int(tokens)
        self.segment_count += 1
