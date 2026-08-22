"""Transactional admission for resumable prompt prefill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class TailWork:
    """Prompt work remaining after a validated prefix-cache restore."""

    prompt_tokens: int
    cached_tokens: int = 0

    def __post_init__(self):
        if self.prompt_tokens < 1:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= self.cached_tokens < self.prompt_tokens:
            raise ValueError("cached_tokens must leave at least one tail token")

    @property
    def uncached_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


@dataclass
class ResumablePrefill:
    """Opaque cache/input state owned by the generation thread."""

    request_id: str
    work: TailWork
    cache_state: Any
    remaining_input: Any
    processed_tokens: int = 0
    deferrals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_tokens(self) -> int:
        return self.work.uncached_tokens - self.processed_tokens

    @property
    def complete(self) -> bool:
        return self.remaining_tokens == 0

    def commit_segment(self, tokens: int, next_remaining_input: Any) -> None:
        """Commit only after the generation thread materializes cache writes."""
        if tokens < 1 or tokens > self.remaining_tokens:
            raise ValueError("segment must be within remaining uncached tail")
        self.processed_tokens += int(tokens)
        self.remaining_input = next_remaining_input
        self.deferrals = 0


class TailFairScheduler:
    """Shortest-remaining-tail scheduling with bounded starvation."""

    def __init__(self, *, slice_tokens: int, max_deferrals: int = 3):
        if slice_tokens < 1:
            raise ValueError("slice_tokens must be positive")
        if max_deferrals < 0:
            raise ValueError("max_deferrals must be non-negative")
        self.slice_tokens = int(slice_tokens)
        self.max_deferrals = int(max_deferrals)
        self._pending: list[ResumablePrefill] = []

    def enqueue(self, item: ResumablePrefill) -> None:
        if item.complete:
            raise ValueError("cannot schedule a completed prefill")
        self._pending.append(item)

    def __len__(self) -> int:
        return len(self._pending)

    def choose(self) -> Optional[tuple[ResumablePrefill, int]]:
        if not self._pending:
            return None
        forced = [x for x in self._pending if x.deferrals >= self.max_deferrals]
        chosen = (
            forced[0]
            if forced
            else min(self._pending, key=lambda x: x.remaining_tokens)
        )
        self._pending.remove(chosen)
        for other in self._pending:
            other.deferrals += 1
        return chosen, min(self.slice_tokens, chosen.remaining_tokens)

    def requeue(self, item: ResumablePrefill) -> None:
        if not item.complete:
            self._pending.append(item)

    def pending_ids(self) -> Iterable[str]:
        return tuple(x.request_id for x in self._pending)

    def cancel(self, request_id: str) -> Optional[ResumablePrefill]:
        for index, item in enumerate(self._pending):
            if item.request_id == request_id:
                return self._pending.pop(index)
        return None

    def drain(self) -> tuple[ResumablePrefill, ...]:
        pending = tuple(self._pending)
        self._pending.clear()
        return pending
