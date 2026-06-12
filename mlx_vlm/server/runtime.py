from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


class ModelCacheRegistry:
    def __init__(self, default_kind: str = "text_generation"):
        self.default_kind = default_kind
        self._caches: dict[str, dict] = {}

    def _kind(self, kind: Optional[str] = None) -> str:
        return kind or self.default_kind

    def for_kind(self, kind: Optional[str] = None) -> dict:
        return self._caches.get(self._kind(kind), {})

    def ensure_kind(self, kind: Optional[str] = None) -> dict:
        return self._caches.setdefault(self._kind(kind), {})

    def set(self, kind: str, cache: dict) -> dict:
        self._caches[self._kind(kind)] = cache
        return cache

    def pop(self, kind: str) -> Optional[dict]:
        return self._caches.pop(self._kind(kind), None)

    def clear(self, kind: Optional[str] = None) -> None:
        if kind is None:
            self._caches.clear()
        else:
            self._caches.pop(self._kind(kind), None)

    def get(self, key: str, default=None, *, kind: Optional[str] = None):
        return self.for_kind(kind).get(key, default)

    def __getitem__(self, key: str):
        return self.for_kind()[key]

    def __setitem__(self, key: str, value) -> None:
        self.ensure_kind()[key] = value

    def __delitem__(self, key: str) -> None:
        kind = self._kind()
        cache = self._caches[kind]
        del cache[key]
        if not cache:
            self._caches.pop(kind, None)

    def __contains__(self, key: str) -> bool:
        return key in self.for_kind()

    def __bool__(self) -> bool:
        return bool(self._caches)

    def items(self) -> Iterator[tuple[str, dict]]:
        return iter(self._caches.items())

    def values(self) -> Iterator[dict]:
        return iter(self._caches.values())


@dataclass
class ServerRuntime:
    model_cache: ModelCacheRegistry = field(default_factory=ModelCacheRegistry)
    response_generator: Optional[Any] = None
    audio_queue: Optional[Any] = None
    apc_manager: Optional[Any] = None
    metrics: Optional[Any] = None


runtime = ServerRuntime()
