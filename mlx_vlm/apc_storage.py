"""Component-major storage primitives for Automatic Prefix Caching (issue #1629)."""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

import mlx.core as mx

ComponentId = str


def _array_bytes(t: mx.array) -> int:
    nbytes = getattr(t, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return int(t.size) * int(t.itemsize)
    except Exception:
        return 0


@runtime_checkable
class StateHandle(Protocol):
    """Physical storage for one component of one logical cache node.

    A handle owns its tensors plus their resident-byte accounting,
    serialization, and release. The logical index (hashes, parent links,
    lock counts) lives on the node and never touches tensors directly.
    """

    def resident_bytes(self) -> int: ...

    def release(self) -> None: ...


class KVBlockHandle:
    """Per-layer float K/V slabs for a node's pageable KV component."""

    __slots__ = ("keys", "values")

    def __init__(
        self,
        keys: Optional[List[mx.array]] = None,
        values: Optional[List[mx.array]] = None,
    ):
        self.keys = keys
        self.values = values

    def resident_bytes(self) -> int:
        total = 0
        for group in (self.keys, self.values):
            for t in group or ():
                total += _array_bytes(t)
        return total

    def release(self) -> None:
        self.keys = None
        self.values = None


class APCNode:
    """Logical prefix-index entry: identity plus per-component storage handles.

    Physical tensors live only inside the handles in ``components``; the node
    carries the logical index. Concrete nodes supply the identity fields and a
    ``components`` dict, and expose them through the node-view accessors below.
    """

    KV_COMPONENT: ComponentId = "kv"

    components: Dict[ComponentId, StateHandle]

    def kv_handle(self) -> Optional[KVBlockHandle]:
        handle = self.components.get(self.KV_COMPONENT)
        return handle if isinstance(handle, KVBlockHandle) else None

    @property
    def keys(self) -> Optional[List[mx.array]]:
        handle = self.kv_handle()
        return handle.keys if handle is not None else None

    @property
    def values(self) -> Optional[List[mx.array]]:
        handle = self.kv_handle()
        return handle.values if handle is not None else None

    def set_kv(self, keys: List[mx.array], values: List[mx.array]) -> None:
        self.components[self.KV_COMPONENT] = KVBlockHandle(keys, values)

    def release_components(self) -> None:
        for handle in self.components.values():
            handle.release()
        self.components.clear()

    def resident_bytes(self) -> int:
        return sum(handle.resident_bytes() for handle in self.components.values())
