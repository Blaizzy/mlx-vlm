"""Pluggable cache-component adapters for Automatic Prefix Caching (APC).

Phase 1 of the APC redesign (issue #1629): a capability-driven adapter layer so
APC does not need a central allowlist of concrete cache classes.

This module provides:

* :class:`Capability` — how a cache component can be reused.
* :class:`PrefixStateAdapter` — the capture/restore/merge/(de)serialize contract.
* :class:`CheckpointAdapter` — the universal fallback built on the
  ``prefix_cache_snapshot`` / ``prefix_cache_restore`` contract on ``_BaseCache``
  (i.e. ``state`` + ``meta_state``).
* :class:`CompositeAdapter` — recurses into ``CacheList`` / tuple caches.
* a capability registry + :func:`resolve_adapter` / :func:`resolve_capability`.
* :func:`build_prefix_cache_plan` — resolves one adapter per cache entry a model
  produces, without APC knowing the concrete classes.

It intentionally does not yet change APC storage or behaviour (that is phase 2).
The key correctness rule enforced here (issue #1629): an *unregistered*
``KVCache`` subclass is **not** assumed pageable via inheritance — it falls back
to the conservative checkpoint capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import mlx.core as mx


class Capability(str, Enum):
    """Reuse capability of a single cache component."""

    PAGEABLE = "pageable"  # token-aligned KV, splittable + concatenable
    WINDOWED = "windowed"  # adapter-defined retained window/anchors
    CHECKPOINT = "checkpoint"  # opaque state valid only at an exact boundary
    COMPOSITE = "composite"  # container (CacheList/tuple) of sub-components
    UNSUPPORTED = "unsupported"  # no restorable contract


def _copy_array(x: mx.array) -> mx.array:
    """Materialize ``x`` into a fresh MLX-owned contiguous buffer (detach)."""
    return mx.contiguous(mx.array(x, dtype=x.dtype))


def _snapshot_tree(obj: Any) -> Any:
    """Deep-copy the MLX arrays inside a state tree; pass scalars/None through."""
    if isinstance(obj, mx.array):
        return _copy_array(obj)
    if isinstance(obj, tuple):
        return tuple(_snapshot_tree(o) for o in obj)
    if isinstance(obj, list):
        return [_snapshot_tree(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _snapshot_tree(v) for k, v in obj.items()}
    return obj


def _eval_tree(obj: Any, out: List[mx.array]) -> None:
    if isinstance(obj, mx.array):
        out.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _eval_tree(o, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _eval_tree(v, out)


@dataclass
class StateFragment:
    """Captured, restorable state for one cache component at a boundary."""

    capability: Capability
    prefix_len: int
    payload: Any  # adapter-defined; for CheckpointAdapter this is the snapshot dict
    schema: str = "v1"

    def eval_targets(self) -> List[mx.array]:
        out: List[mx.array] = []
        _eval_tree(self.payload, out)
        return out


@runtime_checkable
class PrefixStateAdapter(Protocol):
    """Capture/restore/merge contract for one cache-component kind."""

    capability: Capability

    def capture(self, cache: Any, prefix_len: int) -> Optional[StateFragment]: ...

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None: ...

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]: ...

    def serialize(self, fragment: StateFragment) -> Any: ...

    def deserialize(self, tree: Any) -> StateFragment: ...


def _is_snapshotable(cache: Any) -> bool:
    """True if ``cache`` exposes a restorable snapshot contract."""
    if callable(getattr(cache, "prefix_cache_snapshot", None)):
        return True
    # duck-typed: has a ``state`` we can capture (covers non-_BaseCache caches
    # such as MiniMaxM3KVCache that expose state/meta_state/merge).
    return hasattr(cache, "state") and hasattr(cache, "meta_state")


class CheckpointAdapter:
    """Universal fallback: snapshot ``state`` + ``meta_state`` as an opaque blob.

    Correct for any cache exposing the ``prefix_cache_snapshot`` contract (all
    ``_BaseCache`` subclasses) or ``state``/``meta_state`` (e.g. custom caches).
    Not batch-mergeable by default (single-row exact reuse only).
    """

    capability = Capability.CHECKPOINT

    def capture(self, cache: Any, prefix_len: int) -> Optional[StateFragment]:
        if not _is_snapshotable(cache):
            return None
        snap = getattr(cache, "prefix_cache_snapshot", None)
        raw = (
            snap()
            if callable(snap)
            else {
                "state": cache.state,
                "meta_state": cache.meta_state,
            }
        )
        return StateFragment(self.capability, prefix_len, payload=_snapshot_tree(raw))

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None:
        payload = _snapshot_tree(fragment.payload)
        restore = getattr(fresh_cache, "prefix_cache_restore", None)
        if callable(restore):
            restore(payload)
        else:
            fresh_cache.state = payload["state"]
            fresh_cache.meta_state = payload["meta_state"]

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]:
        # A cache may still declare batch merge via the contract.
        first = caches[0] if caches else None
        merge = getattr(first, "prefix_cache_merge", None)
        return merge(caches, prefix_lens) if callable(merge) else None

    def serialize(self, fragment: StateFragment) -> Any:
        return {
            "capability": fragment.capability.value,
            "prefix_len": fragment.prefix_len,
            "schema": fragment.schema,
            "payload": fragment.payload,
        }

    def deserialize(self, tree: Any) -> StateFragment:
        return StateFragment(
            Capability(tree["capability"]),
            int(tree["prefix_len"]),
            payload=tree["payload"],
            schema=tree.get("schema", "v1"),
        )


class CompositeAdapter:
    """Recurses into ``CacheList`` / tuple caches, one sub-adapter per child."""

    capability = Capability.COMPOSITE

    def _children(self, cache: Any) -> Optional[Sequence[Any]]:
        if isinstance(cache, tuple):
            return list(cache)
        subs = getattr(cache, "caches", None)
        return list(subs) if subs is not None else None

    def capture(self, cache: Any, prefix_len: int) -> Optional[StateFragment]:
        children = self._children(cache)
        if children is None:
            return None
        frags = []
        for sub in children:
            adapter = resolve_adapter(sub)
            frag = adapter.capture(sub, prefix_len)
            if frag is None:
                return None
            frags.append(frag)
        return StateFragment(self.capability, prefix_len, payload=frags)

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None:
        children = self._children(fresh_cache)
        for sub, frag in zip(children, fragment.payload):
            resolve_adapter(sub).restore(sub, frag)

    def merge_rows(
        self, caches: Sequence[Any], prefix_lens: Sequence[int]
    ) -> Optional[Any]:
        return None

    def serialize(self, fragment: StateFragment) -> Any:
        return {
            "capability": fragment.capability.value,
            "prefix_len": fragment.prefix_len,
            "children": [
                resolve_adapter_by_capability(f.capability).serialize(f)
                for f in fragment.payload
            ],
        }

    def deserialize(self, tree: Any) -> StateFragment:
        children = [
            resolve_adapter_by_capability(Capability(c["capability"])).deserialize(c)
            for c in tree["children"]
        ]
        return StateFragment(self.capability, int(tree["prefix_len"]), payload=children)


# --- registry ------------------------------------------------------------

_CAPABILITY: Dict[type, Capability] = {}
_ADAPTERS: Dict[Capability, PrefixStateAdapter] = {
    Capability.PAGEABLE: CheckpointAdapter(),  # phase-1: checkpoint fallback
    Capability.WINDOWED: CheckpointAdapter(),  # phase-1: checkpoint fallback
    Capability.CHECKPOINT: CheckpointAdapter(),
    Capability.COMPOSITE: CompositeAdapter(),
}


def register_capability(cls: type, capability: Capability) -> None:
    _CAPABILITY[cls] = capability


def register_default_capabilities() -> None:
    """Register the in-tree cache classes' declared capabilities."""
    from .models import cache as c

    for cls in (c.KVCache, c.QuantizedKVCache, c.ChunkedKVCache):
        register_capability(cls, Capability.PAGEABLE)
    for cls in (c.RotatingKVCache,):
        register_capability(cls, Capability.WINDOWED)
    for cls in (c.ArraysCache, c.PoolingCache, c.StaticPrefixKVCache):
        register_capability(cls, Capability.CHECKPOINT)
    register_capability(c.CacheList, Capability.COMPOSITE)


def resolve_capability(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> Capability:
    """Resolve the capability of ``cache``.

    Key rule: an unregistered subclass is never assumed ``PAGEABLE`` via
    inheritance (that would misclassify ring/windowed caches subclassing
    ``KVCache``); such cases downgrade to the conservative ``CHECKPOINT``.
    """
    if not _CAPABILITY:
        register_default_capabilities()
    t = type(cache)
    if isinstance(cache, tuple):
        return Capability.COMPOSITE
    if overrides and t in overrides:
        return overrides[t]
    if t in _CAPABILITY:  # exact registration wins
        return _CAPABILITY[t]
    for base in t.__mro__[1:]:  # inherited capability (conservative only)
        if base in _CAPABILITY:
            cap = _CAPABILITY[base]
            return Capability.CHECKPOINT if cap == Capability.PAGEABLE else cap
    if _is_snapshotable(cache):
        return Capability.CHECKPOINT
    return Capability.UNSUPPORTED


def resolve_adapter(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> PrefixStateAdapter:
    return _ADAPTERS[resolve_capability(cache, overrides)]


def resolve_adapter_by_capability(cap: Capability) -> PrefixStateAdapter:
    return _ADAPTERS[cap]


# --- plan ----------------------------------------------------------------


@dataclass(frozen=True)
class ComponentPlan:
    index: int
    type_name: str
    capability: Capability
    restorable: bool
    reason: Optional[str] = None


@dataclass
class PrefixCachePlan:
    """Per-model description of how each cache entry is captured/restored."""

    components: List[ComponentPlan] = field(default_factory=list)

    @property
    def restorable(self) -> bool:
        return bool(self.components) and all(c.restorable for c in self.components)

    @property
    def capabilities(self) -> List[Capability]:
        return [c.capability for c in self.components]

    def adapter_for(self, index: int) -> PrefixStateAdapter:
        return _ADAPTERS[self.components[index].capability]

    def describe(self) -> str:
        head = (
            f"PrefixCachePlan: {len(self.components)} components, "
            f"restorable={self.restorable}"
        )
        lines = [
            f"  [{c.index}] {c.type_name}: {c.capability.value}"
            + ("" if c.restorable else f"  REJECTED ({c.reason})")
            for c in self.components
        ]
        return "\n".join([head, *lines])


def build_prefix_cache_plan(
    model: Any, overrides: Optional[Dict[type, Capability]] = None
) -> PrefixCachePlan:
    """Build a plan by resolving one adapter per entry ``model.make_cache()``."""
    lm = getattr(model, "language_model", model)
    make_cache = getattr(lm, "make_cache", None) or getattr(model, "make_cache", None)
    if not callable(make_cache):
        return PrefixCachePlan()
    caches = make_cache()
    comps: List[ComponentPlan] = []
    for i, entry in enumerate(caches):
        cap = resolve_capability(entry, overrides)
        restorable = cap != Capability.UNSUPPORTED
        comps.append(
            ComponentPlan(
                index=i,
                type_name=type(entry).__name__,
                capability=cap,
                restorable=restorable,
                reason=None if restorable else "no snapshot/restore contract",
            )
        )
    return PrefixCachePlan(components=comps)
