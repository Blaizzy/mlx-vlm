"""Pluggable cache-component adapters for Automatic Prefix Caching (issue #1629)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

ADAPTER_SCHEMA_VERSION = 3


class Capability(str, Enum):
    """Reuse capability of a single cache component."""

    PAGEABLE = "pageable"
    WINDOWED = "windowed"
    CHECKPOINT = "checkpoint"
    COMPOSITE = "composite"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class CacheSpec:
    """Logical cache format for one model cache entry.

    This intentionally mirrors vLLM's ``KVCacheSpec`` boundary without
    importing allocator concerns that do not apply to MLX's contiguous runtime
    caches.  APC only needs to know whether an entry can be stored as pageable
    K/V, must be restored as a window/state checkpoint, or is a composite of
    other entries.
    """

    capability: Capability
    type_name: str
    block_eligible: bool = False
    window_size: Optional[int] = None
    children: tuple["CacheSpec", ...] = ()

    @property
    def pageable(self) -> bool:
        return self.capability == Capability.PAGEABLE or (
            self.capability == Capability.COMPOSITE
            and bool(self.children)
            and all(child.pageable for child in self.children)
        )

    @property
    def restorable(self) -> bool:
        return self.capability != Capability.UNSUPPORTED and all(
            child.restorable for child in self.children
        )

    @property
    def group_key(self) -> tuple:
        """Stable grouping key for entries sharing one storage policy."""
        return (
            self.capability.value,
            self.type_name,
            self.block_eligible,
            self.window_size,
            tuple(child.group_key for child in self.children),
        )


@dataclass(frozen=True)
class CacheGroupSpec:
    """Homogeneous cache entries coordinated as one APC cache group."""

    group_id: int
    spec: CacheSpec
    layer_indices: tuple[int, ...]


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
    """Detached state captured from one cache component."""

    payload: Any


def _is_snapshotable(cache: Any) -> bool:
    """True if ``cache`` exposes a restorable snapshot contract."""
    if callable(getattr(cache, "prefix_cache_snapshot", None)):
        return True

    return hasattr(cache, "state") and hasattr(cache, "meta_state")


def _has_explicit_snapshot_contract(cache: Any) -> bool:
    """True when a custom type explicitly opts into prefix snapshots."""
    from .models.cache import _BaseCache

    base_snapshot = _BaseCache.__dict__.get("prefix_cache_snapshot")
    for klass in type(cache).__mro__:
        if "prefix_cache_snapshot" in klass.__dict__:
            return klass.__dict__["prefix_cache_snapshot"] is not base_snapshot
    return False


class CheckpointAdapter:
    """Universal fallback: snapshot ``state`` + ``meta_state`` as an opaque blob."""

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
        return StateFragment(payload=_snapshot_tree(raw))

    def restore(self, fresh_cache: Any, fragment: StateFragment) -> None:
        # ``capture`` already detached every array. Re-copying here builds a
        # second lazy graph, doubles memory traffic, and can leak unevaluated
        # arrays into the asynchronous disk writer.
        payload = fragment.payload
        restore = getattr(fresh_cache, "prefix_cache_restore", None)
        if callable(restore):
            restore(payload)
        else:
            fresh_cache.state = payload["state"]
            fresh_cache.meta_state = payload["meta_state"]


_CAPABILITY: Dict[type, Capability] = {}
_DEFAULTS_REGISTERED = False


def register_capability(cls: type, capability: Capability) -> None:
    """Register the cache policy for an in-tree or third-party cache type.

    Models with a custom cache should prefer implementing
    ``prefix_cache_snapshot`` / ``prefix_cache_restore``.  Registration is
    useful when a cache is semantically windowed or pageable and that cannot be
    inferred from its state contract alone.
    """
    _CAPABILITY[cls] = capability


def register_default_capabilities() -> None:
    """Register the in-tree cache classes' declared capabilities."""
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    from .models import cache as c

    for cls in (
        c.KVCache,
        c.QuantizedKVCache,
        c.BatchKVCache,
        c.BatchQuantizedKVCache,
        c.SimpleKVCache,
    ):
        register_capability(cls, Capability.PAGEABLE)
    for cls in (c.RotatingKVCache, c.BatchRotatingKVCache, c.ChunkedKVCache):
        register_capability(cls, Capability.WINDOWED)
    for cls in (
        c.ArraysCache,
        c.PoolingCache,
        c.BatchPoolingCache,
        c.StaticPrefixKVCache,
    ):
        register_capability(cls, Capability.CHECKPOINT)
    register_capability(c.CacheList, Capability.COMPOSITE)
    try:
        from .turboquant import BatchTurboQuantKVCache, TurboQuantKVCache

        register_capability(TurboQuantKVCache, Capability.PAGEABLE)
        register_capability(BatchTurboQuantKVCache, Capability.PAGEABLE)
    except ImportError:
        # TurboQuant is optional in stripped-down installations.
        pass
    _DEFAULTS_REGISTERED = True


def cache_spec(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> CacheSpec:
    """Describe one cache entry without model-name checks.

    Composite entries are recursively described so caches such as MLA + index
    state or sparse-attention cache lists remain extensible.
    """
    capability = resolve_capability(cache, overrides)
    children: tuple[CacheSpec, ...] = ()
    if capability == Capability.COMPOSITE:
        if isinstance(cache, tuple):
            raw_children = cache
        else:
            raw_children = tuple(getattr(cache, "caches", ()))
        children = tuple(cache_spec(child, overrides) for child in raw_children)
        if not children or not all(child.restorable for child in children):
            capability = Capability.UNSUPPORTED

    window_size = None
    if capability == Capability.WINDOWED:
        for attr in ("max_size", "window_size", "chunk_size"):
            value = getattr(cache, attr, None)
            if value is not None:
                try:
                    window_size = int(value)
                except (TypeError, ValueError):
                    pass
                break
    return CacheSpec(
        capability=capability,
        type_name=type(cache).__name__,
        block_eligible=apc_block_eligible(cache),
        window_size=window_size,
        children=children,
    )


def resolve_capability(
    cache: Any, overrides: Optional[Dict[type, Capability]] = None
) -> Capability:
    """Resolve the capability of ``cache``."""
    register_default_capabilities()
    t = type(cache)
    if isinstance(cache, tuple):
        return Capability.COMPOSITE
    if overrides and t in overrides:
        return overrides[t]
    if t in _CAPABILITY:
        return _CAPABILITY[t]
    for base in t.__mro__[1:]:
        if base in _CAPABILITY:
            cap = _CAPABILITY[base]
            if cap == Capability.PAGEABLE:
                if any(
                    getattr(cache, attr, None) is not None
                    for attr in ("max_size", "window_size", "chunk_size")
                ):
                    return Capability.WINDOWED
                return Capability.CHECKPOINT
            return cap
    if _has_explicit_snapshot_contract(cache) or _custom_state_contract(cache):
        return Capability.CHECKPOINT
    return Capability.UNSUPPORTED


_APC_EXACT_TYPES: Optional[tuple] = None
_APC_BLOCK_TYPES: Optional[set] = None


def _apc_type_tables():
    global _APC_EXACT_TYPES, _APC_BLOCK_TYPES
    if _APC_EXACT_TYPES is None:
        from .models import cache as c

        _APC_EXACT_TYPES = (
            c.KVCache,
            c.BatchKVCache,
            c.BatchRotatingKVCache,
            c.BatchQuantizedKVCache,
            c.RotatingKVCache,
            c.ChunkedKVCache,
            c.ArraysCache,
        )
        _APC_BLOCK_TYPES = {c.KVCache}
    return _APC_EXACT_TYPES, _APC_BLOCK_TYPES


def apc_block_eligible(cache: Any) -> bool:
    """True if ``cache`` supports block-level (pageable KV) APC reuse."""
    if hasattr(cache, "dequantize_for_apc"):
        return True
    _, block_types = _apc_type_tables()
    return type(cache) in block_types


def apc_exact_eligible(cache: Any) -> bool:
    """True if ``cache`` supports exact whole-prefix snapshot APC reuse."""
    from .models import cache as c

    exact_types, _ = _apc_type_tables()
    if isinstance(cache, exact_types) or hasattr(cache, "dequantize_for_apc"):
        return True
    if isinstance(cache, c.CacheList):
        return all(apc_exact_eligible(s) for s in cache.caches)
    if isinstance(cache, tuple):
        return all(apc_exact_eligible(s) for s in cache)
    return _has_explicit_snapshot_contract(cache) or _custom_state_contract(cache)


def apc_mode(caches: Sequence[Any]) -> Optional[str]:
    """APC strategy for a prompt cache: ``"block"``, ``"exact"`` or ``None``."""
    return build_prefix_cache_plan_from_caches(caches).legacy_mode


def _apc_array_helpers():
    from .apc import _copy_mlx_array, _pad_kv_for_capacity

    return _copy_mlx_array, _pad_kv_for_capacity


class KVCacheCloneAdapter:
    capability = Capability.PAGEABLE

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, pad = _apc_array_helpers()
        out = type(c)()
        off = int(getattr(c, "offset", 0) or 0)
        if c.keys is not None and c.values is not None and off > 0:
            keys = copy(c.keys[..., :off, :])
            values = copy(c.values[..., :off, :])
            step = int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
            keys, values = pad(
                keys,
                values,
                offset=off,
                min_capacity_tokens=min_capacity_tokens,
                step=step,
            )
            out.keys, out.values, out.offset = keys, values, off
            eval_targets.extend([keys, values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchKVCache.merge(caches)


class RotatingKVCacheCloneAdapter:
    capability = Capability.WINDOWED

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, _ = _apc_array_helpers()
        out = type(c)(max_size=int(c.max_size), keep=int(getattr(c, "keep", 0)))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out._idx = int(getattr(c, "_idx", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys, out.values = copy(c.keys), copy(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchRotatingKVCache.merge(caches)


class ChunkedKVCacheCloneAdapter:
    capability = Capability.WINDOWED

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, _ = _apc_array_helpers()
        out = type(c)(chunk_size=int(c.chunk_size))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out.start_position = int(getattr(c, "start_position", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys, out.values = copy(c.keys), copy(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        return lm.BatchKVCache.merge(caches)


class ArraysCacheCloneAdapter:
    capability = Capability.CHECKPOINT

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        from .models import cache as lm

        copy, _ = _apc_array_helpers()
        out = lm.ArraysCache(len(c.cache))
        out.cache = []
        for state in c.cache:
            if state is None:
                out.cache.append(None)
                continue
            cp = copy(state)
            out.cache.append(cp)
            eval_targets.append(cp)
        if c.left_padding is not None:
            out.left_padding = copy(c.left_padding)
            eval_targets.append(out.left_padding)
        if c.lengths is not None:
            out.lengths = copy(c.lengths)
            eval_targets.append(out.lengths)
        return out

    def merge_rows(self, caches, prefix_lens):
        from .models import cache as lm

        size = len(caches[0].cache)
        out = lm.ArraysCache(size)
        merged: List[Optional[mx.array]] = []
        for i in range(size):
            states = [c.cache[i] for c in caches]
            sample = next((s for s in states if s is not None), None)
            if sample is None:
                merged.append(None)
                continue
            rows = [
                (
                    mx.zeros((1,) + sample.shape[1:], dtype=sample.dtype)
                    if s is None
                    else s[:1]
                )
                for s in states
            ]
            merged.append(mx.concatenate(rows, axis=0))
        out.cache = merged
        return out


class PoolingCacheCloneAdapter:
    capability = Capability.CHECKPOINT

    def clone(self, c, *, min_capacity_tokens, eval_targets):
        copy, _ = _apc_array_helpers()
        out = type(c)(int(c.ratio))
        out.remainder = int(c.remainder)
        for name in ("buf_kv", "buf_gate", "pooled"):
            value = getattr(c, name, None)
            if value is not None:
                value = copy(value)
                eval_targets.append(value)
            setattr(out, name, value)
        return out

    def merge_rows(self, caches, prefix_lens):
        return type(caches[0]).merge(caches)


_CLONE_RULES: Optional[list] = None


def _clone_rules():
    global _CLONE_RULES
    if _CLONE_RULES is None:
        from .models import cache as lm

        _CLONE_RULES = [
            (lm.KVCache, KVCacheCloneAdapter()),
            (lm.RotatingKVCache, RotatingKVCacheCloneAdapter()),
            (lm.ChunkedKVCache, ChunkedKVCacheCloneAdapter()),
            (lm.ArraysCache, ArraysCacheCloneAdapter()),
            (lm.PoolingCache, PoolingCacheCloneAdapter()),
        ]
    return _CLONE_RULES


def _custom_state_contract(c) -> bool:
    """True if ``c`` defines its own ``state`` property, not the trivial base one."""
    from .models.cache import _BaseCache

    base_state = _BaseCache.__dict__.get("state")
    for klass in type(c).__mro__:
        if "state" in klass.__dict__:
            return klass.__dict__["state"] is not base_state
    return False


def _state_clone(c, eval_targets):
    """Clone via the state/meta_state contract (from_state), detaching arrays."""
    detached = _snapshot_tree(c.state)
    _eval_tree(detached, eval_targets)
    from_state = getattr(type(c), "from_state", None)
    if callable(from_state):
        return from_state(detached, c.meta_state)
    out = type(c).__new__(type(c))
    out.state = detached
    out.meta_state = c.meta_state
    return out


def _snapshot_contract_clone(c, eval_targets):
    """Clone a cache that explicitly implements the snapshot protocol."""
    adapter = CheckpointAdapter()
    fragment = adapter.capture(c, prefix_len=int(getattr(c, "offset", 0) or 0))
    if fragment is None:
        return None
    _eval_tree(fragment.payload, eval_targets)
    try:
        out = type(c)()
    except TypeError:
        out = type(c).__new__(type(c))
    adapter.restore(out, fragment)
    return out


def clone_cache_entry(c, *, min_capacity_tokens, eval_targets):
    from .models import cache as lm

    if callable(getattr(c, "extract", None)) and callable(
        getattr(c, "is_single_row", None)
    ):
        if not c.is_single_row():
            return None
        if c.empty():
            if isinstance(c, lm.BatchRotatingKVCache):
                return lm.RotatingKVCache(max_size=int(c.max_size))
            return lm.KVCache()
        return clone_cache_entry(
            c.extract(0),
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=eval_targets,
        )
    for typ, adapter in _clone_rules():

        matched = type(c) is typ if typ is lm.KVCache else isinstance(c, typ)
        if matched:
            return adapter.clone(
                c, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
    if isinstance(c, lm.CacheList):
        subs = [
            clone_cache_entry(
                s, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
            for s in c.caches
        ]
        return None if any(s is None for s in subs) else lm.CacheList(*subs)
    if isinstance(c, tuple):
        subs = [
            clone_cache_entry(
                s, min_capacity_tokens=min_capacity_tokens, eval_targets=eval_targets
            )
            for s in c
        ]
        return None if any(s is None for s in subs) else tuple(subs)
    if hasattr(c, "dequantize_for_apc"):
        copy, _ = _apc_array_helpers()
        dk, dv = c.dequantize_for_apc()
        if dk is None or dv is None:
            return lm.KVCache()
        out = lm.KVCache()
        out.keys, out.values, out.offset = copy(dk), copy(dv), dk.shape[-2]
        eval_targets.extend([out.keys, out.values])
        return out
    if _has_explicit_snapshot_contract(c):
        return _snapshot_contract_clone(c, eval_targets)
    if _custom_state_contract(c):
        return _state_clone(c, eval_targets)
    return None


def merge_cache_entries(entries, prefix_lens):
    from .models import cache as lm

    if not entries:
        return None
    first = entries[0]
    for typ, adapter in _clone_rules():
        if typ is lm.KVCache:
            ok = all(type(c) is typ for c in entries)
        else:
            ok = all(isinstance(c, typ) for c in entries)
        if ok:
            return adapter.merge_rows(entries, prefix_lens)
    if all(isinstance(c, lm.CacheList) for c in entries):
        merged = [
            merge_cache_entries([e.caches[i] for e in entries], prefix_lens)
            for i in range(len(first.caches))
        ]
        return None if any(m is None for m in merged) else lm.CacheList(*merged)
    if all(isinstance(c, tuple) for c in entries):
        merged = [
            merge_cache_entries([e[i] for e in entries], prefix_lens)
            for i in range(len(first))
        ]
        return None if any(m is None for m in merged) else lm.CacheList(*merged)

    if "merge" in type(first).__dict__ and all(type(c) is type(first) for c in entries):
        return type(first).merge(entries, prefix_lens)
    return None


@dataclass(frozen=True)
class ComponentPlan:
    index: int
    type_name: str
    capability: Capability
    restorable: bool
    group_id: int = -1
    reason: Optional[str] = None


@dataclass
class PrefixCachePlan:
    """Per-model description of how cache entries are captured/restored.

    Layers with the same :class:`CacheSpec` are placed in one homogeneous
    group.  The coordinator intersects reuse at a single token boundary across
    all groups, which is the same core rule used by vLLM's hybrid KV cache
    manager.
    """

    components: List[ComponentPlan] = field(default_factory=list)
    layer_specs: List[CacheSpec] = field(default_factory=list)
    groups: List[CacheGroupSpec] = field(default_factory=list)

    @property
    def restorable(self) -> bool:
        return bool(self.components) and all(c.restorable for c in self.components)

    @property
    def capabilities(self) -> List[Capability]:
        return [c.capability for c in self.components]

    @property
    def is_hybrid(self) -> bool:
        return len(self.groups) > 1 or any(
            not spec.pageable for spec in self.layer_specs
        )

    @property
    def strategy(self) -> Optional[str]:
        """Physical APC strategy used by today's contiguous MLX caches."""
        if not self.restorable:
            return None
        # The block pool currently pages native KVCache leaves. Composite,
        # windowed, recurrent and custom caches are stored as checkpoints.
        return (
            "block"
            if all(spec.block_eligible for spec in self.layer_specs)
            else "checkpoint"
        )

    @property
    def legacy_mode(self) -> Optional[str]:
        """Compatibility spelling for the old block/exact API."""
        strategy = self.strategy
        return "exact" if strategy == "checkpoint" else strategy

    def describe(self) -> str:
        head = (
            f"PrefixCachePlan: {len(self.components)} components, "
            f"groups={len(self.groups)}, strategy={self.strategy}, "
            f"restorable={self.restorable}"
        )
        lines = [
            f"  [{c.index}] {c.type_name}: {c.capability.value} group={c.group_id}"
            + ("" if c.restorable else f"  REJECTED ({c.reason})")
            for c in self.components
        ]
        return "\n".join([head, *lines])


def build_prefix_cache_plan_from_caches(
    caches: Sequence[Any], overrides: Optional[Dict[type, Capability]] = None
) -> PrefixCachePlan:
    """Build a vLLM-style grouped plan from a concrete cache layout."""
    specs = [cache_spec(entry, overrides) for entry in caches]
    grouped: Dict[tuple, List[int]] = {}
    for index, spec in enumerate(specs):
        grouped.setdefault(spec.group_key, []).append(index)

    groups: List[CacheGroupSpec] = []
    group_for_layer: Dict[int, int] = {}
    for group_id, indices in enumerate(grouped.values()):
        groups.append(
            CacheGroupSpec(
                group_id=group_id,
                spec=specs[indices[0]],
                layer_indices=tuple(indices),
            )
        )
        for index in indices:
            group_for_layer[index] = group_id

    components = [
        ComponentPlan(
            index=index,
            type_name=spec.type_name,
            capability=spec.capability,
            restorable=spec.restorable,
            group_id=group_for_layer.get(index, -1),
            reason=None if spec.restorable else "no snapshot/restore contract",
        )
        for index, spec in enumerate(specs)
    ]
    return PrefixCachePlan(
        components=components,
        layer_specs=specs,
        groups=groups,
    )


def build_prefix_cache_plan(
    model: Any, overrides: Optional[Dict[type, Capability]] = None
) -> PrefixCachePlan:
    """Build a plan by resolving one adapter per entry ``model.make_cache()``."""
    lm = getattr(model, "language_model", model)
    make_cache = getattr(lm, "make_cache", None) or getattr(model, "make_cache", None)
    try:
        if callable(make_cache):
            caches = make_cache()
        else:
            # Match generation's default for dense language models which only
            # expose ``layers`` and intentionally omit a custom cache factory.
            # This is common among VLM wrappers around a conventional LM.
            from .models.cache import make_prompt_cache

            caches = make_prompt_cache(lm)
    except Exception:
        return PrefixCachePlan()
    return build_prefix_cache_plan_from_caches(caches, overrides)
