"""Cache-commit helpers for speculative decoding.

After a verification forward on ``L+1`` candidate tokens, only the first
``accept_len + 1`` entries should remain committed. The reference z-lab
DFlash implementation handles this via ``DynamicCache.crop(start)``, which
HF's ``LinearAttentionAndFullAttentionLayer.crop`` implements as:

    * full-attention layers — drop K/V entries past ``start``
    * linear-attention layers — *no-op*; the recurrent state is left as-is,
      i.e. it includes the contribution of rejected drafted tokens

This is counter-intuitive but deliberate: the ``Qwen3_5GatedDeltaNet``
recurrent state cannot be rolled back without replaying, and experiments
have shown that the drafter is robust enough that the "polluted" linear
state doesn't meaningfully hurt subsequent verification.

This module provides ``trim_fa_cache`` which mimics HF's crop semantics by
trimming only the ``KVCache`` layers (FA) and leaving ``ArraysCache`` layers
(``Qwen3_5GatedDeltaNet``) alone.

``snapshot_caches`` / ``restore_caches`` are retained for potential future
use (e.g. on pure-FA targets where a true rollback is desirable) but the
DFlash loop no longer calls them.
"""

from dataclasses import dataclass
from typing import Any, List

from mlx_lm.models.cache import ArraysCache, KVCache


def trim_fa_cache(caches: List[Any], n: int) -> None:
    """Trim the last ``n`` positions from every full-attention layer in the
    cache list. Linear-attention layers (``ArraysCache``) are intentionally
    left untouched to match the reference implementation's crop semantics.
    """
    if n <= 0:
        return
    for c in caches:
        if isinstance(c, KVCache):
            c.trim(n)
        elif isinstance(c, ArraysCache) or c is None:
            continue
        else:
            raise NotImplementedError(
                f"trim_fa_cache does not know how to handle {type(c).__name__}"
            )


# ---------------------------------------------------------------------------
# Snapshot/restore (unused by the current loop; kept for pure-FA targets)
# ---------------------------------------------------------------------------


@dataclass
class _CacheFrame:
    kind: str
    state: Any
    offset: int = 0


def snapshot_caches(caches: List[Any]) -> List[_CacheFrame]:
    frames: List[_CacheFrame] = []
    for c in caches:
        if isinstance(c, KVCache):
            frames.append(_CacheFrame(kind="kv", state=(c.keys, c.values), offset=c.offset))
        elif isinstance(c, ArraysCache):
            frames.append(_CacheFrame(kind="arrays", state=list(c.cache)))
        elif c is None:
            frames.append(_CacheFrame(kind="none", state=None))
        else:
            raise NotImplementedError(f"snapshot not implemented for {type(c).__name__}")
    return frames


def restore_caches(caches: List[Any], frames: List[_CacheFrame]) -> None:
    for c, f in zip(caches, frames):
        if f.kind == "kv":
            keys, values = f.state
            c.keys = keys
            c.values = values
            c.offset = f.offset
        elif f.kind == "arrays":
            c.cache = list(f.state)
        elif f.kind == "none":
            continue
        else:
            raise NotImplementedError(f.kind)
