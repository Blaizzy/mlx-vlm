"""Atomic speculative checkpoints in the existing bounded APC store."""

from pathlib import Path

from ..apc import semantic_extra_hash
from .cache_state import SpeculativeCache


def _identity(model):
    path = getattr(model, "model_path", None)
    if path is None:
        # Unnamed in-memory weights must never share a persisted checkpoint.
        return id(model)
    path = Path(path).resolve()
    files = sorted([*path.glob("*.safetensors"), path / "config.json"])
    return [
        str(path),
        [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.exists()],
    ]


class SpeculativePrefixCache:
    def __init__(self, manager, target, draft):
        self.manager = manager
        self.namespace = semantic_extra_hash(
            model=target,
            media={
                "mtp_schema": 2,
                "target": _identity(target),
                "draft": _identity(draft),
                "draft_dependencies": semantic_extra_hash(model=draft),
            },
        )

    def _key(self, extra_hash):
        return semantic_extra_hash(
            media={"speculative": self.namespace, "request": extra_hash}
        )

    def lookup(self, tokens, *, extra_hash=0):
        # The key includes the pending target token because the shifted MTP
        # cache already depends on it. Replay that token on the target only.
        bundle, length = self.manager.lookup_exact_cache(
            list(tokens) + [-1],
            extra_hash=self._key(extra_hash),
            max_prefix_tokens=len(tokens),
        )
        if bundle is None:
            return None, 0
        state = SpeculativeCache.restore(bundle)
        position = int(state.position.item())
        if position + 1 != length or state.bonus.item() != tokens[position]:
            raise ValueError(
                "Speculative prefix checkpoint is not aligned with its token key."
            )
        return state, position

    def store(self, tokens, state, *, row=0, extra_hash=0):
        position = int(state.position[row].item())
        if position < 1 or position >= len(tokens):
            return False
        if state.bonus[row].item() != tokens[position]:
            raise ValueError("Speculative prefix key must include its pending token.")
        return self.manager.store_exact_cache(
            tokens[: position + 1],
            state.checkpoint(row),
            extra_hash=self._key(extra_hash),
        )
