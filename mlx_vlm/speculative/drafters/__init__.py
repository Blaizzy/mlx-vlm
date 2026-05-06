import json
import logging
from typing import Optional, Tuple

from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp"}

# Drafter HF ``model_type`` → required round-loop kind. Anything not listed
# here falls back to ``DEFAULT_DRAFTER_KIND`` when the caller didn't pass one.
DRAFTER_KIND_BY_MODEL_TYPE = {
    "gemma4_assistant": "mtp",
}

DEFAULT_DRAFTER_KIND = "dflash"

logger = logging.getLogger(__name__)


def _peek_drafter_model_type(model_path) -> Optional[str]:
    """Read the drafter's HF ``config.json`` ``model_type`` without loading
    weights. Returns ``None`` if the config can't be read."""
    try:
        with open(model_path / "config.json") as f:
            return json.load(f).get("model_type")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def resolve_drafter_kind(model_path, kind: Optional[str] = None) -> str:
    """Reconcile the caller's ``kind`` with the drafter's actual model type.

    When ``kind`` is None, auto-detect from the drafter's HF ``model_type``;
    if the model_type is unknown, fall back to :data:`DEFAULT_DRAFTER_KIND`.

    When the caller passes a ``kind`` that disagrees with the drafter's
    ``model_type``, we override (and warn). This avoids the trap where a
    user points ``--draft-model`` at e.g. a ``gemma4_assistant`` checkpoint
    but forgets ``--draft-kind mtp``: rather than crashing deep inside
    ``draft_block`` with an opaque error, we pick the right kind for them.
    """
    model_type = _peek_drafter_model_type(model_path)
    expected = DRAFTER_KIND_BY_MODEL_TYPE.get(model_type)

    if kind is None:
        resolved = expected or DEFAULT_DRAFTER_KIND
        logger.info(
            "Auto-detected --draft-kind=%r for drafter %r (model_type=%r).",
            resolved,
            str(model_path),
            model_type,
        )
        return resolved

    if expected is not None and expected != kind:
        logger.warning(
            "Drafter %r has model_type=%r which requires --draft-kind=%r; "
            "got --draft-kind=%r. Overriding to %r.",
            str(model_path),
            model_type,
            expected,
            kind,
            expected,
        )
        return expected
    return kind


def load_drafter(
    path_or_repo: str, kind: Optional[str] = None, **kwargs
) -> Tuple[object, str]:
    """Load a speculative drafter and return ``(model, resolved_kind)``.

    ``kind`` defaults to ``None``, which triggers auto-detection from the
    drafter's HF ``model_type`` (see :func:`resolve_drafter_kind`). Callers
    should use ``resolved_kind`` for downstream dispatch instead of trusting
    their original ``kind`` arg.
    """
    if kind is not None and kind not in KNOWN_DRAFTER_KINDS:
        raise ValueError(
            f"Unknown drafter kind {kind!r}. Known: {sorted(KNOWN_DRAFTER_KINDS)}"
        )
    from ...utils import get_model_path, load_model

    path = get_model_path(path_or_repo)
    resolved = resolve_drafter_kind(path, kind)
    return load_model(path, **kwargs), resolved


__all__ = [
    "DFlashDraftModel",
    "KNOWN_DRAFTER_KINDS",
    "DRAFTER_KIND_BY_MODEL_TYPE",
    "DEFAULT_DRAFTER_KIND",
    "resolve_drafter_kind",
    "load_drafter",
]
