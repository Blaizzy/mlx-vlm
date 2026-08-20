import json
import logging
from typing import Any, Optional, Tuple

from .laguna_dflash import LagunaDFlashDraftModel
from .muse_glimmer_assistant import MuseGlimmerAssistantDraftModel
from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp", "eagle3"}

# Drafter HF ``model_type`` → required round-loop kind. Anything not listed
# here falls back to ``DEFAULT_DRAFTER_KIND`` when the caller didn't pass one.
DRAFTER_KIND_BY_MODEL_TYPE = {
    "deepseek_v4_mtp": "mtp",
    "eagle3": "eagle3",
    "gemma4_assistant": "mtp",
    "gemma4_unified_assistant": "mtp",
    "glm4_moe_lite_mtp": "mtp",
    "inkling_mtp": "mtp",
    "qwen3_5_mtp": "mtp",
    "laguna": "dflash",
    "muse_glimmer_assistant": "dflash",
}

DEFAULT_DRAFTER_KIND = "dflash"

logger = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _hidden_size(config: Any) -> Any:
    return _cfg_get(_cfg_get(config, "text_config", config), "hidden_size")


def _noop_target_compatibility(target_model: Any) -> None:
    del target_model


def _validate_model_specific_compatibility(target_model: Any, draft_model: Any) -> None:
    validator = getattr(
        draft_model,
        "validate_target_compatibility",
        _noop_target_compatibility,
    )
    validator(target_model)


def validate_drafter_compatibility(
    target_model: Any,
    draft_model: Any,
    draft_kind: Optional[str],
) -> None:
    """Validate that a loaded drafter can safely pair with a target model.

    This intentionally uses architecture/config fields instead of repository
    names, so quantized MLX conversions and local checkpoints remain accepted.
    """
    draft_cfg = getattr(draft_model, "config", None)
    if draft_cfg is None:
        return

    model_type = _cfg_get(draft_cfg, "model_type")
    expected_kind = DRAFTER_KIND_BY_MODEL_TYPE.get(model_type)
    if expected_kind is None and "mtp" in str(model_type).lower():
        expected_kind = "mtp"
    if expected_kind is not None and draft_kind != expected_kind:
        raise ValueError(
            f"Drafter model_type={model_type!r} requires draft_kind={expected_kind!r}. "
            f"Got draft_kind={draft_kind!r}."
        )

    _validate_model_specific_compatibility(target_model, draft_model)

    if draft_kind != "mtp":
        return

    draft_hidden_size = (
        _cfg_get(draft_cfg, "backbone_hidden_size")
        or _cfg_get(draft_cfg, "target_hidden_size")
        or _hidden_size(draft_cfg)
    )
    target = getattr(target_model, "language_model", target_model)
    target_hidden_size = _hidden_size(getattr(target, "config", None))

    if (
        draft_hidden_size is not None
        and target_hidden_size is not None
        and draft_hidden_size != target_hidden_size
    ):
        raise ValueError(
            "Drafter is incompatible with the target model. "
            "Use the drafter checkpoint for the same target family and size. "
            f"Drafter target hidden_size={draft_hidden_size!r}, "
            f"target hidden_size={target_hidden_size!r}."
        )


def _peek_drafter_model_type(model_path) -> Optional[str]:
    """Read the drafter's HF ``config.json`` ``model_type`` without loading
    weights. Returns ``None`` if the config can't be read."""
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
            return config.get("model_type") or config.get("speculators_model_type")
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


def quantize_drafter(model: Any, bits: int, group_size: int = 64) -> Any:
    """Quantize a loaded drafter in memory for faster draft forwards."""
    if bits not in (4, 8):
        raise ValueError(f"Drafter quantization bits must be 4 or 8; got {bits}.")
    if group_size not in (32, 64, 128):
        raise ValueError(
            "Drafter quantization group size must be 32, 64, or 128; "
            f"got {group_size}."
        )

    import mlx.core as mx
    import mlx.nn as nn

    nn.quantize(model, group_size=group_size, bits=bits)
    mx.eval(model.parameters())
    model.draft_quantization = {"bits": bits, "group_size": group_size}
    return model


def load_drafter(
    path_or_repo: str,
    kind: Optional[str] = None,
    *,
    draft_bits: Optional[int] = None,
    draft_group_size: int = 64,
    **kwargs,
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
    model = load_model(path, **kwargs)
    if draft_bits is not None:
        quantize_drafter(model, draft_bits, draft_group_size)
    return model, resolved


__all__ = [
    "DEFAULT_DRAFTER_KIND",
    "DRAFTER_KIND_BY_MODEL_TYPE",
    "KNOWN_DRAFTER_KINDS",
    "DFlashDraftModel",
    "LagunaDFlashDraftModel",
    "MuseGlimmerAssistantDraftModel",
    "quantize_drafter",
    "load_drafter",
    "resolve_drafter_kind",
    "validate_drafter_compatibility",
]
