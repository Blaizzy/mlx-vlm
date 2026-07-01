import json
import logging
from typing import Any, Optional, Tuple

from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp", "eagle3", "dspark"}

# Drafter HF ``model_type`` → required round-loop kind. Anything not listed
# here falls back to ``DEFAULT_DRAFTER_KIND`` when the caller didn't pass one.
DRAFTER_KIND_BY_MODEL_TYPE = {
    "deepseek_v4_mtp": "mtp",
    "eagle3": "eagle3",
    "gemma4_assistant": "mtp",
    "gemma4_unified_assistant": "mtp",
    "qwen3_5_mtp": "mtp",
}

# DSpark drafters declare a generic ``model_type`` (e.g. ``gemma4_text``); their
# identity is carried by the architecture tag and draft hyper-parameters instead.
DSPARK_ARCHITECTURE = "Gemma4DSparkModel"

DEFAULT_DRAFTER_KIND = "dflash"


def _config_is_dspark(config: Any) -> bool:
    """A DSpark drafter checkpoint, detected from its architecture tag (preferred) or its
    distinctive draft hyper-parameter cluster (Markov head + confidence head + mask token).
    """
    if config is None:
        return False
    architectures = _cfg_get(config, "architectures") or []
    if DSPARK_ARCHITECTURE in architectures:
        return True
    return (
        _cfg_get(config, "markov_rank") is not None
        and _cfg_get(config, "mask_token_id") is not None
        and _cfg_get(config, "block_size") is not None
        and _cfg_get(config, "target_layer_ids") is not None
        and bool(_cfg_get(config, "enable_confidence_head"))
    )


logger = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _hidden_size(config: Any) -> Any:
    return _cfg_get(_cfg_get(config, "text_config", config), "hidden_size")


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

    if draft_kind == "dspark":
        # The DSpark drafter projects the concatenated target hidden states, so its
        # hidden_size must equal the target's; it also needs the speculative hooks.
        target = getattr(target_model, "language_model", target_model)
        if not hasattr(target, "rollback_speculative_cache"):
            raise ValueError(
                f"Target {type(target).__name__} does not implement "
                "rollback_speculative_cache; DSpark needs a Gemma 4 (or compatible) target."
            )
        draft_hidden_size = _cfg_get(draft_cfg, "hidden_size")
        target_hidden_size = _hidden_size(getattr(target, "config", None))
        if (
            draft_hidden_size is not None
            and target_hidden_size is not None
            and draft_hidden_size != target_hidden_size
        ):
            raise ValueError(
                "DSpark drafter is incompatible with the target model. "
                f"Drafter hidden_size={draft_hidden_size!r}, "
                f"target hidden_size={target_hidden_size!r}."
            )
        return

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


def _peek_drafter_config(model_path) -> Optional[dict]:
    """Read the drafter's HF ``config.json`` without loading weights."""
    try:
        with open(model_path / "config.json") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _peek_drafter_model_type(model_path) -> Optional[str]:
    """Read the drafter's HF ``config.json`` ``model_type`` without loading
    weights. Returns ``None`` if the config can't be read."""
    config = _peek_drafter_config(model_path)
    if config is None:
        return None
    return config.get("model_type") or config.get("speculators_model_type")


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
    config = _peek_drafter_config(model_path)
    model_type = (
        (config.get("model_type") or config.get("speculators_model_type"))
        if config
        else None
    )
    # DSpark drafters use a generic model_type; detect them by architecture/markers.
    if _config_is_dspark(config):
        if kind is not None and kind != "dspark":
            logger.warning(
                "Drafter %r is a DSpark checkpoint which requires --draft-kind='dspark'; "
                "got --draft-kind=%r. Overriding to 'dspark'.",
                str(model_path),
                kind,
            )
        else:
            logger.info(
                "Auto-detected --draft-kind='dspark' for drafter %r (model_type=%r).",
                str(model_path),
                model_type,
            )
        return "dspark"

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
    "validate_drafter_compatibility",
    "resolve_drafter_kind",
    "load_drafter",
]
