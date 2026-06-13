import json
import logging
from typing import Any, Optional, Tuple

from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp", "eagle3"}

# Drafter HF ``model_type`` → required round-loop kind. Anything not listed
# here falls back to ``DEFAULT_DRAFTER_KIND`` when the caller didn't pass one.
DRAFTER_KIND_BY_MODEL_TYPE = {
    "deepseek_v4_mtp": "mtp",
    "eagle3": "eagle3",
    "gemma4_assistant": "mtp",
    "gemma4_unified_assistant": "mtp",
    "qwen3_5_mtp": "mtp",
}

DEFAULT_DRAFTER_KIND = "dflash"

logger = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _hidden_size(config: Any) -> Any:
    return _cfg_get(_cfg_get(config, "text_config", config), "hidden_size")


def _target_num_layers(target_model: Any) -> Optional[int]:
    target = getattr(target_model, "language_model", target_model)
    model = getattr(target, "model", None)
    layers = getattr(model, "layers", None)
    if layers is not None:
        return len(layers)
    config = getattr(target, "config", None)
    text_config = _cfg_get(config, "text_config", config)
    value = _cfg_get(text_config, "num_hidden_layers")
    return int(value) if value is not None else None


def _set_eagle3_default_aux_layers(target_model: Any, draft_cfg: Any) -> None:
    if _cfg_get(draft_cfg, "target_layer_ids_explicit", False):
        return

    num_layers = _target_num_layers(target_model)
    if not num_layers:
        return

    target_layer_ids = [2, num_layers // 2, max(num_layers - 3, 0)]
    capture_layer_ids = [max(int(i) - 1, 0) for i in target_layer_ids]
    try:
        draft_cfg.target_layer_ids = target_layer_ids
        draft_cfg.capture_layer_ids = capture_layer_ids
        draft_cfg.num_aux_hidden_states = len(target_layer_ids)
    except AttributeError:
        pass


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

    if draft_kind == "eagle3":
        _set_eagle3_default_aux_layers(target_model, draft_cfg)
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


def _peek_drafter_model_type(model_path) -> Optional[str]:
    """Read the drafter's HF ``config.json`` ``model_type`` without loading
    weights. Returns ``None`` if the config can't be read."""
    try:
        with open(model_path / "config.json") as f:
            config = json.load(f)
            architectures = config.get("architectures") or []
            if any("eagle3" in str(arch).lower() for arch in architectures):
                return "eagle3"
            return config.get("speculators_model_type") or config.get("model_type")
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
    "validate_drafter_compatibility",
    "resolve_drafter_kind",
    "load_drafter",
]
