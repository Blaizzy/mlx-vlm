"""Supported speculative checkpoint loading and architecture validation."""

import json

KNOWN_DRAFTER_KINDS = {"mtp"}
DEFAULT_DRAFTER_KIND = "mtp"
DRAFTER_KIND_BY_MODEL_TYPE = {"glm5_next_mtp": "mtp"}


def resolve_drafter_kind(model_path, kind=None):
    config = json.loads((model_path / "config.json").read_text())
    if kind not in (None, "mtp") or config.get("model_type") != "glm5_next_mtp":
        raise ValueError(
            "Only GLM-5.3-Flash native MTP is supported. Extract its head with mlx_vlm.split_mtp."
        )
    return "mtp"


def validate_drafter_compatibility(target_model, draft_model, draft_kind):
    target = getattr(target_model, "language_model", target_model)
    target_config = target.config
    draft_config = draft_model.config
    if (
        draft_kind != "mtp"
        or draft_config.model_type != "glm5_next_mtp"
        or target_config.model_type not in ("glm5_next", "glm5_next_text")
    ):
        raise ValueError(
            "Only GLM-5.3-Flash paired with its native MTP head is supported."
        )
    for field in ("hidden_size", "vocab_size", "num_hidden_layers", "n_routed_experts"):
        if getattr(target_config, field) != getattr(draft_config.text_config, field):
            raise ValueError(f"MTP checkpoint does not match target {field}.")


def load_drafter(path_or_repo, kind=None, **kwargs):
    from ...utils import get_model_path, load_model

    path = get_model_path(path_or_repo)
    resolved = resolve_drafter_kind(path, kind)
    return load_model(path, **kwargs), resolved
