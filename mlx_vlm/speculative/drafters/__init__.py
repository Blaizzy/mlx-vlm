"""Supported speculative checkpoint loading and architecture validation."""

import json

KNOWN_DRAFTER_KINDS = {"mtp"}
DEFAULT_DRAFTER_KIND = "mtp"
DRAFTER_TARGETS = {
    "glm5_next_mtp": ("glm5_next", "glm5_next_text"),
    "qwen3_5_mtp": ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"),
}
DRAFTER_KIND_BY_MODEL_TYPE = {name: "mtp" for name in DRAFTER_TARGETS}


def resolve_drafter_kind(model_path, kind=None):
    config = json.loads((model_path / "config.json").read_text())
    if kind not in (None, "mtp") or config.get("model_type") not in DRAFTER_TARGETS:
        raise ValueError(
            "Supported native MTP heads are GLM-5.3-Flash and Qwen3.5. Extract with mlx_vlm.split_mtp."
        )
    return "mtp"


def validate_drafter_compatibility(target_model, draft_model, draft_kind):
    target = getattr(target_model, "language_model", target_model)
    target_config = getattr(target, "args", None) or target.config
    draft_config = draft_model.config
    if draft_kind != "mtp" or target_config.model_type not in DRAFTER_TARGETS.get(
        draft_config.model_type, ()
    ):
        raise ValueError("Pair GLM-5.3-Flash or Qwen3.5 with its own native MTP head.")
    for field in (
        "hidden_size",
        "vocab_size",
        "num_hidden_layers",
        "n_routed_experts",
        "num_experts",
    ):
        if not hasattr(target_config, field):
            continue
        if getattr(target_config, field) != getattr(draft_config.text_config, field):
            raise ValueError(f"MTP checkpoint does not match target {field}.")


def load_drafter(path_or_repo, kind=None, **kwargs):
    from ...utils import get_model_path, load_model

    path = get_model_path(path_or_repo)
    resolved = resolve_drafter_kind(path, kind)
    return load_model(path, **kwargs), resolved
