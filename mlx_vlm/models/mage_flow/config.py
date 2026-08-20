from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MageFlowVariant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    task: str
    default_steps: int
    default_guidance: float

    @property
    def supports_generation(self) -> bool:
        return self.task == "generate"

    @property
    def supports_edit(self) -> bool:
        return self.task == "edit"


def _variant(
    name: str,
    repo_id: str,
    *,
    task: str,
    steps: int,
    guidance: float,
    aliases: tuple[str, ...] = (),
) -> MageFlowVariant:
    return MageFlowVariant(
        name=name,
        aliases=(name, repo_id, repo_id.rsplit("/", 1)[-1], *aliases),
        repo_id=repo_id,
        task=task,
        default_steps=steps,
        default_guidance=guidance,
    )


VARIANTS: dict[str, MageFlowVariant] = {
    "mage-flow-base": _variant(
        "mage-flow-base",
        "microsoft/Mage-Flow-Base",
        task="generate",
        steps=30,
        guidance=5.0,
        aliases=("mage-flow-4b-base",),
    ),
    "mage-flow": _variant(
        "mage-flow",
        "microsoft/Mage-Flow",
        task="generate",
        steps=20,
        guidance=5.0,
        aliases=("mage-flow-4b",),
    ),
    "mage-flow-turbo": _variant(
        "mage-flow-turbo",
        "microsoft/Mage-Flow-Turbo",
        task="generate",
        steps=4,
        guidance=1.0,
        aliases=("mage-flow-4b-turbo",),
    ),
    "mage-flow-edit-base": _variant(
        "mage-flow-edit-base",
        "microsoft/Mage-Flow-Edit-Base",
        task="edit",
        steps=30,
        guidance=5.0,
        aliases=("mage-flow-edit-4b-base",),
    ),
    "mage-flow-edit": _variant(
        "mage-flow-edit",
        "microsoft/Mage-Flow-Edit",
        task="edit",
        steps=30,
        guidance=5.0,
        aliases=("mage-flow-edit-4b",),
    ),
    "mage-flow-edit-turbo": _variant(
        "mage-flow-edit-turbo",
        "microsoft/Mage-Flow-Edit-Turbo",
        task="edit",
        steps=4,
        guidance=1.0,
        aliases=("mage-flow-edit-4b-turbo",),
    ),
}

_ALIASES = {
    alias.lower(): variant for variant in VARIANTS.values() for alias in variant.aliases
}


def get_variant(name: str | MageFlowVariant = "mage-flow") -> MageFlowVariant:
    if isinstance(name, MageFlowVariant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"Unknown Mage-Flow variant {name!r}. Supported: {supported}"
        ) from exc


def variant_from_local_path(model_path: str | Path) -> MageFlowVariant:
    root = Path(model_path).expanduser()
    metadata_path = root / "mlx_mage_flow.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        variant = metadata.get("variant")
        if variant:
            return get_variant(str(variant))

    name = str(root).lower().replace("_", "-")
    is_edit = "edit" in name
    if "turbo" in name:
        return VARIANTS["mage-flow-edit-turbo" if is_edit else "mage-flow-turbo"]
    if "base" in name:
        return VARIANTS["mage-flow-edit-base" if is_edit else "mage-flow-base"]
    if is_edit:
        return VARIANTS["mage-flow-edit"]

    model_index = root / "model_index.json"
    if model_index.exists():
        metadata = json.loads(model_index.read_text())
        if metadata.get("_class_name") == "MageFlowPipeline":
            return VARIANTS["mage-flow"]

    raise ValueError(
        f"Could not infer a Mage-Flow variant from local model path: {root}. "
        "Use a recognized Hugging Face model id or a directory name containing "
        "Base, Turbo, and/or Edit."
    )


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 512 or value > 2048:
            raise ValueError(f"{label} must be in [512, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")


def list_variants() -> tuple[str, ...]:
    return tuple(VARIANTS)


__all__ = [
    "MageFlowVariant",
    "VARIANTS",
    "get_variant",
    "list_variants",
    "validate_dimensions",
    "variant_from_local_path",
]
