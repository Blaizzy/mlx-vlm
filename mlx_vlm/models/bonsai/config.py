from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BonsaiVariant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    local_dir_name: str
    precision: str


VARIANTS: dict[str, BonsaiVariant] = {
    "ternary": BonsaiVariant(
        name="ternary",
        aliases=(
            "bonsai",
            "bonsai-ternary",
            "ternary",
            "ternary-mlx",
            "bonsai-ternary-mlx",
            "2bit",
            "prism-ml/bonsai-image-ternary-4b-mlx-2bit",
        ),
        repo_id="prism-ml/bonsai-image-ternary-4B-mlx-2bit",
        local_dir_name="bonsai-image-4B-ternary-mlx",
        precision="2bit",
    ),
}

_ALIASES = {
    alias: variant for variant in VARIANTS.values() for alias in variant.aliases
}


def get_variant(name: str | BonsaiVariant = "ternary") -> BonsaiVariant:
    if isinstance(name, BonsaiVariant):
        return name
    key = name.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown Bonsai variant {name!r}. Supported: {supported}"
        ) from exc


def list_variants() -> tuple[str, ...]:
    return tuple(VARIANTS)


def default_models_dir() -> Path:
    return Path.cwd() / "models"


def default_model_path(
    variant: BonsaiVariant, models_dir: str | Path | None = None
) -> Path:
    root = (
        Path(models_dir).expanduser()
        if models_dir is not None
        else default_models_dir()
    )
    return root / variant.local_dir_name


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        width_s, height_s = normalized.split("x", 1)
        width = int(width_s)
        height = int(height_s)
    except ValueError as exc:
        raise ValueError(f"Size must be WIDTHxHEIGHT, got {value!r}") from exc
    validate_dimensions(width=width, height=height)
    return width, height


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")
