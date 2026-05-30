from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Flux2Variant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    local_dir_name: str
    transformer_overrides: dict[str, int]
    text_encoder_overrides: dict[str, int]
    supports_generation: bool = True
    supports_edit: bool = False
    uses_reference_kv_cache: bool = False


FLUX2_KLEIN_4B_TRANSFORMER = {
    "num_layers": 5,
    "num_single_layers": 20,
    "num_attention_heads": 24,
    "joint_attention_dim": 7680,
}
FLUX2_KLEIN_9B_TRANSFORMER = {
    "num_layers": 8,
    "num_single_layers": 24,
    "num_attention_heads": 32,
    "joint_attention_dim": 12288,
}
FLUX2_KLEIN_4B_TEXT_ENCODER = {
    "hidden_size": 2560,
    "intermediate_size": 9728,
}
FLUX2_KLEIN_9B_TEXT_ENCODER = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
}


VARIANTS: dict[str, Flux2Variant] = {
    "flux2-klein-4b": Flux2Variant(
        name="flux2-klein-4b",
        aliases=(
            "flux2-klein-4b",
            "flux2-klein-4B",
            "flux2-klein",
            "klein-4b",
            "klein-4B",
            "black-forest-labs/FLUX.2-klein-4B",
        ),
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        local_dir_name="FLUX.2-klein-4B",
        transformer_overrides=FLUX2_KLEIN_4B_TRANSFORMER,
        text_encoder_overrides=FLUX2_KLEIN_4B_TEXT_ENCODER,
    ),
    "flux2-klein-9b": Flux2Variant(
        name="flux2-klein-9b",
        aliases=(
            "flux2-klein-9b",
            "flux2-klein-9B",
            "klein-9b",
            "klein-9B",
            "black-forest-labs/FLUX.2-klein-9B",
        ),
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        local_dir_name="FLUX.2-klein-9B",
        transformer_overrides=FLUX2_KLEIN_9B_TRANSFORMER,
        text_encoder_overrides=FLUX2_KLEIN_9B_TEXT_ENCODER,
        supports_edit=True,
    ),
    "flux2-klein-base-4b": Flux2Variant(
        name="flux2-klein-base-4b",
        aliases=(
            "flux2-klein-base-4b",
            "flux2-klein-base-4B",
            "flux2-base-4b",
            "flux2-base-4B",
            "klein-base-4b",
            "klein-base-4B",
            "black-forest-labs/FLUX.2-klein-base-4B",
        ),
        repo_id="black-forest-labs/FLUX.2-klein-base-4B",
        local_dir_name="FLUX.2-klein-base-4B",
        transformer_overrides=FLUX2_KLEIN_4B_TRANSFORMER,
        text_encoder_overrides=FLUX2_KLEIN_4B_TEXT_ENCODER,
    ),
    "flux2-klein-base-9b": Flux2Variant(
        name="flux2-klein-base-9b",
        aliases=(
            "flux2-klein-base-9b",
            "flux2-klein-base-9B",
            "flux2-base-9b",
            "flux2-base-9B",
            "klein-base-9b",
            "klein-base-9B",
            "black-forest-labs/FLUX.2-klein-base-9B",
        ),
        repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        local_dir_name="FLUX.2-klein-base-9B",
        transformer_overrides=FLUX2_KLEIN_9B_TRANSFORMER,
        text_encoder_overrides=FLUX2_KLEIN_9B_TEXT_ENCODER,
    ),
    "flux2-klein-9b-kv": Flux2Variant(
        name="flux2-klein-9b-kv",
        aliases=(
            "flux2-klein-9b-kv",
            "flux2-klein-9B-kv",
            "klein-9b-kv",
            "klein-9B-kv",
            "black-forest-labs/FLUX.2-klein-9b-kv",
        ),
        repo_id="black-forest-labs/FLUX.2-klein-9b-kv",
        local_dir_name="FLUX.2-klein-9b-kv",
        transformer_overrides=FLUX2_KLEIN_9B_TRANSFORMER,
        text_encoder_overrides=FLUX2_KLEIN_9B_TEXT_ENCODER,
        supports_edit=True,
        uses_reference_kv_cache=True,
    ),
}

_ALIASES = {
    alias.lower(): variant for variant in VARIANTS.values() for alias in variant.aliases
}


def get_variant(name: str | Flux2Variant = "flux2-klein-4b") -> Flux2Variant:
    if isinstance(name, Flux2Variant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown Flux2 variant {name!r}. Supported: {supported}"
        ) from exc


def variant_from_local_path(model_path: str | Path) -> Flux2Variant:
    root = Path(model_path).expanduser()
    name = str(root).lower()
    if "kv" in name or (root / "flux-2-klein-9b-kv.safetensors").exists():
        return VARIANTS["flux2-klein-9b-kv"]
    if "base" in name and "9b" in name:
        return VARIANTS["flux2-klein-base-9b"]
    if "base" in name and "4b" in name:
        return VARIANTS["flux2-klein-base-4b"]
    if "9b" in name:
        return VARIANTS["flux2-klein-9b"]
    if "4b" in name:
        return VARIANTS["flux2-klein-4b"]

    transformer_config = root / "transformer" / "config.json"
    if transformer_config.exists():
        config = json.loads(transformer_config.read_text())
        num_layers = config.get("num_layers")
        num_attention_heads = config.get("num_attention_heads")
        if num_layers == 8 or num_attention_heads == 32:
            return VARIANTS["flux2-klein-9b"]
        if num_layers == 5 or num_attention_heads == 24:
            return VARIANTS["flux2-klein-4b"]

    text_config = root / "text_encoder" / "config.json"
    if text_config.exists():
        config = json.loads(text_config.read_text())
        hidden_size = config.get("hidden_size")
        if hidden_size == 4096:
            return VARIANTS["flux2-klein-9b"]
        if hidden_size == 2560:
            return VARIANTS["flux2-klein-4b"]

    raise ValueError(
        f"Could not infer Flux2 variant from local model path: {root}. "
        "Use a recognized model id or a path containing 4B/9B in its name."
    )


def list_variants() -> tuple[str, ...]:
    return tuple(VARIANTS)


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")
