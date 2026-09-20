from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QwenImageVariant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    local_dir_name: str
    transformer_overrides: dict[str, int]
    text_encoder_overrides: dict[str, int]
    supports_generation: bool = True
    supports_edit: bool = False
    uses_reference_kv_cache: bool = False


QWEN_IMAGE_2_1_TRANSFORMER = {
    "num_layers": 32,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "in_channels": 64,
    "out_channels": 64,
    "context_in_dim": 4096,
    "mlp_ratio": 3,
}
QWEN_IMAGE_2_1_TEXT_ENCODER = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}


VARIANTS: dict[str, QwenImageVariant] = {
    "qwen-image-2.1": QwenImageVariant(
        name="qwen-image-2.1",
        aliases=(
            "qwen-image-2.1",
            "qwen-image-2_1",
            "qwen_image_2_1",
            "qwen-image",
            "qwen_image",
            "Qwen/Qwen-Image-2.1",
        ),
        repo_id="Qwen/Qwen-Image-2.1",
        local_dir_name="Qwen-Image-2.1",
        transformer_overrides=QWEN_IMAGE_2_1_TRANSFORMER,
        text_encoder_overrides=QWEN_IMAGE_2_1_TEXT_ENCODER,
        supports_edit=True,
    ),
}

_ALIASES = {
    alias.lower(): variant for variant in VARIANTS.values() for alias in variant.aliases
}


def get_variant(name: str | QwenImageVariant = "qwen-image-2.1") -> QwenImageVariant:
    if isinstance(name, QwenImageVariant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown Qwen-Image variant {name!r}. Supported: {supported}"
        ) from exc


def variant_from_local_path(model_path: str | Path) -> QwenImageVariant:
    root = Path(model_path).expanduser()
    name = root.name.lower()
    if root.parent.name == "snapshots" and root.parent.parent.name.startswith(
        "models--"
    ):
        name = root.parent.parent.name.removeprefix("models--").replace("--", "/")
    if "qwen" in name and "image" in name:
        return VARIANTS["qwen-image-2.1"]

    transformer_config = root / "transformer" / "config.json"
    if transformer_config.exists():
        config = json.loads(transformer_config.read_text())
        if "QwenImage" in str(config.get("_class_name", "")):
            return VARIANTS["qwen-image-2.1"]
        if config.get("num_layers") == 32 and config.get("context_in_dim") == 4096:
            return VARIANTS["qwen-image-2.1"]

    raise ValueError(
        f"Could not infer a Qwen-Image variant from local model path: {root}. "
        "Use a recognized model id or a snapshot of Qwen/Qwen-Image-2.1."
    )


def list_variants() -> tuple[str, ...]:
    return tuple(VARIANTS)


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")
