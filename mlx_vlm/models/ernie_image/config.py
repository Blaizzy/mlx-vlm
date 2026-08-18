from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ErnieImageVariant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    default_steps: int
    default_guidance: float
    is_turbo: bool


@dataclass(frozen=True, slots=True)
class ErnieImageTransformerConfig:
    hidden_size: int = 4096
    ffn_hidden_size: int = 12288
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 36
    num_attention_heads: int = 32
    patch_size: int = 1
    qk_layernorm: bool = True
    rope_axes_dim: tuple[int, int, int] = (32, 48, 48)
    rope_theta: float = 256.0
    text_in_dim: int = 3072
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ErnieImageTransformerConfig":
        allowed = {field.name for field in fields(cls)}
        data = {key: item for key, item in value.items() if key in allowed}
        if "rope_axes_dim" in data:
            data["rope_axes_dim"] = tuple(data["rope_axes_dim"])
        config = cls(**data)
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if sum(config.rope_axes_dim) != config.head_dim:
            raise ValueError(
                "rope_axes_dim must sum to the attention head dimension: "
                f"{config.rope_axes_dim} vs {config.head_dim}"
            )
        return config

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _variant(
    name: str,
    repo_id: str,
    *,
    steps: int,
    guidance: float,
    turbo: bool,
    aliases: tuple[str, ...] = (),
) -> ErnieImageVariant:
    return ErnieImageVariant(
        name=name,
        aliases=(name, repo_id, repo_id.rsplit("/", 1)[-1], *aliases),
        repo_id=repo_id,
        default_steps=steps,
        default_guidance=guidance,
        is_turbo=turbo,
    )


VARIANTS: dict[str, ErnieImageVariant] = {
    "ernie-image": _variant(
        "ernie-image",
        "baidu/ERNIE-Image",
        steps=50,
        guidance=4.0,
        turbo=False,
        aliases=("ernie-image-base",),
    ),
    "ernie-image-turbo": _variant(
        "ernie-image-turbo",
        "baidu/ERNIE-Image-Turbo",
        steps=8,
        guidance=1.0,
        turbo=True,
        aliases=("ernie-turbo",),
    ),
}

_ALIASES = {
    alias.lower().rstrip("/"): variant
    for variant in VARIANTS.values()
    for alias in variant.aliases
}


def get_variant(
    name: str | ErnieImageVariant = "ernie-image-turbo",
) -> ErnieImageVariant:
    if isinstance(name, ErnieImageVariant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"Unknown ERNIE-Image variant {name!r}. Supported: {supported}"
        ) from exc


def variant_from_local_path(model_path: str | Path) -> ErnieImageVariant:
    root = Path(model_path).expanduser()
    native_metadata = root / "mlx_ernie_image.json"
    if native_metadata.exists():
        metadata = json.loads(native_metadata.read_text())
        if variant := metadata.get("variant"):
            return get_variant(str(variant))

    name = root.name.lower()
    if root.parent.name == "snapshots" and root.parent.parent.name.startswith(
        "models--"
    ):
        name = root.parent.parent.name.removeprefix("models--").replace("--", "/")
    if "ernie-image" not in name and not _has_ernie_layout(root):
        raise ValueError(f"Could not infer an ERNIE-Image variant from {root}")
    return VARIANTS["ernie-image-turbo" if "turbo" in name else "ernie-image"]


def _has_ernie_layout(root: Path) -> bool:
    model_index = root / "model_index.json"
    if model_index.exists():
        metadata = json.loads(model_index.read_text())
        if metadata.get("_class_name") == "ErnieImagePipeline":
            return True
    transformer_config = root / "transformer" / "config.json"
    if transformer_config.exists():
        metadata = json.loads(transformer_config.read_text())
        return metadata.get("_class_name") == "ErnieImageTransformer2DModel"
    return False


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 16:
            raise ValueError(f"{label} must be at least 16, got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")


__all__ = [
    "ErnieImageTransformerConfig",
    "ErnieImageVariant",
    "VARIANTS",
    "get_variant",
    "validate_dimensions",
    "variant_from_local_path",
]
