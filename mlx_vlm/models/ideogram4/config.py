from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Ideogram4Variant:
    name: str
    repo_id: str
    default_sampler_preset: str = "V4_DEFAULT_20"


@dataclass(frozen=True, slots=True)
class Ideogram4TransformerConfig:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * 13
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Ideogram4TransformerConfig":
        heads = data.get("num_heads", data.get("num_attention_heads", cls.num_heads))
        head_dim = data.get("attention_head_dim")
        emb_dim = data.get("emb_dim")
        if emb_dim is None and head_dim is not None:
            emb_dim = int(heads) * int(head_dim)
        return cls(
            emb_dim=int(emb_dim or cls.emb_dim),
            num_layers=int(data.get("num_layers", cls.num_layers)),
            num_heads=int(heads),
            intermediate_size=int(data.get("intermediate_size", cls.intermediate_size)),
            adanln_dim=int(data.get("adaln_dim", cls.adanln_dim)),
            in_channels=int(data.get("in_channels", cls.in_channels)),
            llm_features_dim=int(data.get("llm_features_dim", cls.llm_features_dim)),
            rope_theta=int(data.get("rope_theta", cls.rope_theta)),
            mrope_section=tuple(data.get("mrope_section", cls.mrope_section)),
            norm_eps=float(data.get("norm_eps", cls.norm_eps)),
        )


IDEOGRAM_4_FP8_REPO_ID = "ideogram-ai/ideogram-4-fp8"

VARIANTS: dict[str, Ideogram4Variant] = {
    IDEOGRAM_4_FP8_REPO_ID: Ideogram4Variant(
        name="ideogram-4-fp8",
        repo_id=IDEOGRAM_4_FP8_REPO_ID,
    ),
}


def get_variant(model: str | Ideogram4Variant | None = None) -> Ideogram4Variant:
    if isinstance(model, Ideogram4Variant):
        return model
    if model is None:
        return VARIANTS[IDEOGRAM_4_FP8_REPO_ID]
    key = str(model).strip().lower().rstrip("/")
    try:
        return VARIANTS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported Ideogram 4 variant: {model}") from exc


def variant_from_local_path(model_path: str | Path) -> Ideogram4Variant:
    root = Path(model_path).expanduser()
    index_path = root / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing Ideogram 4 model_index.json under {root}")
    data = json.loads(index_path.read_text())
    if data.get("_class_name") != "Ideogram4Pipeline":
        raise ValueError(f"{root} is not an Ideogram4Pipeline snapshot")
    return get_variant(IDEOGRAM_4_FP8_REPO_ID)


def validate_dimensions(width: int, height: int) -> None:
    for name, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{name} must be in [256, 2048], got {value}")
        if value % 16 != 0:
            raise ValueError(f"{name} must be divisible by 16, got {value}")
    ratio = max(width / height, height / width)
    if ratio > 6:
        raise ValueError(f"aspect ratio must be at most 6:1, got {width}x{height}")
