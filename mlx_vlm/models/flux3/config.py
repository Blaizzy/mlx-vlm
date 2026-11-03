from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Flux3Config:
    hidden: int = 3072
    heads: int = 24
    depth: int = 5
    depth_single: int = 28
    mlp_ratio: float = 3.0
    axes_dim: tuple[int, ...] = (32, 32, 32, 32)
    theta: int = 10000
    in_channels: dict[str, int] = field(
        default_factory=lambda: {"image": 128, "image_cond": 128, "text": 20480}
    )
    image_rope_scale: float = 0.5
    flow_map: bool = True
    text_layers: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32)
    pad_multiple: int = 80
    max_tokens: int = 4096
    default_steps: int = 16
    default_guidance: float = 1.0
    alpha: float = 6.93

    def __post_init__(self):
        if self.hidden % self.heads or sum(self.axes_dim) != self.hidden // self.heads:
            raise ValueError("Flux3 axes_dim must sum to the attention head dimension")
        if any(d <= 0 or d % 2 for d in self.axes_dim):
            raise ValueError("Flux3 rotary axis dimensions must be positive and even")

    @classmethod
    def from_dict(cls, metadata: dict) -> Flux3Config:
        dit = metadata.get("dit", {})
        if metadata.get("quantization"):
            raise ValueError("Flux3 currently supports the root BF16 package only")
        unsupported = set(dit.get("out_modes", ["image"])) - {"image"}
        if unsupported or dit.get("vec_in_dim") is not None:
            raise ValueError("Unsupported Flux3 output modes or vector conditioning")
        params = {k: v for k, v in dit.items() if k in cls.__dataclass_fields__}
        text = metadata.get("text_encoder", {})
        sampler = metadata.get("sampler", {})
        params.update(
            text_layers=tuple(text.get("layers", cls.text_layers)),
            pad_multiple=text.get("pad_multiple", 80),
            # The reference encoder truncates at 4096, including chat tokens.
            max_tokens=min(text.get("max_tokens", 4096), 4096),
            default_steps=sampler.get("steps", 16),
            default_guidance=(
                1.0 if sampler.get("guidance") is None else sampler["guidance"]
            ),
            alpha=sampler.get("alpha", 6.93),
        )
        return cls(**params)

    @classmethod
    def from_path(cls, path: str | Path) -> Flux3Config:
        metadata = json.loads((Path(path) / "config.json").read_text())
        if not is_flux3_config(metadata):
            raise ValueError(f"Not a Flux3 model package: {path}")
        return cls.from_dict(metadata)


def is_flux3_config(metadata: dict) -> bool:
    dit = metadata.get("dit")
    return (
        isinstance(dit, dict)
        and {"image", "image_cond", "text"} <= set(dit.get("in_channels", {}))
        and metadata.get("model", "").startswith("flux-3-image")
    )


def repo_id(model: str) -> str:
    aliases = {
        "flux3": "flux-3-image-klein",
        "flux3-klein": "flux-3-image-klein",
        "flux3-klein-base": "flux-3-image-klein-base",
    }
    name = aliases.get(model.lower(), model)
    if name in {"flux-3-image-klein", "flux-3-image-klein-base"}:
        return f"black-forest-labs/{name}"
    return name


def validate_dimensions(width: int, height: int) -> None:
    if any(v < 64 or v > 4096 or v % 16 for v in (width, height)):
        raise ValueError("Flux3 width and height must be multiples of 16 in [64, 4096]")
