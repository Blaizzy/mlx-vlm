from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class MOSSImageTransformerConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    def __post_init__(self):
        if (
            self.num_heads < 1
            or self.dim % self.num_heads
            or (self.dim // self.num_heads) % 8
            or self.freq_dim < 2
            or self.freq_dim % 2
        ):
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, "
                "and positive even freq_dim"
            )

    @classmethod
    def from_dict(cls, values):
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})


def detect_moss_image_layout(path: str | Path) -> bool:
    root = Path(path).expanduser()
    try:
        metadata = json.loads((root / "model_index.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("_class_name") == "MOSSImagePipeline"
    )
