from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class LLaDAImageTransformerConfig:
    dim: int = 3840
    n_heads: int = 30
    n_layers: int = 30
    n_refiner_layers: int = 2
    in_channels: int = 128
    cap_feat_dim: int = 2560
    semantic_feat_dim: int = 4096
    norm_eps: float = 1e-5
    qk_norm: bool = True
    rope_theta: float = 256.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    t_scale: float = 1000.0

    def __post_init__(self):
        if self.dim % self.n_heads or sum(self.axes_dims) != self.dim // self.n_heads:
            raise ValueError("LLaDA-Image head dimension must match axes_dims")

    @classmethod
    def from_dict(cls, config: dict) -> LLaDAImageTransformerConfig:
        if config.get("all_patch_size", [1]) != [1] or config.get(
            "all_f_patch_size", [1]
        ) != [1]:
            raise ValueError("LLaDA-Image currently supports only patch size 1")
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in config.items() if key in allowed}
        if "axes_dims" in values:
            values["axes_dims"] = tuple(values["axes_dims"])
        return cls(**values)


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    if (
        read_config(root / "model_index.json").get("_class_name")
        != "LLaDAImagePipeline"
    ):
        raise ValueError(f"Not a LLaDA-Image checkpoint: {root}")
    for component in (
        "text_encoder",
        "queryformer",
        "text_projection",
        "transformer",
        "vae",
    ):
        if not (root / component / "config.json").is_file():
            raise FileNotFoundError(root / component / "config.json")
        if not any((root / component).glob("*.safetensors")):
            raise FileNotFoundError(f"No {component} weights in {root}")
    return root


def validate_dimensions(width: int, height: int) -> None:
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError(
            "LLaDA-Image width and height must be positive multiples of 16"
        )
