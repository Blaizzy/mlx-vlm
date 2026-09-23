"""Load Ming-Image-0.1-Design components from a diffusers checkpoint into MLX."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import MingImageConfig
from .transformer import MingImageTransformer, sanitize_transformer_weights


def _load_shards(directory: Path) -> dict[str, mx.array]:
    files = sorted(
        p for p in directory.glob("*.safetensors") if not p.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors under {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def _read_quant(directory: Path) -> dict | None:
    config = directory / "config.json"
    if config.exists():
        quant = json.loads(config.read_text()).get("quantization")
        if isinstance(quant, dict):
            return quant
    return None


def _apply(
    model: nn.Module,
    weights: list[tuple[str, mx.array]],
    quant: dict | None,
    *,
    strict: bool = True,
) -> nn.Module:
    if quant is not None:
        quantized = {
            key[: -len(".scales")] for key, _ in weights if key.endswith(".scales")
        }
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            class_predicate=lambda path, module: path in quantized,
        )
    model.load_weights(weights, strict=strict)
    model.eval()
    return model


def load_transformer(
    model_path: str | Path, config: MingImageConfig | None = None
) -> MingImageTransformer:
    root = Path(model_path).expanduser()
    config = config or MingImageConfig.from_model_path(root)
    model = MingImageTransformer(config.dit)
    weights = sanitize_transformer_weights(_load_shards(root / "transformer"))
    return _apply(model, list(weights.items()), _read_quant(root / "transformer"))


__all__ = ["load_transformer"]
