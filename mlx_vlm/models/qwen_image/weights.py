"""Load Qwen-Image-2.1 components from a diffusers checkpoint into MLX.

Weight-key remapping is small: the transformer is all ``Linear`` (no transpose)
with two renames; the VAE convolutions transpose from PyTorch ``OIHW`` to MLX
``OHWI`` and RMS-norm ``gamma`` flattens to 1-D. The text encoder is the shared
``qwen3_vl`` model, loaded through its own path.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.qwen3_vl.config import ModelConfig as Qwen3VLConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

from .config import QwenImageVariant, get_variant
from .transformer import QwenImageTransformer
from .vae import QwenImageVAE


def _read_quant(directory: Path) -> dict | None:
    config = directory / "config.json"
    if config.exists():
        quant = json.loads(config.read_text()).get("quantization")
        if isinstance(quant, dict):
            return quant
    return None


def _is_mlx_native(directory: Path) -> bool:
    """True if weights are already in MLX layout (a converted checkpoint)."""
    config = directory / "config.json"
    return config.exists() and bool(json.loads(config.read_text()).get("mlx_format"))


def _apply(
    model, weights: list[tuple[str, mx.array]], quant: dict | None, *, strict: bool
):
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


def _remap_transformer(weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
    out = []
    for key, value in weights.items():
        key = key.replace("time_text_embed.timestep_embedder.", "time_text_embed.")
        key = key.replace("modulation.1.", "modulation.0.")
        out.append((key, value))
    return out


def _remap_vae(weights: dict[str, mx.array]) -> list[tuple[str, mx.array]]:
    out = []
    for key, value in weights.items():
        if key.endswith(".gamma"):
            value = value.reshape(-1)
        elif value.ndim == 4:
            # PyTorch Conv2d OIHW -> MLX Conv2d OHWI
            value = value.transpose(0, 2, 3, 1)
        out.append((key, value))
    return out


def load_transformer(
    model_path: str | Path, variant: QwenImageVariant | str | None = None
) -> QwenImageTransformer:
    root = Path(model_path).expanduser()
    if not isinstance(variant, QwenImageVariant):
        variant = get_variant(variant if variant is not None else "qwen-image-2.1")
    model = QwenImageTransformer(**variant.transformer_overrides)
    shards = _load_shards(root / "transformer")
    if _is_mlx_native(root / "transformer"):
        weights = list(shards.items())
    else:
        weights = _remap_transformer(shards)
    return _apply(model, weights, _read_quant(root / "transformer"), strict=True)


def load_vae(model_path: str | Path) -> QwenImageVAE:
    root = Path(model_path).expanduser()
    config = json.loads((root / "vae" / "config.json").read_text())
    model = QwenImageVAE(
        base_dim=config["base_dim"],
        decoder_base_dim=config.get("decoder_base_dim") or config["base_dim"],
        z_dim=config["z_dim"],
        dim_mult=tuple(config["dim_mult"]),
        num_res_blocks=config["num_res_blocks"],
        temperal_downsample=tuple(config["temperal_downsample"]),
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        is_residual=config.get("is_residual", True),
    )
    shards = _load_shards(root / "vae")
    if _is_mlx_native(root / "vae"):
        weights = list(shards.items())
    else:
        weights = _remap_vae(shards)
    return _apply(model, weights, _read_quant(root / "vae"), strict=True)


def load_text_encoder(model_path: str | Path) -> Qwen3VLModel:
    root = Path(model_path).expanduser()
    config = json.loads((root / "text_encoder" / "config.json").read_text())
    model = Qwen3VLModel(Qwen3VLConfig.from_dict(config))
    weights = dict(_load_shards(root / "text_encoder"))
    if not _is_mlx_native(root / "text_encoder") and hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    # The vision patch convolution needs OITHW -> OTHWI, including checkpoints
    # converted before the edit path was supported (marked mlx_format but still
    # containing this one convolution in its original layout).
    if model.vision_tower is not None:
        weights = model.vision_tower.sanitize(weights)
    return _apply(
        model, list(weights.items()), _read_quant(root / "text_encoder"), strict=True
    )


__all__ = ["load_text_encoder", "load_transformer", "load_vae"]
