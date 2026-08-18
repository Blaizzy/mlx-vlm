from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from mlx_vlm.quant_utils import quantize_model
from mlx_vlm.utils import save_weights, upload_to_hub

from .config import ZImageConfig
from .text_encoder import ZImageTextEncoder, sanitize_text_encoder_weights
from .transformer import ZImageTransformer, sanitize_transformer_weights
from .vae import ZImageVAE, sanitize_vae_weights


def is_z_image_model_path(model_path: str | Path) -> bool:
    root = Path(model_path)
    model_index = root / "model_index.json"
    if not model_index.exists():
        return False
    try:
        metadata = json.loads(model_index.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("_class_name") == "ZImagePipeline"


def _load_safetensors(directory: Path) -> dict[str, mx.array]:
    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"No safetensors found in {directory}")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    if any(key.endswith(".scales") for key in weights):
        raise ValueError(
            f"{directory} is already quantized; convert from the original "
            "BF16 Z-Image checkpoint"
        )
    return weights


def _sanitize_vae_for_conversion(
    directory: Path,
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    index_path = directory / "model.safetensors.index.json"
    metadata = (
        json.loads(index_path.read_text()).get("metadata", {})
        if index_path.exists()
        else {}
    )
    return sanitize_vae_weights(
        weights,
        source_layout=(False if metadata.get("mlx_vlm_format") == "z_image" else None),
    )


def _cast_weights(weights: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    return {
        key: value.astype(dtype) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in weights.items()
    }


def _quantize_component(
    model: nn.Module,
    config: dict[str, Any],
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> tuple[nn.Module, dict[str, Any]]:
    def predicate(path: str, module: nn.Module) -> bool:
        return (
            hasattr(module, "to_quantized")
            and module.weight.shape[-1] % group_size == 0
        )

    return quantize_model(
        model,
        config,
        group_size,
        bits,
        mode=mode,
        quant_predicate=predicate,
    )


def _save_component(
    output_path: Path,
    name: str,
    model: nn.Module,
    config: dict[str, Any],
) -> None:
    component_path = output_path / name
    component_path.mkdir(parents=True, exist_ok=True)
    save_weights(component_path, model, donate_weights=True)
    (component_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    index_path = component_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    metadata = index.setdefault("metadata", {})
    metadata["mlx_vlm_format"] = "z_image"
    if quantization := config.get("quantization"):
        metadata.update(
            {
                "quantization_level": str(quantization["bits"]),
                "quantization_mode": str(quantization["mode"]),
                "quantization_group_size": str(quantization["group_size"]),
            }
        )
    index_path.write_text(json.dumps(index, indent=4, sort_keys=True) + "\n")


def _copy_metadata(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if (
            path.suffix == ".safetensors"
            or path.name.endswith(".safetensors.index.json")
            or relative.parts[0] in {"assets", ".cache"}
        ):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def convert_z_image(
    model_path: str | Path,
    output_path: str | Path,
    *,
    quantize: bool = False,
    q_group_size: int = 32,
    q_bits: int = 8,
    q_mode: str = "mxfp8",
    dtype: str | None = None,
    quantize_vae: bool = False,
    upload_repo: str | None = None,
) -> Path:
    source = Path(model_path).expanduser()
    destination = Path(output_path).expanduser()
    if not is_z_image_model_path(source):
        raise ValueError(f"Not a Z-Image Diffusers checkpoint: {source}")
    if source.resolve() == destination.resolve():
        raise ValueError("Z-Image conversion output must differ from the source")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Z-Image conversion output is not empty: {destination}")

    config = ZImageConfig.from_model_path(source)
    precision_name = dtype or "bfloat16"
    try:
        precision = getattr(mx, precision_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported Z-Image dtype: {precision_name}") from exc

    _copy_metadata(source, destination)
    component_specs = (
        (
            "transformer",
            lambda: ZImageTransformer(config.transformer),
            sanitize_transformer_weights,
        ),
        (
            "text_encoder",
            lambda: ZImageTextEncoder(config.text_encoder),
            sanitize_text_encoder_weights,
        ),
        (
            "vae",
            lambda: ZImageVAE(config.vae),
            lambda weights: _sanitize_vae_for_conversion(source / "vae", weights),
        ),
    )

    for name, make_model, sanitizer in component_specs:
        print(f"[INFO] Converting Z-Image {name}")
        model = make_model()
        component_config_path = source / name / "config.json"
        component_config = json.loads(component_config_path.read_text())
        weights = sanitizer(_load_safetensors(source / name))
        weights = _cast_weights(weights, precision)
        model.load_weights(list(weights.items()), strict=True)

        if quantize and (name != "vae" or quantize_vae):
            model, component_config = _quantize_component(
                model,
                component_config,
                group_size=q_group_size,
                bits=q_bits,
                mode=q_mode,
            )
        _save_component(destination, name, model, component_config)

    if upload_repo is not None:
        upload_to_hub(destination, upload_repo)
    return destination


__all__ = ["convert_z_image", "is_z_image_model_path"]
