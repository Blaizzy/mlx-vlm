from __future__ import annotations

import argparse
import gc
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_map_with_path

from mlx_vlm.utils import (
    MODEL_CONVERSION_DTYPES,
    get_model_path,
    save_weights,
    upload_to_hub,
)

from .config import VARIANTS, MageFlowVariant, get_variant, variant_from_local_path
from .weights import load_text_encoder, load_transformer, load_vae

_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


def is_mage_flow_checkpoint(model_path: str | Path) -> bool:
    root = Path(model_path).expanduser()
    if (root / "mlx_mage_flow.json").exists():
        return True
    model_index = _read_json(root / "model_index.json", required=False)
    return bool(model_index and model_index.get("_class_name") == "MageFlowPipeline")


def convert_mage_flow(
    model_path: str | Path,
    output_path: str | Path,
    *,
    source_id: str | None = None,
    variant: str | None = None,
    quantize: bool = False,
    q_group_size: int | None = None,
    q_bits: int | None = None,
    q_mode: str = "affine",
    dtype: str | None = None,
    quantize_vae: bool = False,
    upload_repo: str | None = None,
    revision: str | None = None,
) -> Path:
    source = Path(model_path).expanduser()
    destination = Path(output_path).expanduser()
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        raise ValueError("Mage-Flow conversion output must differ from its source")
    if destination_resolved.is_relative_to(source_resolved):
        raise ValueError("Mage-Flow conversion output cannot be inside its source")
    if not is_mage_flow_checkpoint(source):
        raise ValueError(f"Not a Mage-Flow checkpoint: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Mage-Flow conversion output is not empty: {destination}"
        )

    spec = _resolve_conversion_variant(source, source_id, variant)
    precision_name = dtype or "bfloat16"
    try:
        precision = getattr(mx, precision_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported Mage-Flow conversion dtype: {dtype}") from exc
    if precision not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(f"Unsupported Mage-Flow conversion dtype: {dtype}")
    quantization = (
        _quantization_parameters(q_mode, q_group_size, q_bits) if quantize else None
    )

    _copy_metadata(source, destination)
    component_quantization: dict[str, dict[str, int | str]] = {}
    plans: tuple[
        tuple[
            str,
            Callable[[], nn.Module],
            Callable[[str, nn.Module], bool] | None,
        ],
        ...,
    ] = (
        ("text_encoder", lambda: load_text_encoder(source).model, None),
        (
            "transformer",
            lambda: load_transformer(source),
            _transformer_quantization_predicate,
        ),
        (
            "vae",
            lambda: load_vae(source),
            (lambda path, module: True) if quantize_vae else None,
        ),
    )
    for name, loader, predicate in plans:
        print(f"[INFO] Converting Mage-Flow {name}")
        model = loader()
        existing_quantization = getattr(model, "quantization_config", None)
        if quantization is not None and predicate is not None:
            if existing_quantization is not None:
                raise ValueError(
                    f"{name} is already quantized; convert from the original "
                    "Mage-Flow checkpoint"
                )
            _cast_component(model, precision)
            _quantize_component(model, quantization, predicate)
            active_quantization = quantization
        else:
            _cast_component(model, precision)
            active_quantization = existing_quantization
        _save_component(destination / name, model, active_quantization)
        if active_quantization:
            component_quantization[name] = dict(active_quantization)
        del model
        gc.collect()
        mx.clear_cache()

    metadata = {
        "format_version": 1,
        "model_type": "mage_flow",
        "variant": spec.name,
        "source": source_id or str(source),
        "source_revision": revision,
        "tensor_layout": "mlx_nhwc",
        "component_quantization": component_quantization,
    }
    (destination / "mlx_mage_flow.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    if upload_repo is not None:
        upload_to_hub(destination, upload_repo)
    return destination


def convert(
    model: str,
    output_path: str | Path,
    *,
    revision: str | None = None,
    variant: str | None = None,
    quantize: bool = False,
    q_group_size: int | None = None,
    q_bits: int | None = None,
    q_mode: str = "affine",
    dtype: str | None = None,
    quantize_vae: bool = False,
    upload_repo: str | None = None,
) -> Path:
    model_path = get_model_path(model, revision=revision)
    return convert_mage_flow(
        model_path,
        output_path,
        source_id=None if Path(model).expanduser().exists() else model,
        variant=variant,
        quantize=quantize,
        q_group_size=q_group_size,
        q_bits=q_bits,
        q_mode=q_mode,
        dtype=dtype,
        quantize_vae=quantize_vae,
        upload_repo=upload_repo,
        revision=revision,
    )


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a Mage-Flow Diffusers checkpoint to MLX format."
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        dest="model",
        required=True,
        help="Local checkpoint path or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Hugging Face revision to download.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default=None,
        help="Mage-Flow variant for an otherwise ambiguous local checkpoint.",
    )
    parser.add_argument(
        "--mlx-path",
        dest="output_path",
        default="mlx_model",
        help="Directory for the converted MLX model.",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize compatible diffusion-transformer block layers.",
    )
    parser.add_argument("--q-group-size", type=int, default=None)
    parser.add_argument("--q-bits", type=int, default=None)
    parser.add_argument(
        "--q-mode",
        choices=tuple(_MODE_DEFAULTS),
        default="affine",
    )
    parser.add_argument(
        "--dtype",
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
        help="Floating-point dtype for unquantized weights.",
    )
    parser.add_argument(
        "--quantize-vae",
        action="store_true",
        help="Also quantize compatible VAE layers.",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="Hugging Face repository for the converted model.",
    )
    return parser


def main() -> None:
    args = configure_parser().parse_args()
    convert(**vars(args))


def _resolve_conversion_variant(
    source: Path,
    source_id: str | None,
    variant: str | None,
) -> MageFlowVariant:
    if variant is not None:
        return get_variant(variant)
    if (source / "mlx_mage_flow.json").exists():
        return variant_from_local_path(source)
    if source_id is not None:
        try:
            return get_variant(source_id.rsplit("/", 1)[-1])
        except ValueError:
            pass
    normalized_path = str(source).lower().replace("_", "-")
    if any(marker in normalized_path for marker in ("base", "turbo", "edit")):
        return variant_from_local_path(source)
    raise ValueError(
        "Could not infer the Mage-Flow variant. Pass --variant when converting "
        "an ambiguously named local checkpoint."
    )


def _quantization_parameters(
    mode: str,
    group_size: int | None,
    bits: int | None,
) -> dict[str, int | str]:
    if mode not in _MODE_DEFAULTS:
        raise ValueError(f"Unsupported Mage-Flow quantization mode: {mode}")
    default_group_size, default_bits = _MODE_DEFAULTS[mode]
    resolved_group_size = group_size or default_group_size
    resolved_bits = bits or default_bits
    if mode != "affine" and (resolved_group_size, resolved_bits) != (
        default_group_size,
        default_bits,
    ):
        raise ValueError(
            f"{mode} requires group_size={default_group_size}, bits={default_bits}"
        )
    return {
        "group_size": resolved_group_size,
        "bits": resolved_bits,
        "mode": mode,
    }


def _quantize_component(
    model: nn.Module,
    config: dict[str, int | str],
    predicate: Callable[[str, nn.Module], bool] | None = None,
) -> None:
    group_size = int(config["group_size"])

    def compatible(path: str, module: nn.Module) -> bool:
        return (
            (predicate is None or predicate(path, module))
            and hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.ndim == 2
            and module.weight.shape[-1] % group_size == 0
        )

    nn.quantize(
        model,
        group_size=group_size,
        bits=int(config["bits"]),
        mode=str(config["mode"]),
        class_predicate=compatible,
    )
    model.quantization_config = dict(config)


def _transformer_quantization_predicate(
    path: str,
    module: nn.Module,
) -> bool:
    return (
        path.startswith("transformer_blocks.")
        and ".img_mod." not in path
        and ".txt_mod." not in path
    )


def _cast_component(model: nn.Module, precision: mx.Dtype) -> None:
    def cast(path: str, value: mx.array) -> mx.array:
        return (
            value.astype(precision)
            if mx.issubdtype(value.dtype, mx.floating)
            else value
        )

    model.update(tree_map_with_path(cast, model.parameters()))


def _save_component(
    directory: Path,
    model: nn.Module,
    quantization: dict[str, int | str] | None,
) -> None:
    save_weights(directory, model, donate_weights=True)
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    metadata = index.setdefault("metadata", {})
    metadata.update(
        {
            "mlx_vlm_format": "mage_flow",
            "tensor_layout": "mlx_nhwc",
        }
    )
    if quantization:
        metadata.update(
            {
                "quantization_mode": str(quantization["mode"]),
                "quantization_group_size": str(quantization["group_size"]),
                "quantization_level": str(quantization["bits"]),
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


def _read_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


if __name__ == "__main__":
    main()


__all__ = [
    "configure_parser",
    "convert",
    "convert_mage_flow",
    "is_mage_flow_checkpoint",
]
