from __future__ import annotations

import gc
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_map_with_path

from mlx_vlm.utils import (
    create_model_card,
    make_shards,
    upload_to_hub,
)

from .config import (
    ErnieImageTransformerConfig,
    get_variant,
    variant_from_local_path,
)
from .text_encoder import ErnieImageTextConfig
from .weights import (
    load_prompt_enhancer,
    load_text_encoder,
    load_transformer,
    load_vae,
)

_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "nvfp4": (16, 4),
    "mxfp8": (32, 8),
}


def is_ernie_image_checkpoint(model_path: str | Path) -> bool:
    root = Path(model_path).expanduser()
    if (root / "mlx_ernie_image.json").exists():
        return True
    model_index = _read_json(root / "model_index.json", required=False)
    if model_index and model_index.get("_class_name") == "ErnieImagePipeline":
        return True
    config = _read_json(root / "transformer" / "config.json", required=False)
    if config and config.get("_class_name") == "ErnieImageTransformer2DModel":
        return True
    index = _read_json(
        root / "transformer" / "model.safetensors.index.json", required=False
    )
    if not index:
        return False
    keys = set(index.get("weight_map", {}))
    return {
        "final_norm.linear.weight",
        "layers.0.adaLN_sa_ln.weight",
    } <= keys and (
        "adaLN_modulation.1.weight" in keys
        or "adaln_modulation.weight" in keys
    )


def convert_ernie_image(
    model_path: str | Path,
    output_path: str | Path,
    *,
    source_id: str | None = None,
    quantize: bool = False,
    q_group_size: int | None = None,
    q_bits: int | None = None,
    q_mode: str = "affine",
    dtype: str | None = None,
    upload_repo: str | None = None,
    revision: str | None = None,
) -> Path:
    source = Path(model_path).expanduser()
    destination = Path(output_path).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("ERNIE-Image conversion output must differ from its source")
    if not is_ernie_image_checkpoint(source):
        raise ValueError(f"Not an ERNIE-Image checkpoint: {source}")
    try:
        variant = variant_from_local_path(source)
    except ValueError:
        source_name = (source_id or str(source)).lower()
        variant = get_variant(
            "ernie-image-turbo" if "turbo" in source_name else "ernie-image"
        )
    precision = getattr(mx, dtype) if dtype is not None else mx.bfloat16
    if precision not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError(f"Unsupported ERNIE-Image conversion dtype: {dtype}")
    quantization = (
        _quantization_parameters(q_mode, q_group_size, q_bits)
        if quantize
        else None
    )

    destination.mkdir(parents=True, exist_ok=True)
    component_metadata = {
        "format": "mlx",
        "model_type": "ernie_image",
        "tensor_layout": "mlx_nhwc",
        "source_tensor_layout": _source_layout(source),
    }

    components: dict[str, dict[str, Any]] = {}
    include_vae_encoder = _vae_checkpoint_has_encoder(source)
    if not include_vae_encoder:
        print(
            "[WARNING] Source VAE has no encoder weights; the converted "
            "checkpoint will support generation but not img2img."
        )
    plans: list[
        tuple[
            str,
            Callable[[], nn.Module],
            Callable[[str, nn.Module], bool] | None,
        ]
    ] = [
        (
            "text_encoder",
            lambda: load_text_encoder(source),
            lambda path, module: hasattr(module, "to_quantized"),
        ),
        (
            "transformer",
            lambda: load_transformer(source),
            lambda path, module: path.startswith("layers.")
            and hasattr(module, "to_quantized"),
        ),
        (
            "vae",
            lambda: load_vae(source, include_encoder=include_vae_encoder),
            None,
        ),
    ]
    if list((source / "pe").glob("*.safetensors")):
        plans.append(
            (
                "pe",
                lambda: load_prompt_enhancer(source).model,
                lambda path, module: hasattr(module, "to_quantized"),
            )
        )

    for name, loader, predicate in plans:
        print(f"[INFO] Converting ERNIE-Image {name}")
        model = loader()
        existing_quantization = getattr(model, "quantization_config", None)
        if quantization is not None:
            if existing_quantization is not None:
                raise ValueError(
                    f"{name} is already quantized; dequantize before re-quantizing"
                )
            if predicate is not None:
                _quantize_component(model, quantization, predicate)
        _cast_component(model, precision)
        metadata = dict(component_metadata)
        active_quantization = (
            quantization
            if predicate is not None and quantization
            else existing_quantization
        )
        if active_quantization:
            metadata["quantization"] = dict(active_quantization)
        _save_component(destination / name, model, metadata)
        components[name] = {
            "tensor_layout": "mlx_nhwc",
            "quantization": active_quantization,
        }
        if name == "vae":
            components[name]["supports_img2img"] = bool(
                getattr(model, "encoder", None) is not None
            )
        del model
        gc.collect()
        mx.clear_cache()

    _copy_sidecars(source, destination)
    _write_missing_configs(destination)
    native_metadata = {
        "format_version": 1,
        "model_type": "ernie_image",
        "variant": variant.name,
        "source": source_id or str(source),
        "source_revision": revision,
        "source_tensor_layout": _source_layout(source),
        "tensor_layout": "mlx_nhwc",
        "components": components,
        "supports_img2img": include_vae_encoder,
    }
    (destination / "mlx_ernie_image.json").write_text(
        json.dumps(native_metadata, indent=2, sort_keys=True) + "\n"
    )
    create_model_card(destination, source_id)
    if upload_repo is not None:
        upload_to_hub(destination, upload_repo)
    return destination


def _vae_checkpoint_has_encoder(source: Path) -> bool:
    index = _read_json(
        source / "vae" / "model.safetensors.index.json",
        required=False,
    )
    if index is None:
        return True
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(
            "VAE model.safetensors.index.json is missing a weight_map"
        )
    return any(key.startswith("encoder.") for key in weight_map) and any(
        key.startswith("quant_conv.") for key in weight_map
    )


def _quantization_parameters(
    mode: str, group_size: int | None, bits: int | None
) -> dict[str, int | str]:
    if mode not in _MODE_DEFAULTS:
        raise ValueError(f"Unsupported ERNIE-Image quantization mode: {mode}")
    default_group, default_bits = _MODE_DEFAULTS[mode]
    resolved_group = group_size or default_group
    resolved_bits = bits or default_bits
    if mode != "affine" and (resolved_group, resolved_bits) != (
        default_group,
        default_bits,
    ):
        raise ValueError(
            f"{mode} requires group_size={default_group}, bits={default_bits}"
        )
    return {
        "group_size": resolved_group,
        "bits": resolved_bits,
        "mode": mode,
    }


def _quantize_component(
    model: nn.Module,
    config: dict[str, int | str],
    predicate: Callable[[str, nn.Module], bool],
) -> None:
    group_size = int(config["group_size"])

    def compatible(path: str, module: nn.Module) -> bool:
        return (
            predicate(path, module)
            and hasattr(module, "weight")
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


def _cast_component(model: nn.Module, precision: mx.Dtype) -> None:
    def cast(path: str, value: mx.array) -> mx.array:  # noqa: ARG001
        return (
            value.astype(precision)
            if mx.issubdtype(value.dtype, mx.floating)
            else value
        )

    model.update(tree_map_with_path(cast, model.parameters()))


def _save_component(
    directory: Path,
    model: nn.Module,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    shards = make_shards(weights)
    shard_count = len(shards)
    file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shard_count > 1
        else "model.safetensors"
    )
    index_metadata = dict(metadata)
    index_metadata["total_size"] = sum(value.nbytes for value in weights.values())
    index = {"metadata": index_metadata, "weight_map": {}}
    safetensor_metadata = {
        "format": "mlx",
        "model_type": "ernie_image",
        "tensor_layout": "mlx_nhwc",
    }
    quantization = metadata.get("quantization")
    if isinstance(quantization, dict):
        safetensor_metadata.update(
            {
                "quantization_mode": str(quantization["mode"]),
                "quantization_group_size": str(quantization["group_size"]),
                "quantization_level": str(quantization["bits"]),
            }
        )
    for index_number, shard in enumerate(shards, start=1):
        name = file_format.format(index_number, shard_count)
        mx.save_safetensors(
            str(directory / name),
            shard,
            metadata=safetensor_metadata,
        )
        index["weight_map"].update({key: name for key in shard})
    index["weight_map"] = dict(sorted(index["weight_map"].items()))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


def _copy_sidecars(source: Path, destination: Path) -> None:
    for name in ("tokenizer", "pe_tokenizer"):
        source_dir = source / name
        if source_dir.exists():
            target = destination / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_dir, target)
    for component in ("transformer", "text_encoder", "vae", "pe", "scheduler"):
        source_config = source / component / (
            "scheduler_config.json" if component == "scheduler" else "config.json"
        )
        if source_config.exists():
            target_dir = destination / component
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_config, target_dir / source_config.name)
    model_index = source / "model_index.json"
    if model_index.exists():
        shutil.copy2(model_index, destination / "model_index.json")


def _write_missing_configs(destination: Path) -> None:
    transformer_config = destination / "transformer" / "config.json"
    if not transformer_config.exists():
        value = asdict(ErnieImageTransformerConfig())
        value["_class_name"] = "ErnieImageTransformer2DModel"
        transformer_config.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    text_config = destination / "text_encoder" / "config.json"
    if not text_config.exists():
        value = {"text_config": asdict(ErnieImageTextConfig())}
        value["_class_name"] = "Mistral3Model"
        text_config.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    vae_config = destination / "vae" / "config.json"
    if not vae_config.exists():
        vae_config.write_text(
            json.dumps(
                {
                    "_class_name": "AutoencoderKLFlux2",
                    "batch_norm_eps": 0.0001,
                    "block_out_channels": [128, 256, 512, 512],
                    "latent_channels": 32,
                    "layers_per_block": 2,
                    "mid_block_add_attention": True,
                    "norm_num_groups": 32,
                    "patch_size": [2, 2],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    scheduler_config = destination / "scheduler" / "scheduler_config.json"
    if not scheduler_config.exists():
        scheduler_config.parent.mkdir(parents=True, exist_ok=True)
        scheduler_config.write_text(
            json.dumps(
                {
                    "_class_name": "FlowMatchEulerDiscreteScheduler",
                    "num_train_timesteps": 1000,
                    "shift": 4.0,
                    "use_dynamic_shifting": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    model_index = destination / "model_index.json"
    if not model_index.exists():
        model_index.write_text(
            json.dumps(
                {
                    "_class_name": "ErnieImagePipeline",
                    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                    "text_encoder": ["transformers", "Mistral3Model"],
                    "tokenizer": ["transformers", "TokenizersBackend"],
                    "transformer": [
                        "diffusers",
                        "ErnieImageTransformer2DModel",
                    ],
                    "vae": ["diffusers", "AutoencoderKLFlux2"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _source_layout(source: Path) -> str:
    native = _read_json(source / "mlx_ernie_image.json", required=False)
    if native:
        return str(native.get("tensor_layout") or "mlx_nhwc")
    for component in ("transformer", "text_encoder", "vae"):
        index = _read_json(
            source / component / "model.safetensors.index.json",
            required=False,
        )
        metadata = index.get("metadata", {}) if index else {}
        if "mflux_version" in metadata or metadata.get("format") == "mlx":
            return "mlx_nhwc"
    return "pytorch_nchw"


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


__all__ = [
    "convert_ernie_image",
    "is_ernie_image_checkpoint",
]
