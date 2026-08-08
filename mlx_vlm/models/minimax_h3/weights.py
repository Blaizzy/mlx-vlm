from __future__ import annotations

import gc
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen3_vl.config import ModelConfig as Qwen3VLConfig
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.utils import make_shards

from .audio_vae import MiniMaxH3AudioVAE
from .conditioner import MiniMaxH3Conditioner
from .config import (
    MiniMaxH3AudioVAEConfig,
    MiniMaxH3TransformerConfig,
    MiniMaxH3VideoVAEConfig,
)
from .constants import MINIMAX_H3_TEXT_ENCODER_LAYER
from .pipeline import MiniMaxH3Pipeline
from .transformer import MiniMaxH3Transformer
from .visual_vae import MiniMaxH3VideoVAE

Partition = Literal["fl2va", "ref2va"]

FORMAT_NAME = "mlx-vlm-minimax-h3"
FORMAT_VERSION = 1
SOURCE_REPO_ID = "MiniMaxAI/MiniMax-H3"
SOURCE_REVISION = "b3c7290e66afdf293bef3b9077b7a266ef421f34"
DIFFUSERS_REVISION = "9f169d98d0bce392a889c3b6524d0d97734dfc0e"
MLX_VIDEO_REVISION = "87db56a51758fefb748a359b90a5283bb8ba4837"

_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


class MiniMaxH3WeightError(ValueError):
    """Raised when an H3 component does not have exact tensor coverage."""


@dataclass(frozen=True, slots=True)
class MiniMaxH3ConversionReport:
    source: Path
    destination: Path
    partition: Partition
    text_only: bool
    source_bytes: int
    converted_bytes: int
    tensor_counts: dict[str, int]
    dry_run: bool


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _component_files(directory: Path) -> list[Path]:
    index_path = directory / "model.safetensors.index.json"
    if not index_path.exists():
        diffusers_indexes = sorted(directory.glob("*.safetensors.index.json"))
        if len(diffusers_indexes) > 1:
            raise MiniMaxH3WeightError(
                f"multiple safetensor indexes found under {directory}"
            )
        if diffusers_indexes:
            index_path = diffusers_indexes[0]
    if index_path.exists():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise MiniMaxH3WeightError(f"index has no weight map: {index_path}")
        names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) for name in names):
            raise MiniMaxH3WeightError(f"index has invalid shard names: {index_path}")
        files = [directory / name for name in names]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing safetensor shards under {directory}: {', '.join(missing)}"
            )
        extras = sorted(
            path.name
            for path in directory.glob("*.safetensors")
            if path not in files and not path.name.startswith("._")
        )
        if extras:
            raise MiniMaxH3WeightError(
                f"unindexed safetensor shards under {directory}: {', '.join(extras)}"
            )
        return files

    files = sorted(
        path
        for path in directory.glob("*.safetensors")
        if not path.name.startswith("._")
    )
    if not files:
        raise FileNotFoundError(f"no safetensors found under {directory}")
    return files


def load_safetensors_strict(directory: str | Path) -> dict[str, mx.array]:
    directory = Path(directory).expanduser()
    files = _component_files(directory)
    weights: dict[str, mx.array] = {}
    owners: dict[str, str] = {}
    for path in files:
        shard = mx.load(str(path))
        duplicates = sorted(set(weights).intersection(shard))
        if duplicates:
            details = ", ".join(
                f"{key} ({owners[key]}, {path.name})" for key in duplicates[:8]
            )
            raise MiniMaxH3WeightError(f"duplicate tensors: {details}")
        weights.update(shard)
        owners.update({key: path.name for key in shard})

    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if indexes:
        index = _load_json(indexes[0])
        expected_map = index["weight_map"]
        expected = set(expected_map)
        actual = set(weights)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        wrong_shard = sorted(
            key for key, owner in owners.items() if expected_map.get(key) != owner
        )
        if missing or unexpected or wrong_shard:
            raise MiniMaxH3WeightError(
                _coverage_message(
                    directory.name,
                    missing,
                    unexpected,
                    wrong_shard=wrong_shard,
                )
            )
    return weights


def _coverage_message(
    component: str,
    missing: list[str],
    unexpected: list[str],
    *,
    wrong_shape: list[str] | None = None,
    wrong_shard: list[str] | None = None,
) -> str:
    groups = []
    for label, values in (
        ("missing", missing),
        ("unexpected", unexpected),
        ("wrong shape", wrong_shape or []),
        ("wrong shard", wrong_shard or []),
    ):
        if values:
            suffix = " ..." if len(values) > 8 else ""
            groups.append(f"{label} ({len(values)}): {', '.join(values[:8])}{suffix}")
    return f"{component} tensor coverage failed; " + "; ".join(groups)


def _apply_exact_weights(
    component: str,
    model: nn.Module,
    weights: dict[str, mx.array],
) -> None:
    parameters = dict(tree_flatten(model.parameters()))
    missing = sorted(set(parameters) - set(weights))
    unexpected = sorted(set(weights) - set(parameters))
    wrong_shape = sorted(
        key
        for key in set(parameters).intersection(weights)
        if tuple(parameters[key].shape) != tuple(weights[key].shape)
    )
    if missing or unexpected or wrong_shape:
        raise MiniMaxH3WeightError(
            _coverage_message(component, missing, unexpected, wrong_shape=wrong_shape)
        )
    model.load_weights(sorted(weights.items()), strict=True)


def _manifest(root: Path) -> dict[str, Any] | None:
    path = root / "h3_manifest.json"
    if not path.exists():
        return None
    manifest = _load_json(path)
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"unsupported MiniMax-H3 format in {path}")
    if manifest.get("schema_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported MiniMax-H3 schema in {path}: "
            f"{manifest.get('schema_version')!r}"
        )
    return manifest


def _validate_partition(partition: str) -> Partition:
    if partition not in ("fl2va", "ref2va"):
        raise ValueError(f"partition must be 'fl2va' or 'ref2va', got {partition!r}")
    return partition


def load_component_configs(
    model_path: str | Path,
    *,
    partition: Partition,
) -> tuple[
    MiniMaxH3TransformerConfig,
    MiniMaxH3VideoVAEConfig,
    MiniMaxH3AudioVAEConfig,
    dict[str, Any],
]:
    root = Path(model_path).expanduser()
    partition = _validate_partition(partition)
    manifest = _manifest(root)
    if manifest is not None:
        recorded = manifest.get("partition")
        if recorded != partition:
            raise ValueError(
                f"converted directory contains {recorded!r}, not {partition!r}"
            )
        transformer_dir = root / "transformer"
        video_dir = root / "video_vae"
        conditioner_dir = root / "conditioner"
    else:
        transformer_dir = root / (
            "transformer" if partition == "fl2va" else "transformer_ref"
        )
        video_dir = root / ("vae" if (root / "vae").exists() else "video_vae")
        conditioner_dir = root / "text_encoder"
    return (
        MiniMaxH3TransformerConfig.from_dict(
            _load_json(transformer_dir / "config.json")
        ),
        MiniMaxH3VideoVAEConfig.from_dict(_load_json(video_dir / "config.json")),
        MiniMaxH3AudioVAEConfig.from_dict(
            _load_json(root / "audio_vae" / "config.json")
        ),
        _load_json(conditioner_dir / "config.json"),
    )


def load_transformer(
    model_path: str | Path,
    *,
    partition: Partition,
) -> MiniMaxH3Transformer:
    root = Path(model_path).expanduser()
    partition = _validate_partition(partition)
    manifest = _manifest(root)
    converted = manifest is not None
    if converted and manifest.get("partition") != partition:
        raise ValueError(
            f"converted directory contains {manifest.get('partition')!r}, "
            f"not {partition!r}"
        )
    directory = root / (
        "transformer" if converted or partition == "fl2va" else "transformer_ref"
    )
    config = MiniMaxH3TransformerConfig.from_dict(_load_json(directory / "config.json"))
    model = MiniMaxH3Transformer(config)
    _apply_exact_weights("transformer", model, load_safetensors_strict(directory))
    return model


def load_video_vae(model_path: str | Path) -> MiniMaxH3VideoVAE:
    root = Path(model_path).expanduser()
    converted = _manifest(root) is not None
    directory = root / (
        "video_vae"
        if converted
        else ("vae" if (root / "vae").exists() else "video_vae")
    )
    config = MiniMaxH3VideoVAEConfig.from_dict(_load_json(directory / "config.json"))
    model = MiniMaxH3VideoVAE(config)
    weights = load_safetensors_strict(directory)
    if not converted:
        weights = model.sanitize(weights)
    _apply_exact_weights("video_vae", model, weights)
    return model


def load_audio_vae(model_path: str | Path) -> MiniMaxH3AudioVAE:
    root = Path(model_path).expanduser()
    converted = _manifest(root) is not None
    directory = root / "audio_vae"
    config = MiniMaxH3AudioVAEConfig.from_dict(_load_json(directory / "config.json"))
    model = MiniMaxH3AudioVAE(config)
    weights = load_safetensors_strict(directory)
    if not converted:
        weights = model.sanitize(weights)
    _apply_exact_weights("audio_vae", model, weights)
    return model


def _trim_conditioner_config(
    config: dict[str, Any], *, text_only: bool
) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("Qwen3-VL config has no text_config")
    text_config["num_hidden_layers"] = MINIMAX_H3_TEXT_ENCODER_LAYER
    text_config["tie_word_embeddings"] = True
    text_config["use_final_norm"] = False
    config["skip_vision"] = text_only
    config["h3_conditioner_layer"] = MINIMAX_H3_TEXT_ENCODER_LAYER
    return config


def _prepare_conditioner_weights(
    weights: dict[str, mx.array], *, text_only: bool
) -> dict[str, mx.array]:
    selected: dict[str, mx.array] = {}
    for key, value in weights.items():
        if _conditioner_key_selected(key, text_only=text_only):
            selected[key] = value

    qwen = Qwen3VLModel.sanitize(None, selected)
    patch_key = "vision_tower.patch_embed.proj.weight"
    if patch_key in qwen:
        # This path is explicitly official-format, so avoid the generic
        # Qwen shape heuristic (small synthetic channels can be ambiguous).
        qwen[patch_key] = qwen[patch_key].transpose(0, 2, 3, 4, 1)
    return qwen


def _conditioner_key_selected(key: str, *, text_only: bool) -> bool:
    if key in ("lm_head.weight", "model.language_model.norm.weight"):
        return False
    layer_prefix = "model.language_model.layers."
    if key.startswith(layer_prefix):
        layer = key[len(layer_prefix) :].split(".", 1)[0]
        if not layer.isdigit():
            raise MiniMaxH3WeightError(f"invalid Qwen decoder key: {key}")
        return int(layer) < MINIMAX_H3_TEXT_ENCODER_LAYER
    if key.startswith("model.visual."):
        return not text_only
    if key == "model.language_model.embed_tokens.weight":
        return True
    raise MiniMaxH3WeightError(f"unexpected Qwen conditioner tensor: {key}")


def _load_tokenizer(root: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "loading MiniMax-H3 requires transformers for tokenizer file I/O"
        ) from exc
    for directory in (root / "tokenizer", root / "conditioner", root / "text_encoder"):
        if (directory / "tokenizer.json").exists():
            return AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
    raise FileNotFoundError(f"no tokenizer.json found under {root}")


def load_conditioner(
    model_path: str | Path,
    *,
    text_only: bool | None = None,
) -> MiniMaxH3Conditioner:
    root = Path(model_path).expanduser()
    manifest = _manifest(root)
    converted = manifest is not None
    directory = root / ("conditioner" if converted else "text_encoder")
    config_dict = _load_json(directory / "config.json")
    if converted:
        recorded_text_only = bool(manifest.get("text_only", False))
        if text_only is not None and text_only != recorded_text_only:
            raise ValueError(
                f"converted conditioner text_only={recorded_text_only}, not {text_only}"
            )
        text_only = recorded_text_only
    else:
        text_only = bool(text_only)
        config_dict = _trim_conditioner_config(config_dict, text_only=text_only)

    model = Qwen3VLModel(Qwen3VLConfig.from_dict(config_dict))
    weights = load_safetensors_strict(directory)
    if not converted:
        weights = _prepare_conditioner_weights(weights, text_only=text_only)
    _apply_exact_weights("conditioner", model, weights)
    return MiniMaxH3Conditioner(
        model,
        _load_tokenizer(root),
        has_vision=not text_only,
    )


def load_pipeline(
    model_path: str | Path,
    *,
    workflow: Literal["t2va", "fl2va", "ref2va"] | None = None,
    partition: Partition | None = None,
    text_only: bool | None = None,
    revision: str | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
) -> MiniMaxH3Pipeline:
    """Load a local H3 checkpoint or one selectively downloaded from the Hub.

    Remote repositories require an explicit workflow so loading never fetches
    both task transformers. ``t2va`` and ``fl2va`` select the shared FL2VA
    partition; ``ref2va`` selects the reference transformer.
    """
    from .download import partition_for_workflow, resolve_model_path

    if workflow is not None:
        workflow_partition = partition_for_workflow(workflow)
        if partition is not None and partition != workflow_partition:
            raise ValueError(
                f"workflow {workflow!r} uses the {workflow_partition!r} "
                f"partition, not {partition!r}"
            )
        partition = workflow_partition
    root = resolve_model_path(
        model_path,
        workflow=workflow,
        partition=partition,
        revision=revision,
        local_dir=local_dir,
        token=token,
        force_download=force_download,
        max_workers=max_workers,
    )
    manifest = _manifest(root)
    if partition is None:
        if manifest is None:
            raise ValueError(
                "workflow is required for an official-format checkpoint; choose "
                "'t2va', 'fl2va', or 'ref2va'"
            )
        partition = manifest.get("partition")
    partition = _validate_partition(partition)
    if text_only and partition == "ref2va":
        raise ValueError("Ref2VA requires the Qwen3-VL vision tower")
    return MiniMaxH3Pipeline(
        transformer=load_transformer(root, partition=partition),
        conditioner=load_conditioner(root, text_only=text_only),
        video_vae=load_video_vae(root),
        audio_vae=load_audio_vae(root),
        partition=partition,
    )


def _cast_weights(
    weights: dict[str, mx.array], dtype: str | None
) -> dict[str, mx.array]:
    if dtype is None:
        return weights
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    target = _DTYPES[dtype]
    return {
        key: value.astype(target) if mx.issubdtype(value.dtype, mx.floating) else value
        for key, value in weights.items()
    }


def _save_safetensors(directory: Path, weights: dict[str, mx.array]) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(sorted(weights.items()))
    shards = make_shards(weights)
    count = len(shards)
    pattern = (
        "model.safetensors" if count == 1 else "model-{:05d}-of-{:05d}.safetensors"
    )
    weight_map: dict[str, str] = {}
    total_size = 0
    for index, shard in enumerate(shards, start=1):
        name = pattern if count == 1 else pattern.format(index, count)
        mx.save_safetensors(
            str(directory / name),
            dict(sorted(shard.items())),
            metadata={"format": "mlx"},
        )
        for key, value in shard.items():
            weight_map[key] = name
            total_size += value.nbytes
    _write_json(
        directory / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    return total_size


def _copy_tokenizer(source: Path, destination: Path) -> None:
    candidates = (source / "tokenizer", source / "processor", source / "text_encoder")
    tokenizer_source = next(
        (path for path in candidates if (path / "tokenizer.json").exists()), None
    )
    if tokenizer_source is None:
        raise FileNotFoundError(f"no tokenizer assets found under {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        path = tokenizer_source / name
        if path.exists():
            shutil.copy2(path, destination / name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_bytes(directories: list[Path]) -> int:
    files: set[Path] = set()
    for directory in directories:
        files.update(_component_files(directory))
    return sum(path.stat().st_size for path in files)


def _safetensor_sizes(directory: Path) -> dict[str, tuple[str, int]]:
    tensors: dict[str, tuple[str, int]] = {}
    for path in _component_files(directory):
        with path.open("rb") as handle:
            length_data = handle.read(8)
            if len(length_data) != 8:
                raise MiniMaxH3WeightError(f"invalid safetensor header: {path}")
            header_length = struct.unpack("<Q", length_data)[0]
            try:
                header = json.loads(handle.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MiniMaxH3WeightError(
                    f"invalid safetensor header: {path}"
                ) from exc
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise MiniMaxH3WeightError(f"duplicate tensor in headers: {key}")
            offsets = metadata.get("data_offsets")
            dtype = metadata.get("dtype")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not isinstance(dtype, str)
            ):
                raise MiniMaxH3WeightError(
                    f"invalid tensor metadata for {key} in {path}"
                )
            tensors[key] = (dtype, int(offsets[1]) - int(offsets[0]))
    return tensors


def _converted_tensor_bytes(dtype: str, size: int, target_dtype: str | None) -> int:
    if target_dtype is None or not dtype.startswith("F") and dtype != "BF16":
        return size
    source_bytes = {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }.get(dtype)
    if source_bytes is None:
        return size
    target_bytes = 4 if target_dtype == "float32" else 2
    return size // source_bytes * target_bytes


def _dry_run_conversion_stats(
    *,
    transformer: Path,
    video: Path,
    audio: Path,
    conditioner: Path,
    text_only: bool,
    dtype: str | None,
) -> tuple[int, dict[str, int]]:
    component_metadata = {
        "transformer": _safetensor_sizes(transformer),
        "video_vae": _safetensor_sizes(video),
        "audio_vae": _safetensor_sizes(audio),
        "conditioner": _safetensor_sizes(conditioner),
    }
    selected = {
        "transformer": component_metadata["transformer"],
        "video_vae": component_metadata["video_vae"],
        "audio_vae": {
            key: value
            for key, value in component_metadata["audio_vae"].items()
            if not key.endswith("weight_g")
        },
        "conditioner": {
            key: value
            for key, value in component_metadata["conditioner"].items()
            if _conditioner_key_selected(key, text_only=text_only)
        },
    }
    for key in component_metadata["audio_vae"]:
        if key.endswith("weight_v"):
            scale_key = f"{key[:-8]}weight_g"
            if scale_key not in component_metadata["audio_vae"]:
                raise MiniMaxH3WeightError(
                    f"missing weight-normalization tensor: {scale_key}"
                )
    total = sum(
        _converted_tensor_bytes(source_dtype, size, dtype)
        for tensors in selected.values()
        for source_dtype, size in tensors.values()
    )
    return total, {component: len(tensors) for component, tensors in selected.items()}


def convert_minimax_h3(
    source: str | Path,
    destination: str | Path,
    *,
    partition: Partition,
    text_only: bool = False,
    dtype: str | None = None,
    source_revision: str | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> MiniMaxH3ConversionReport:
    """Convert one local official H3 partition into a strict MLX directory."""
    source = Path(source).expanduser()
    destination = Path(destination).expanduser()
    partition = _validate_partition(partition)
    if text_only and partition == "ref2va":
        raise ValueError("Ref2VA conversion cannot omit the vision tower")
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if dtype is not None and dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")

    transformer_source = source / (
        "transformer" if partition == "fl2va" else "transformer_ref"
    )
    video_source = source / ("vae" if (source / "vae").exists() else "video_vae")
    audio_source = source / "audio_vae"
    conditioner_source = source / "text_encoder"
    directories = [
        transformer_source,
        video_source,
        audio_source,
        conditioner_source,
    ]
    for directory in directories:
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"missing component config: {directory / 'config.json'}"
            )
    source_bytes = _source_bytes(directories)
    if dry_run:
        converted_bytes, tensor_counts = _dry_run_conversion_stats(
            transformer=transformer_source,
            video=video_source,
            audio=audio_source,
            conditioner=conditioner_source,
            text_only=text_only,
            dtype=dtype,
        )
        return MiniMaxH3ConversionReport(
            source=source,
            destination=destination,
            partition=partition,
            text_only=text_only,
            source_bytes=source_bytes,
            converted_bytes=converted_bytes,
            tensor_counts=tensor_counts,
            dry_run=True,
        )

    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"destination is not empty: {destination}; pass overwrite=True"
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    tensor_counts: dict[str, int] = {}
    converted_bytes = 0

    transformer_weights = _cast_weights(
        load_safetensors_strict(transformer_source), dtype
    )
    tensor_counts["transformer"] = len(transformer_weights)
    converted_bytes += _save_safetensors(
        destination / "transformer", transformer_weights
    )
    del transformer_weights
    gc.collect()
    mx.clear_cache()
    _write_json(
        destination / "transformer" / "config.json",
        _load_json(transformer_source / "config.json"),
    )

    video_weights = MiniMaxH3VideoVAE.sanitize(load_safetensors_strict(video_source))
    video_weights = _cast_weights(video_weights, dtype)
    tensor_counts["video_vae"] = len(video_weights)
    converted_bytes += _save_safetensors(destination / "video_vae", video_weights)
    del video_weights
    gc.collect()
    mx.clear_cache()
    _write_json(
        destination / "video_vae" / "config.json",
        _load_json(video_source / "config.json"),
    )

    audio_weights = MiniMaxH3AudioVAE.sanitize(load_safetensors_strict(audio_source))
    audio_weights = _cast_weights(audio_weights, dtype)
    tensor_counts["audio_vae"] = len(audio_weights)
    converted_bytes += _save_safetensors(destination / "audio_vae", audio_weights)
    del audio_weights
    gc.collect()
    mx.clear_cache()
    _write_json(
        destination / "audio_vae" / "config.json",
        _load_json(audio_source / "config.json"),
    )

    conditioner_config = _trim_conditioner_config(
        _load_json(conditioner_source / "config.json"), text_only=text_only
    )
    conditioner_weights = _prepare_conditioner_weights(
        load_safetensors_strict(conditioner_source), text_only=text_only
    )
    conditioner_weights = _cast_weights(conditioner_weights, dtype)
    tensor_counts["conditioner"] = len(conditioner_weights)
    converted_bytes += _save_safetensors(
        destination / "conditioner", conditioner_weights
    )
    del conditioner_weights
    gc.collect()
    mx.clear_cache()
    _write_json(destination / "conditioner" / "config.json", conditioner_config)
    _copy_tokenizer(source, destination / "tokenizer")

    for name in ("LICENSE", "LICENSE.md"):
        license_path = source / name
        if license_path.exists():
            shutil.copy2(license_path, destination / license_path.name)
            break

    tracked_files = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "h3_manifest.json"
    )
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": FORMAT_VERSION,
        "partition": partition,
        "text_only": text_only,
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "revision": source_revision or SOURCE_REVISION,
        },
        "reference_revisions": {
            "diffusers": DIFFUSERS_REVISION,
            "mlx_video": MLX_VIDEO_REVISION,
        },
        "conversion": {
            "dtype": dtype or "source",
            "fold_audio_weight_norm": True,
            "qwen_decoder_layers": MINIMAX_H3_TEXT_ENCODER_LAYER,
        },
        "tensor_counts": tensor_counts,
        "sha256": {
            str(path.relative_to(destination)): _sha256(path) for path in tracked_files
        },
        "modified_notice": (
            "Weights were converted for MLX; Qwen was trimmed after layer 50 and "
            "audio weight normalization was folded."
        ),
    }
    _write_json(destination / "h3_manifest.json", manifest)
    return MiniMaxH3ConversionReport(
        source=source,
        destination=destination,
        partition=partition,
        text_only=text_only,
        source_bytes=source_bytes,
        converted_bytes=converted_bytes,
        tensor_counts=tensor_counts,
        dry_run=False,
    )


__all__ = [
    "MiniMaxH3ConversionReport",
    "MiniMaxH3WeightError",
    "convert_minimax_h3",
    "load_audio_vae",
    "load_component_configs",
    "load_conditioner",
    "load_pipeline",
    "load_safetensors_strict",
    "load_transformer",
    "load_video_vae",
]
