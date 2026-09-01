from __future__ import annotations

import gc
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from ...utils import _load_safetensors
from .config import ModelConfig
from .language import LanguageModel, normalize_checkpoint_key

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_SIDECAR_SUFFIXES = {".json", ".jinja", ".model", ".py", ".tiktoken", ".txt"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def is_deepseek_v4_vision_checkpoint(path: str | Path) -> bool:
    root = Path(path)
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        return False
    config = _read_json(config_path)
    return (
        config.get("model_type") == "deepseek_v4"
        and int(config.get("vision_n_layers", 0)) > 0
    )


def _is_base_weight(key: str, num_hidden_layers: int) -> bool:
    key = normalize_checkpoint_key(key)
    if key.startswith("mtp."):
        return False
    match = _LAYER_RE.match(key)
    return match is None or int(match.group(1)) < num_hidden_layers


def _source_groups(
    weight_map: dict[str, str], num_hidden_layers: int
) -> tuple[list[list[str]], dict[str, list[str]]]:
    relevant = {
        key: shard
        for key, shard in weight_map.items()
        if _is_base_weight(key, num_hidden_layers)
    }
    by_shard: dict[str, list[str]] = {}
    layer_shards: dict[int, set[str]] = {}
    for key, shard in relevant.items():
        by_shard.setdefault(shard, []).append(key)
        match = _LAYER_RE.match(normalize_checkpoint_key(key))
        if match is not None:
            layer_shards.setdefault(int(match.group(1)), set()).add(shard)

    shards = sorted(by_shard)
    parents = {shard: shard for shard in shards}

    def find(shard: str) -> str:
        while parents[shard] != shard:
            parents[shard] = parents[parents[shard]]
            shard = parents[shard]
        return shard

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for dependencies in layer_shards.values():
        dependencies = sorted(dependencies)
        for shard in dependencies[1:]:
            union(dependencies[0], shard)

    components: dict[str, list[str]] = {}
    for shard in shards:
        components.setdefault(find(shard), []).append(shard)
    groups = sorted(components.values(), key=lambda group: min(group))
    return groups, by_shard


def _sanitize_group(weights: dict[str, mx.array], config: ModelConfig):
    language = LanguageModel.sanitize(SimpleNamespace(args=config), weights)

    def wrap(key: str) -> str:
        if key.startswith("language_model."):
            return key
        if key.startswith("model.") or key.startswith("lm_head."):
            return f"language_model.{key}"
        return key

    return {wrap(key): value for key, value in language.items()}


def _quantization_config(module_paths: set[str]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "group_size": 64,
        "bits": 8,
        "mode": "affine",
    }
    for path in sorted(module_paths):
        if ".ffn.switch_mlp." in path:
            config[path] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        else:
            config[path] = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    return config


def _copy_sidecars(source: Path, destination: Path) -> None:
    for path in source.iterdir():
        if not path.is_file() or path.suffix not in _SIDECAR_SUFFIXES:
            continue
        if path.name in {"config.json", "model.safetensors.index.json"}:
            continue
        shutil.copy2(path, destination / path.name)


def convert_deepseek_v4_vision(
    source_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Stream the mixed FP8/FP4 vision checkpoint into native MLX shards."""
    source = Path(source_path).expanduser()
    destination = Path(output_path).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("DeepSeek-V4 conversion output must differ from its source")
    if not is_deepseek_v4_vision_checkpoint(source):
        raise ValueError(f"Not a DeepSeek-V4 vision checkpoint: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"DeepSeek-V4 conversion output is not empty: {destination}")

    raw_config = _read_json(source / "config.json")
    quantization = raw_config.get("quantization_config") or {}
    if quantization.get("quant_method") != "fp8":
        raise ValueError(
            "DeepSeek-V4 vision conversion expects the mixed FP8 checkpoint"
        )
    config = ModelConfig.from_dict(raw_config)
    index = _read_json(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map") or {}
    groups, by_shard = _source_groups(weight_map, config.num_hidden_layers)
    if not groups:
        raise ValueError("DeepSeek-V4 checkpoint index contains no base-model weights")

    destination.mkdir(parents=True, exist_ok=True)
    output_map: dict[str, str] = {}
    quantized_modules: set[str] = set()
    total_size = 0
    output_count = len(groups)

    for output_index, source_shards in enumerate(groups, start=1):
        selected: dict[str, mx.array] = {}
        for shard_name in source_shards:
            shard_path = source / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing DeepSeek-V4 shard: {shard_path}")
            shard = _load_safetensors(str(shard_path))
            for key in by_shard[shard_name]:
                if key in selected:
                    raise ValueError(f"Duplicate DeepSeek-V4 tensor: {key}")
                selected[key] = shard[key]
            del shard

        converted = _sanitize_group(selected, config)
        del selected
        output_name = f"model-{output_index:05d}-of-{output_count:05d}.safetensors"
        mx.save_safetensors(
            str(destination / output_name),
            converted,
            metadata={"format": "mlx", "model_type": "deepseek_v4"},
        )
        for key, value in converted.items():
            if key in output_map:
                raise ValueError(f"Duplicate converted DeepSeek-V4 tensor: {key}")
            output_map[key] = output_name
            total_size += value.nbytes
            if key.endswith(".scales"):
                quantized_modules.add(key[: -len(".scales")])
        del converted
        gc.collect()
        mx.clear_cache()

    output_index = {
        "metadata": {
            "total_size": total_size,
            "source_total_size": (index.get("metadata") or {}).get("total_size"),
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n"
    )

    converted_config = dict(raw_config)
    mlx_quantization = _quantization_config(quantized_modules)
    converted_config["quantization"] = mlx_quantization
    converted_config["quantization_config"] = mlx_quantization
    (destination / "config.json").write_text(
        json.dumps(converted_config, indent=2, sort_keys=True) + "\n"
    )
    _copy_sidecars(source, destination)
    return destination


__all__ = ["convert_deepseek_v4_vision", "is_deepseek_v4_vision_checkpoint"]
