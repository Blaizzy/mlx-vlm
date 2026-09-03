#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound

# Mirrors the opt-in server /v1/models cache filter used by
# --model-discovery hf-cache (mlx_vlm/server/app.py: models_endpoint).
REQUIRED_FILES = {"config.json", "tokenizer_config.json"}


def _main_ref_files(repo) -> dict[str, Path]:
    """Map filename -> cached path for the repo's `main` revision (empty if not a model repo)."""
    if repo.repo_type != "model" or "main" not in repo.refs:
        return {}
    return {file.file_path.name: file.file_path for file in repo.refs["main"].files}


def is_supported_model(files: dict[str, Path]) -> bool:
    has_weights = "model.safetensors.index.json" in files or any(
        name.endswith(".safetensors") for name in files
    )
    return REQUIRED_FILES.issubset(files) and has_weights


def _mlx_vlm_model_types() -> set[str] | None:
    """Model-type folders shipped in mlx_vlm/models, located WITHOUT importing mlx_vlm
    (find_spec does not execute the package, so this stays lightweight — no mlx import).
    """
    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None or not spec.submodule_search_locations:
        return None
    models_dir = Path(next(iter(spec.submodule_search_locations))) / "models"
    if not models_dir.is_dir():
        return None
    return {
        p.name
        for p in models_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    }


def _config_model_type(files: dict[str, Path]) -> str | None:
    cfg = files.get("config.json")
    if cfg is None:
        return None
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return None
    model_type = data.get("model_type") or data.get("speculators_model_type")
    return model_type.lower() if isinstance(model_type, str) else None


def supported_models(
    cache_dir: str | None = None, check_arch: bool = False
) -> list[dict]:
    resolved_cache_dir = Path(cache_dir or HF_HUB_CACHE).expanduser()
    try:
        cache_info = scan_cache_dir(cache_dir=resolved_cache_dir)
    except CacheNotFound:
        return []

    arch_types = _mlx_vlm_model_types() if check_arch else None

    models = []
    for repo in cache_info.repos:
        files = _main_ref_files(repo)
        if not is_supported_model(files):
            continue
        entry = {
            "id": repo.repo_id,
            "repo_type": repo.repo_type,
            "last_modified": int(repo.last_modified),
            "cache_dir": str(resolved_cache_dir),
        }
        if check_arch:
            model_type = _config_model_type(files)
            # Folder-name match; does not resolve MODEL_REMAPPING aliases, so a False
            # here is "probably not loadable" rather than definitive.
            entry["model_type"] = model_type
            if arch_types is None or model_type not in arch_types:
                continue
        models.append(entry)
    return sorted(models, key=lambda model: model["id"].lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List Hugging Face cache model repos that MLX-VLM can expose through "
            "the server's opt-in hf-cache discovery mode."
        )
    )
    parser.add_argument(
        "--cache-dir",
        help="Hugging Face cache directory. Defaults to huggingface_hub's cache.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of one model id per line.",
    )
    parser.add_argument(
        "--check-arch",
        action="store_true",
        help=(
            "Also require an mlx_vlm arch for the model_type (folder in mlx_vlm/models). "
            "Narrows the list from 'cache candidate' to 'probably loadable by mlx-vlm'."
        ),
    )
    args = parser.parse_args()

    models = supported_models(args.cache_dir, check_arch=args.check_arch)
    if args.json:
        print(json.dumps(models, indent=2))
        return

    for model in models:
        print(model["id"])
    label = "loadable" if args.check_arch else "supported"
    print(f"\n{len(models)} {label} model(s)")


if __name__ == "__main__":
    main()
