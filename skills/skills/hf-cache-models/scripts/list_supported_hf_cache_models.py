#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound


def _discovery_helpers():
    # Load only the metadata helper, without importing mlx_vlm or model code.
    package = importlib.util.find_spec("mlx_vlm")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Install mlx-vlm to use its model discovery helpers.")
    path = (
        Path(next(iter(package.submodule_search_locations)))
        / "server"
        / "model_discovery.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mlx_vlm_server_model_discovery", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def supported_models(
    cache_dir: str | None = None, check_arch: bool = False, model_dirs=()
) -> list[dict]:
    resolved_cache_dir = Path(cache_dir or HF_HUB_CACHE).expanduser()
    try:
        cache_info = scan_cache_dir(cache_dir=resolved_cache_dir)
    except CacheNotFound:
        cache_info = SimpleNamespace(repos=[])

    helpers = _discovery_helpers()
    arch_types = _mlx_vlm_model_types() if check_arch else None
    models = []
    for candidate in helpers.discover_models(cache_info, model_dirs):
        entry = {
            "id": candidate["id"],
            "repo_type": "model",
            "last_modified": candidate["created"],
            "cache_dir": str(resolved_cache_dir),
            "path": str(candidate["path"]),
        }
        if check_arch:
            config = helpers.read_json_object(candidate["path"] / "config.json")
            raw_type = config.get("model_type") or config.get("speculators_model_type")
            model_type = raw_type.lower() if isinstance(raw_type, str) else None
            # This optional folder-name check is a hint, not loader validation.
            entry["model_type"] = model_type
            if arch_types is None or model_type not in arch_types:
                continue
        models.append(entry)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List Hugging Face cache model repos that MLX-VLM can expose through "
            "the server's model discovery endpoint."
        )
    )
    parser.add_argument(
        "--cache-dir",
        help="Hugging Face cache directory. Defaults to huggingface_hub's cache.",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        help="Additional model directory or parent folder; repeat for multiple paths.",
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

    models = supported_models(
        args.cache_dir, check_arch=args.check_arch, model_dirs=args.model_dir
    )
    if args.json:
        print(json.dumps(models, indent=2))
        return

    for model in models:
        print(model["id"])
    label = "loadable" if args.check_arch else "supported"
    print(f"\n{len(models)} {label} model(s)")


if __name__ == "__main__":
    main()
