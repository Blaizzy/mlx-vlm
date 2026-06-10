#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import CacheNotFound

REQUIRED_FILES = {"config.json", "tokenizer_config.json"}


def _main_file_names(repo) -> set[str]:
    if repo.repo_type != "model" or "main" not in repo.refs:
        return set()
    return {file.file_path.name for file in repo.refs["main"].files}


def is_supported_model(repo) -> bool:
    file_names = _main_file_names(repo)
    has_weights = "model.safetensors.index.json" in file_names or any(
        file_name.endswith(".safetensors") for file_name in file_names
    )
    return REQUIRED_FILES.issubset(file_names) and has_weights


def supported_models(cache_dir: str | None = None) -> list[dict]:
    resolved_cache_dir = Path(cache_dir or HF_HUB_CACHE).expanduser()
    try:
        cache_info = scan_cache_dir(cache_dir=resolved_cache_dir)
    except CacheNotFound:
        return []

    models = []
    for repo in cache_info.repos:
        if not is_supported_model(repo):
            continue
        models.append(
            {
                "id": repo.repo_id,
                "repo_type": repo.repo_type,
                "last_modified": int(repo.last_modified),
                "cache_dir": str(resolved_cache_dir),
            }
        )
    return sorted(models, key=lambda model: model["id"].lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "List Hugging Face cache model repos that MLX-VLM can expose through "
            "the server /v1/models endpoint."
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
    args = parser.parse_args()

    models = supported_models(args.cache_dir)
    if args.json:
        print(json.dumps(models, indent=2))
        return

    for model in models:
        print(model["id"])
    print(f"\n{len(models)} supported model(s)")


if __name__ == "__main__":
    main()
