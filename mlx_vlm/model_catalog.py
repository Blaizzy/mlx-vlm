from __future__ import annotations

from typing import Callable, Iterable

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound


REQUIRED_MODEL_FILES = {"config.json", "tokenizer_config.json"}


def probably_cached_mlx_model(repo) -> bool:
    if repo.repo_type != "model":
        return False
    if "main" not in repo.refs:
        return False

    file_names = {f.file_path.name for f in repo.refs["main"].files}
    has_weights = "model.safetensors.index.json" in file_names or any(
        file_name.endswith(".safetensors") for file_name in file_names
    )
    return REQUIRED_MODEL_FILES.issubset(file_names) and has_weights


def local_model_infos(
    cache_scan: Callable | None = None,
    *,
    sort: bool = False,
) -> list[dict]:
    cache_scan = cache_scan or scan_cache_dir
    try:
        hf_cache_info = cache_scan()
    except CacheNotFound:
        return []

    repos: Iterable = (
        repo for repo in hf_cache_info.repos if probably_cached_mlx_model(repo)
    )
    if sort:
        repos = sorted(repos, key=lambda repo: repo.repo_id.lower())

    return [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in repos
    ]
