import glob
import os
from pathlib import Path
from typing import Optional

from filelock import FileLock
from huggingface_hub import HfApi, snapshot_download


def _repo_sha_and_files(repo_id: str, revision: str | None = None):
    api = HfApi()
    info = api.repo_info(repo_id, revision=revision, repo_type="model")
    files = api.list_repo_files(repo_id, revision=info.sha, repo_type="model")
    return info.sha, files


def _find_mlpackages(files: list[str]) -> list[str]:
    return sorted(
        {
            p.split("/")[0]
            for p in files
            if p.endswith(".mlpackage") or ".mlpackage/" in p
        }
    )


def _stage_dir(repo_id: str, sha: str, root: str | None = None) -> Path:
    root = root or os.path.expanduser("~/.cache/mlx-vlm/materialized")
    return Path(root) / f"{repo_id.replace('/', '__')}-{sha}"


def _fetch_repo_path(repo_or_path: str, force_download: bool = False) -> Path:
    p = Path(repo_or_path)
    if p.exists():
        return p
    sha, files = _repo_sha_and_files(repo_or_path)
    mlps = _find_mlpackages(files)
    if mlps:
        stage = _stage_dir(repo_or_path, sha)
        lock = FileLock(str(stage) + ".lock")
        with lock:
            if not stage.exists():
                patterns = [f"{m}/**" for m in mlps]
                snapshot_download(
                    repo_or_path,
                    revision=sha,
                    allow_patterns=patterns,
                    local_dir=stage,
                    force_download=force_download,
                )
        return stage
    return Path(snapshot_download(repo_or_path, revision=sha))


def resolve_coreml_mlpackage(
    model_path: Path, path_or_hf_repo: Optional[str], force_download: bool = False
) -> Optional[str]:
    """
    Resolve the Core ML .mlpackage path to load. This is required for Core ML models since model manifests are
    incompatible with the HF snapshot cache.

    Logic:
    - If a local .mlpackage exists in model_path (and exactly one), use fetch_repo_path(path_or_hf_repo)
      to locate and return the corresponding .mlpackage path from the repo cache.
      This avoids loading from HF snapshot cache paths that are invalid for Core ML.
    - If no local .mlpackage is present, return None.

    Returns:
        Optional[str]: The resolved .mlpackage path from the repo cache, or None if none should be loaded.

    Raises:
        ValueError: If multiple .mlpackage files are found locally or in the repo cache, or if
                    a local .mlpackage is found but path_or_hf_repo is not provided.
        FileNotFoundError: If a local .mlpackage is detected but no .mlpackage exists in the
                           resolved repo cache path.
    """
    local_candidates = glob.glob(str(model_path / "*.mlpackage"))
    if len(local_candidates) == 0:
        return None
    if len(local_candidates) > 1:
        raise ValueError("Found multiple vision model packages, aborting.")

    if not path_or_hf_repo:
        raise ValueError(
            "Found a .mlpackage locally, but path_or_hf_repo is required to resolve the correct Core ML package path."
        )

    repo_path = _fetch_repo_path(path_or_hf_repo, force_download)
    repo_candidates = glob.glob(str(repo_path / "*.mlpackage"))
    if len(repo_candidates) == 0:
        raise FileNotFoundError(
            f"No Core ML .mlpackage found in resolved repo path: {repo_path}"
        )
    if len(repo_candidates) > 1:
        raise ValueError("Found multiple vision model packages, aborting.")
    return repo_candidates[0]
