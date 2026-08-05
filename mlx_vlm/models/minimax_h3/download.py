from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download

from .weights import SOURCE_REPO_ID, SOURCE_REVISION

Partition = Literal["fl2va", "ref2va"]

_COMMON_PATTERNS = (
    "LICENSE",
    "README.md",
    "modular_model_index.json",
    "scheduler/*.json",
    "audio_scheduler/*.json",
    "vae/*.json",
    "vae/*.safetensors",
    "vae/*.safetensors.index.json",
    "audio_vae/*.json",
    "audio_vae/*.safetensors",
    "audio_vae/*.safetensors.index.json",
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.safetensors.index.json",
    "tokenizer/**",
    "processor/*.json",
)


@dataclass(frozen=True, slots=True)
class MiniMaxH3DownloadPlan:
    repo_id: str
    revision: str
    partition: Partition
    patterns: tuple[str, ...]


def download_plan(
    partition: Partition,
    *,
    repo_id: str = SOURCE_REPO_ID,
    revision: str = SOURCE_REVISION,
) -> MiniMaxH3DownloadPlan:
    if partition not in ("fl2va", "ref2va"):
        raise ValueError(f"partition must be 'fl2va' or 'ref2va', got {partition!r}")
    transformer = "transformer" if partition == "fl2va" else "transformer_ref"
    model_index = "FL2VA" if partition == "fl2va" else "Ref2VA"
    patterns = _COMMON_PATTERNS + (
        f"{model_index}/model_index.json",
        f"{transformer}/*.json",
        f"{transformer}/*.safetensors",
        f"{transformer}/*.safetensors.index.json",
    )
    return MiniMaxH3DownloadPlan(repo_id, revision, partition, patterns)


def download_model(
    *,
    partition: Partition,
    repo_id: str = SOURCE_REPO_ID,
    revision: str = SOURCE_REVISION,
    local_dir: str | Path | None = None,
    token: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
) -> Path:
    """Download exactly one H3 task partition plus its shared components."""
    plan = download_plan(partition, repo_id=repo_id, revision=revision)
    kwargs = {
        "repo_id": plan.repo_id,
        "revision": plan.revision,
        "allow_patterns": list(plan.patterns),
        "token": token or os.environ.get("HF_TOKEN") or None,
        "force_download": force_download,
        "max_workers": max_workers,
    }
    if local_dir is not None:
        target = Path(local_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(target)
    path = Path(snapshot_download(**kwargs))
    return validate_official_layout(path, partition=partition)


def validate_official_layout(
    model_path: str | Path,
    *,
    partition: Partition,
) -> Path:
    root = Path(model_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    transformer = "transformer" if partition == "fl2va" else "transformer_ref"
    required = (
        f"{transformer}/config.json",
        f"{transformer}/*.safetensors",
        "vae/config.json",
        "vae/*.safetensors",
        "audio_vae/config.json",
        "audio_vae/*.safetensors",
        "text_encoder/config.json",
        "text_encoder/*.safetensors",
        "tokenizer/tokenizer.json",
    )
    missing = [pattern for pattern in required if not list(root.glob(pattern))]
    if missing:
        formatted = "\n".join(f"  - {pattern}" for pattern in missing)
        raise FileNotFoundError(
            f"MiniMax-H3 {partition} snapshot is incomplete:\n{formatted}"
        )
    return root


__all__ = [
    "MiniMaxH3DownloadPlan",
    "download_model",
    "download_plan",
    "validate_official_layout",
]
