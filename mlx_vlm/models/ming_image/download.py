"""Download and validate a Ming-Image-0.1-Design checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "inclusionAI/Ming-Image-0.1-Design"

DOWNLOAD_PATTERNS = (
    "mllm/*.json",
    "mllm/*.safetensors",
    "mllm/tokenizer.json",
    "connector/*.json",
    "connector/*.safetensors",
    "mlp/*.json",
    "mlp/*.safetensors",
    "transformer/*.json",
    "transformer/*.safetensors",
    "vae/*.json",
    "vae/*.safetensors",
    "scheduler/*.json",
)

_REQUIRED = (
    "mllm/config.json",
    "mllm",
    "connector/config.json",
    "connector",
    "mlp/config.json",
    "mlp",
    "transformer/config.json",
    "transformer",
    "vae/config.json",
    "vae",
    "scheduler/scheduler_config.json",
    "mllm/tokenizer.json",
)


def download_model(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> Path:
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "allow_patterns": list(DOWNLOAD_PATTERNS),
        "token": token or os.environ.get("HF_TOKEN") or None,
        "force_download": force_download,
        "max_workers": max_workers,
    }
    if local_dir is not None:
        target = Path(local_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(target)
    return validate_model_layout(Path(snapshot_download(**kwargs)))


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    for entry in _REQUIRED:
        if entry.endswith(".json"):
            if not (root / entry).exists():
                raise FileNotFoundError(f"Missing {entry} under {root}")
        elif not any((root / entry).glob("*.safetensors")):
            raise FileNotFoundError(f"No safetensors in {entry} under {root}")
    return root


__all__ = [
    "DEFAULT_REPO_ID",
    "DOWNLOAD_PATTERNS",
    "download_model",
    "validate_model_layout",
]
