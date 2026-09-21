from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import QwenImageVariant, get_variant

DOWNLOAD_PATTERNS = (
    "model_index.json",
    "scheduler/*.json",
    "transformer/*.json",
    "transformer/*.safetensors",
    "vae/*.json",
    "vae/*.safetensors",
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "processor/*",
)


def download_model(
    variant: str | QwenImageVariant = "qwen-image-2.1",
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    max_workers: int = 8,
) -> Path:
    spec = get_variant(variant)
    kwargs = {
        "repo_id": spec.repo_id,
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
    required = (
        "transformer/config.json",
        "transformer",
        "vae/config.json",
        "vae",
        "text_encoder/config.json",
        "text_encoder",
        "processor/tokenizer.json",
    )
    for entry in required:
        if entry.endswith(".json"):
            if not (root / entry).exists():
                raise FileNotFoundError(f"Missing {entry} under {root}")
        elif not any((root / entry).glob("*.safetensors")):
            raise FileNotFoundError(f"No safetensors in {entry} under {root}")
    return root


__all__ = ["DOWNLOAD_PATTERNS", "download_model", "validate_model_layout"]
