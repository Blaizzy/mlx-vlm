from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import ErnieImageVariant, get_variant

DOWNLOAD_PATTERNS = (
    "model_index.json",
    "scheduler/*.json",
    "transformer/*.json",
    "transformer/*.safetensors",
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "tokenizer/**",
    "vae/*.json",
    "vae/*.safetensors",
    "pe/*.json",
    "pe/*.safetensors",
    "pe_tokenizer/**",
    "mlx_ernie_image.json",
)


def download_model(
    variant: str | ErnieImageVariant = "ernie-image-turbo",
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
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
        "transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "tokenizer/tokenizer.json",
        "vae/*.safetensors",
    )
    missing = [pattern for pattern in required if not list(root.glob(pattern))]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Model snapshot is missing required ERNIE-Image files:\n{formatted}"
        )
    pe_parts = (
        bool(list(root.glob("pe/*.safetensors"))),
        (root / "pe_tokenizer" / "tokenizer.json").exists(),
    )
    if any(pe_parts) and not all(pe_parts):
        raise FileNotFoundError(
            "ERNIE-Image prompt enhancement requires both pe weights and "
            "pe_tokenizer/tokenizer.json"
        )
    return root


__all__ = ["DOWNLOAD_PATTERNS", "download_model", "validate_model_layout"]
