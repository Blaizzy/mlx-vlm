from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from mlx_vlm.models.bonsai.config import BonsaiVariant, default_model_path, get_variant

REQUIRED_FILES = (
    "transformer-packed-mflux/diffusion_pytorch_model.safetensors",
    "transformer-packed-mflux/quantization_config.json",
    "text_encoder-mlx-4bit/model.safetensors",
    "tokenizer/tokenizer.json",
)


def download_model(
    variant: str | BonsaiVariant = "ternary",
    *,
    models_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | None = None,
    max_workers: int = 16,
) -> Path:
    spec = get_variant(variant)
    if local_dir is None and models_dir is None:
        target = Path(
            snapshot_download(
                repo_id=spec.repo_id,
                token=token or os.environ.get("BONSAI_TOKEN") or None,
                max_workers=max_workers,
            )
        )
    else:
        target = (
            Path(local_dir).expanduser()
            if local_dir is not None
            else default_model_path(spec, models_dir)
        )
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(target),
            token=token or os.environ.get("BONSAI_TOKEN") or None,
            max_workers=max_workers,
        )
    validate_model_layout(target)
    return target


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    missing = [
        relative for relative in REQUIRED_FILES if not (root / relative).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Model snapshot is missing required Bonsai files:\n{formatted}"
        )
    return root
