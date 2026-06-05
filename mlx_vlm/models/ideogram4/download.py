from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import (
    IDEOGRAM_4_FP8_REPO_ID,
    Ideogram4Variant,
    get_variant,
    variant_from_local_path,
)

DOWNLOAD_PATTERNS = (
    "model_index.json",
    "LICENSE.md",
    "scheduler/*.json",
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.safetensors.index.json",
    "tokenizer/**",
    "transformer/*.json",
    "transformer/*.safetensors",
    "transformer/*.safetensors.index.json",
    "unconditional_transformer/*.json",
    "unconditional_transformer/*.safetensors",
    "unconditional_transformer/*.safetensors.index.json",
    "vae/*.json",
    "vae/*.safetensors",
)


def download_model(
    variant: str | Ideogram4Variant = IDEOGRAM_4_FP8_REPO_ID,
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    max_workers: int = 16,
) -> Path:
    spec = get_variant(variant)
    if local_dir is None and not force_download:
        cached = find_valid_cached_snapshot(spec)
        if cached is not None:
            return cached

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

    try:
        return validate_model_layout(Path(snapshot_download(**kwargs)))
    except Exception as exc:
        message = str(exc)
        if "GatedRepoError" in type(exc).__name__ or "gated" in message.lower():
            raise RuntimeError(
                "ideogram-ai/ideogram-4-fp8 is gated. Accept the Hugging Face "
                "model terms and provide HF_TOKEN before loading it."
            ) from exc
        raise


def find_valid_cached_snapshot(variant: str | Ideogram4Variant) -> Path | None:
    spec = get_variant(variant)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots = cache_root / f"models--{spec.repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in sorted(
        snapshots.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True
    ):
        try:
            return validate_model_layout(snapshot)
        except (FileNotFoundError, ValueError):
            continue
    return None


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")
    variant_from_local_path(root)
    required_globs = (
        "transformer/*.safetensors",
        "unconditional_transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "vae/*.safetensors",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "unconditional_transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    )
    missing = [pattern for pattern in required_globs if not list(root.glob(pattern))]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Model snapshot is missing required Ideogram 4 files:\n{formatted}"
        )
    return root
