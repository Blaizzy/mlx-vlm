from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from mlx_vlm.models.flux2.config import Flux2Variant, get_variant

DOWNLOAD_PATTERNS = (
    "model_index.json",
    "vae/*.safetensors",
    "vae/*.json",
    "transformer/*.safetensors",
    "transformer/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "tokenizer/**",
    "added_tokens.json",
    "chat_template.jinja",
)


def download_model(
    variant: str | Flux2Variant = "flux2-klein-4b",
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
        model_path = Path(snapshot_download(**kwargs))
        return validate_model_layout(model_path)
    except FileNotFoundError:
        cached = find_valid_cached_snapshot(spec)
        if cached is not None:
            return cached
        raise


def find_valid_cached_snapshot(variant: str | Flux2Variant) -> Path | None:
    spec = get_variant(variant)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{spec.repo_id.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return None
    for snapshot in sorted(
        snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            return validate_model_layout(snapshot)
        except FileNotFoundError:
            continue
    return None


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Model path does not exist: {root}")

    required_globs = (
        "transformer/*.safetensors",
        "text_encoder/*.safetensors",
        "vae/*.safetensors",
        "tokenizer/tokenizer.json",
    )
    missing = [pattern for pattern in required_globs if not list(root.glob(pattern))]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            f"Model snapshot is missing required Flux2 files:\n{formatted}"
        )
    return root
