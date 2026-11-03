from pathlib import Path

from huggingface_hub import snapshot_download

from .config import Flux3Config, repo_id

DOWNLOAD_PATTERNS = (
    "model.safetensors",
    "config.json",
    "manifest.json",
    "ae.safetensors",
    "text_encoder/*",
    "README.md",
    "LICENSE.md",
    "NOTICE.md",
)


def download_model(model="black-forest-labs/flux-3-image-klein", **kwargs):
    path = snapshot_download(
        repo_id(model), allow_patterns=list(DOWNLOAD_PATTERNS), **kwargs
    )
    return validate_model_layout(path)


def validate_model_layout(path):
    root = Path(path).expanduser()
    Flux3Config.from_path(root)
    required = (
        "model.safetensors",
        "ae.safetensors",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
        "text_encoder/*.safetensors",
    )
    missing = [pattern for pattern in required if not list(root.glob(pattern))]
    if missing:
        raise FileNotFoundError(
            f"Missing Flux3 files under {root}: {', '.join(missing)}"
        )
    return root
