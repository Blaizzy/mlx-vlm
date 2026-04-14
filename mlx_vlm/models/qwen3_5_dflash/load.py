"""Loader for the DFlash drafter from a local path or HuggingFace repo id."""

import glob
import json
from pathlib import Path
from typing import Union

import mlx.core as mx
from huggingface_hub import snapshot_download

from .config import DFlashConfig
from .dflash import DFlashDraftModel


def _resolve_path(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.exists():
        return p
    return Path(
        snapshot_download(
            repo_id=path_or_repo,
            allow_patterns=["*.json", "*.safetensors", "*.py"],
        )
    )


def load_dflash_drafter(
    path_or_repo: str,
    dtype: mx.Dtype = mx.bfloat16,
) -> DFlashDraftModel:
    """Load a DFlash drafter checkpoint.

    Returns an eager-initialized ``DFlashDraftModel`` with weights loaded
    from ``model.safetensors`` in ``path_or_repo``.
    """
    model_path = _resolve_path(path_or_repo)

    with open(model_path / "config.json", "r") as f:
        raw_config = json.load(f)
    config = DFlashConfig.from_hf_dict(raw_config)

    model = DFlashDraftModel(config)

    weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    weights = model.sanitize(weights)
    weights = {k: v.astype(dtype) if v.dtype != dtype else v for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model
