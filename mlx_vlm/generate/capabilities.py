from __future__ import annotations

from pathlib import Path
from typing import Any

from ..embedding_loader import EMBEDDING_MODEL_REMAPPING
from ..utils import _has_config, get_model_and_args, load_config
from .edit_image import is_image_edit_model
from .image import (
    _image_model_class_for_type,
    _local_image_model_types,
    _model_type_from_id,
)

IMAGE_GENERATION = "image_generation"
IMAGE_EDITING = "image_editing"
TEXT_GENERATION = "text_generation"
VISION = "vision"
AUDIO = "audio"
EMBEDDINGS = "embeddings"


def model_capabilities(
    model: str | None,
    *,
    snapshot_path: str | Path | None = None,
) -> list[str]:
    """Full capability tokens for a model, without loading it.

    Registry + local config (config.json) only. Image tokens derive from the
    model registries; text/vision/audio/embedding tokens derive from the
    repo's config.json when a snapshot is available locally (HF cache or a
    loaded local path). Unknown models report [] and never block callers.
    """
    if model is None:
        return []
    caps: list[str] = []

    # --- Image capabilities (registry + manifest based; no config needed) ---
    image_any = False
    model_path = Path(model).expanduser()
    local_types = (
        _local_image_model_types(str(model_path)) if model_path.exists() else ()
    )
    for model_type in (*local_types, _model_type_from_id(model)):
        model_class = _image_model_class_for_type(model_type)
        if (
            model_class is not None
            and getattr(model_class, "is_image_generation_model", False)
            and model_class.supports_model(model)
        ):
            caps.append(IMAGE_GENERATION)
            image_any = True
            break
    if is_image_edit_model(model):
        caps.append(IMAGE_EDITING)
        image_any = True

    # --- LM-family capabilities from the repo's config.json ---
    config = _read_config(snapshot_path or model)
    if config is not None:
        if not image_any:
            try:
                arch, _ = get_model_and_args(config)
            except Exception:
                arch = None
            if arch is not None:
                caps.append(TEXT_GENERATION)
                if _has_config(config, "vision_config"):
                    caps.append(VISION)
        if _has_config(config, "audio_config"):
            # TTS vs STT is not distinguishable without loading (mlx-vlm
            # delegates audio to mlx_audio); the token means "audio-capable".
            caps.append(AUDIO)
        model_type = str(config.get("model_type", "")).lower()
        if model_type in EMBEDDING_MODEL_REMAPPING:
            caps.append(EMBEDDINGS)

    return sorted(caps)


def _read_config(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        return None
    try:
        return dict(load_config(candidate))
    except Exception:
        return None
