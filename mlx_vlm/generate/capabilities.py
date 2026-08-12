from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ..embedding_loader import EMBEDDING_MODEL_REMAPPING
from ..tool_parsers import _infer_tool_parser
from ..utils import _has_config, get_model_and_args, load_config
from .dispatch import DEFAULT_THINKING_END_TOKEN, DEFAULT_THINKING_START_TOKEN
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
TOOLS = "tools"
SPEECH_TO_TEXT = "speech_to_text"
TEXT_TO_SPEECH = "text_to_speech"
REASONING = "reasoning"
VIDEO = "video"
DRAFTER = "drafter"

# Loaded-model kind hints carrying server-side truth the config files cannot
# express (audio direction, embedding-only serving). Early-returning here
# also stops a loaded audio repo from falling through to the text-family
# checks and wrongly claiming text_generation.
_AUDIO_KIND_TOKENS = {
    "audio_tts": (AUDIO, TEXT_TO_SPEECH),
    "audio_stt": (AUDIO, SPEECH_TO_TEXT),
}

_DRAFTER_PACKAGE = "mlx_vlm.speculative.drafters"


def model_capabilities(
    model: str | None,
    *,
    snapshot_path: str | Path | None = None,
    kind_hint: str | None = None,
    supports_video_input: bool = False,
) -> list[str]:
    """Full capability tokens for a model, without loading it.

    Registry + local config (config.json, tokenizer_config.json) only.
    Image tokens derive from the model registries; text/vision/audio/
    embedding/tool/reasoning/drafter/video tokens derive from the repo's
    config when a snapshot is available locally (HF cache or a loaded local
    path). ``kind_hint`` and ``supports_video_input`` carry facts only the
    loader knows (audio direction, video-native processors) for already
    loaded entries. Unknown models report [] and never block callers.
    """
    if model is None:
        return []

    # --- Loaded-model truths that config files cannot express ---
    if kind_hint in _AUDIO_KIND_TOKENS:
        return sorted(_AUDIO_KIND_TOKENS[kind_hint])
    if kind_hint == "embedding":
        return [EMBEDDINGS]
    if kind_hint == "audio":
        return [AUDIO]

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

    # --- LM-family capabilities from the repo's config files ---
    config = _read_config(snapshot_path or model)
    if config is not None:
        cfg_type = str(config.get("model_type", "")).lower()
        is_embedding_only = _is_embedding_only(cfg_type)
        if not image_any and not is_embedding_only:
            try:
                arch, model_type = get_model_and_args(config)
            except Exception:
                arch, model_type = None, None
            if arch is not None:
                caps.append(TEXT_GENERATION)
                if _has_config(config, "vision_config"):
                    caps.append(VISION)
                templates = _chat_templates(snapshot_path or model)
                if any(_infer_tool_parser(t) is not None for t in templates):
                    caps.append(TOOLS)
                if any(
                    DEFAULT_THINKING_START_TOKEN in t
                    and DEFAULT_THINKING_END_TOKEN in t
                    for t in templates
                ):
                    caps.append(REASONING)
                if _is_video_generator(arch) or supports_video_input:
                    caps.append(VIDEO)
                if _is_drafter(arch, model_type):
                    caps.append(DRAFTER)
        if _has_config(config, "audio_config"):
            # TTS vs STT is not distinguishable without loading (mlx-vlm
            # delegates audio to mlx_audio); the token means "audio-capable".
            caps.append(AUDIO)
        if _can_embed(cfg_type):
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


def _chat_templates(path: str | Path) -> list[str]:
    """Return chat_template strings from tokenizer_config.json (no load)."""
    candidate = Path(path).expanduser()
    tokenizer_config = candidate / "tokenizer_config.json"
    if not tokenizer_config.is_file():
        return []
    try:
        data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    except Exception:
        return []
    template = data.get("chat_template")
    if isinstance(template, str):
        return [template]
    if isinstance(template, list):
        return [
            entry.get("template")
            for entry in template
            if isinstance(entry, dict) and isinstance(entry.get("template"), str)
        ]
    return []


def _is_embedding_only(cfg_type: str) -> bool:
    """Embedding architectures with no chat generation path (pure encoders
    and explicit *_embedding checkpoints)."""
    return cfg_type in {"bert", "modernbert", "xlm-roberta", "xlm_roberta"} or (
        cfg_type.endswith("_embedding")
    )


def _can_embed(cfg_type: str) -> bool:
    """True when the server can serve embeddings from this model: either a
    documented embedding-architecture type, or a type the embedding loader
    remaps to one (qwen3, gemma3_text, lfm2, xlm-roberta...)."""
    remapped = EMBEDDING_MODEL_REMAPPING.get(cfg_type, cfg_type)
    return remapped != cfg_type or _is_embedding_only(cfg_type)


def _is_video_generator(arch: Any) -> bool:
    """Video generation via the frames-pair hook, visible on the class."""
    model_class = getattr(arch, "Model", None)
    return model_class is not None and hasattr(
        model_class, "prepare_video_frame_pairs"
    )


def _is_drafter(arch: Any, model_type: str | None) -> bool:
    """True when the config's model_type resolves into the drafters
    namespace (the same fallback get_model_and_args uses) - i.e. this
    model *is* a draft model. A target model carrying a dflash_config or
    speculators_model_type is NOT a drafter; it runs with one attached."""
    if getattr(arch, "__name__", "").startswith(_DRAFTER_PACKAGE):
        return True
    if not model_type:
        return False
    try:
        importlib.import_module(f"{_DRAFTER_PACKAGE}.{model_type}")
    except Exception:
        return False
    return True
