"""Public generation API."""

from .ar import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    PromptProcessingBatch,
    batch_generate,
    generate_step,
)
from .cli import main, parse_arguments
from .common import (
    GenerationResult,
    PromptCacheState,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from .dispatch import generate, stream_generate
from .image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
    generate_image,
    image_generation_model_class,
    image_to_b64_json,
    image_to_png_bytes,
    is_image_generation_model,
    load_image_generation_model,
)

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerationResult",
    "ImageGenerationModel",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "PromptCacheState",
    "PromptProcessingBatch",
    "batch_generate",
    "generate",
    "generate_image",
    "generate_step",
    "generation_stream",
    "image_generation_model_class",
    "image_to_b64_json",
    "image_to_png_bytes",
    "is_image_generation_model",
    "load_image_generation_model",
    "main",
    "maybe_quantize_kv_cache",
    "parse_arguments",
    "stream_generate",
    "wired_limit",
]


def __getattr__(name):
    from . import ar, dispatch, image

    if hasattr(dispatch, name):
        return getattr(dispatch, name)
    if hasattr(image, name):
        return getattr(image, name)
    return getattr(ar, name)


def __dir__():
    from . import ar, dispatch, image

    return sorted(set(__all__) | set(dir(ar)) | set(dir(dispatch)) | set(dir(image)))
