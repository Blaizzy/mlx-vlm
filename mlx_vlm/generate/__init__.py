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
from .edit_image import (
    ImageEditModel,
    ImageEditRequest,
    edit_image,
    image_edit_model_class,
    is_image_edit_model,
    load_image_edit_model,
)
from .image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageTask,
    generate_image,
    image_generation_model_class,
    image_to_b64_json,
    image_to_png_bytes,
    is_image_generation_model,
    load_image_generation_model,
    load_image_model,
)
from .types import GenerateKwargs, ProcessorLike
from .video_generation import (
    VideoGenerationModel,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProgressCallback,
    VideoReference,
    VideoReferenceKind,
    VideoWorkflow,
    generate_video,
    is_video_generation_model,
    load_video_generation_model,
    save_video,
    video_generation_model_class,
)

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerateKwargs",
    "GenerationResult",
    "ImageEditModel",
    "ImageEditRequest",
    "ImageGenerationModel",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageTask",
    "PromptCacheState",
    "PromptProcessingBatch",
    "ProcessorLike",
    "VideoGenerationModel",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoProgressCallback",
    "VideoReference",
    "VideoReferenceKind",
    "VideoWorkflow",
    "batch_generate",
    "edit_image",
    "generate",
    "generate_image",
    "generate_step",
    "generate_video",
    "generation_stream",
    "image_edit_model_class",
    "image_generation_model_class",
    "image_to_b64_json",
    "image_to_png_bytes",
    "is_image_edit_model",
    "is_image_generation_model",
    "is_video_generation_model",
    "load_image_edit_model",
    "load_image_generation_model",
    "load_image_model",
    "load_video_generation_model",
    "main",
    "maybe_quantize_kv_cache",
    "parse_arguments",
    "save_video",
    "stream_generate",
    "video_generation_model_class",
    "wired_limit",
]


def __getattr__(name):
    import importlib

    from . import ar, dispatch, image, video_generation

    edit_image_module = importlib.import_module("mlx_vlm.generate.edit_image")

    if hasattr(dispatch, name):
        return getattr(dispatch, name)
    if hasattr(edit_image_module, name):
        return getattr(edit_image_module, name)
    if hasattr(image, name):
        return getattr(image, name)
    if hasattr(video_generation, name):
        return getattr(video_generation, name)
    return getattr(ar, name)


def __dir__():
    import importlib

    from . import ar, dispatch, image, video_generation

    edit_image_module = importlib.import_module("mlx_vlm.generate.edit_image")

    return sorted(
        set(__all__)
        | set(dir(ar))
        | set(dir(dispatch))
        | set(dir(edit_image_module))
        | set(dir(image))
        | set(dir(video_generation))
    )
