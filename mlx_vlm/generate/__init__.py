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

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerationResult",
    "PromptCacheState",
    "PromptProcessingBatch",
    "batch_generate",
    "generate",
    "generate_step",
    "generation_stream",
    "main",
    "maybe_quantize_kv_cache",
    "parse_arguments",
    "stream_generate",
    "wired_limit",
]


def __getattr__(name):
    from . import ar, dispatch

    if hasattr(dispatch, name):
        return getattr(dispatch, name)
    return getattr(ar, name)


def __dir__():
    from . import ar, dispatch

    return sorted(set(__all__) | set(dir(ar)) | set(dir(dispatch)))
