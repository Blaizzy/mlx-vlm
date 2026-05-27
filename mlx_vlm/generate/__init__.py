"""Public generation API."""

from .common import (
    GenerationResult,
    PromptCacheState,
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from .cli import main, parse_arguments
from .dispatch import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    PromptProcessingBatch,
    batch_generate,
    generate,
    generate_step,
    stream_generate,
)

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
    from . import dispatch

    return getattr(dispatch, name)


def __dir__():
    from . import dispatch

    return sorted(set(__all__) | set(dir(dispatch)))
