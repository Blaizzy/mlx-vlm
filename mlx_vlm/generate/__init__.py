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
    GenerationResponse,
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
    "GenerationResponse",
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
