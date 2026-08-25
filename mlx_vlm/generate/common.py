from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from ..kv_quant import from_legacy as kv_quant_from_legacy
from ..models import cache
from ..turboquant import HybridQuantKVCache, TurboQuantKVCache, turboquant_enabled

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_QUANTIZED_KV_START = 5000

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_PREFILL_STEP_SIZE = 2048
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH = 64
DEFAULT_DIFFUSION_MAX_DENOISING_STEPS = 48

# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


def _policy_enabled(policy) -> bool:
    return bool(getattr(policy, "enabled", policy))


def _default_prefill_step_size_for_offload(
    model, prefill_step_size: Optional[int], draft_model, default: int
) -> Optional[int]:
    """A one-shot lazy prefill pins every touched expert in a single graph
    until the final eval, defeating ``ExpertStore``'s LRU budget regardless
    of ``prefill_step_size``'s own chunked-prefill benefits elsewhere -- so
    an explicit ``None`` (chunking off) against an expert-offload model is a
    footgun, not a valid choice, unless a drafter already has its own reason
    to require an unchunked prefill."""
    if (
        prefill_step_size is None
        and draft_model is None
        and getattr(model, "moe_offload_store", None) is not None
    ):
        logger.warning(
            "prefill_step_size=None with an expert-offload model: falling "
            "back to prefill_step_size=%d instead of pinning every touched "
            "expert resident until the final eval.",
            default,
        )
        return default
    return prefill_step_size


def _chunked_prefill_enabled(
    model,
    *,
    input_ids=None,
    inputs_embeds=None,
    prompt_cache=None,
    draft_model=None,
    draft_kind=None,
    prefill_kwargs=None,
) -> bool:
    prefill_kwargs = prefill_kwargs or {}
    candidates = [model]
    language_model = getattr(model, "language_model", None)
    if language_model is not None and language_model is not model:
        candidates.append(language_model)

    for candidate in candidates:
        policy = getattr(candidate, "chunked_prefill_policy", None)
        if callable(policy):
            return _policy_enabled(
                policy(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    prompt_cache=prompt_cache,
                    draft_model=draft_model,
                    draft_kind=draft_kind,
                    prefill_kwargs=prefill_kwargs,
                )
            )

    if any(getattr(candidate, "no_chunked_prefill", False) for candidate in candidates):
        return False

    # Hidden-state speculative prefill is model-contract dependent. Keep unknown
    # target models conservative unless they expose a chunked_prefill_policy.
    return draft_model is None


def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    kv_key_bits: Optional[float] = None,
    kv_value_bits: Optional[float] = None,
    kv_key_scheme: Optional[str] = None,
    kv_value_scheme: Optional[str] = None,
):
    if kv_bits is None:
        return

    policy = kv_quant_from_legacy(
        kv_bits,
        kv_quant_scheme,
        kv_group_size,
        kv_key_bits,
        kv_value_bits,
        kv_key_scheme,
        kv_value_scheme,
    )
    if policy is not None and not policy.is_homogeneous:

        def hybridize(entry):
            if isinstance(entry, (HybridQuantKVCache, cache.RotatingKVCache)):
                return entry
            if isinstance(entry, cache.KVCache):
                if entry.offset >= quantized_kv_start or entry.offset == 0:
                    built = HybridQuantKVCache(policy)
                    if entry.offset:
                        built.update_and_fetch(*entry.state)
                    return built
                return entry
            if isinstance(entry, cache.CacheList):
                entry.caches = [hybridize(sub) for sub in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub in enumerate(entry):
                    entry[i] = hybridize(sub)
                return entry
            if isinstance(entry, tuple):
                return tuple(hybridize(sub) for sub in entry)
            return entry

        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            prompt_cache[index] = hybridize(layer_cache)
        return

    if turboquant_enabled(kv_bits, kv_quant_scheme):

        def quantize_entry(entry):
            if isinstance(entry, TurboQuantKVCache):
                return entry
            if isinstance(entry, cache.RotatingKVCache):
                return entry
            if isinstance(entry, cache.KVCache):
                if entry.offset == 0:
                    # Empty: replace so update_and_fetch quantizes on the fly
                    return TurboQuantKVCache(
                        bits=kv_bits,
                        key_bits=kv_key_bits,
                        value_bits=kv_value_bits,
                    )
                if entry.offset < quantized_kv_start:
                    return entry
                return TurboQuantKVCache.from_cache(
                    entry,
                    bits=kv_bits,
                    key_bits=kv_key_bits,
                    value_bits=kv_value_bits,
                )
            if isinstance(entry, cache.CacheList):
                entry.caches = [quantize_entry(sub_entry) for sub_entry in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub_entry in enumerate(entry):
                    entry[i] = quantize_entry(sub_entry)
                return entry
            if isinstance(entry, tuple):
                return tuple(quantize_entry(sub_entry) for sub_entry in entry)
            return entry

        # Skip the last layer (before final norm/LM head); it is sensitive to
        # quantization in deep models.
        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            prompt_cache[index] = quantize_entry(layer_cache)
        return

    for index, layer_cache in enumerate(prompt_cache):
        if (
            hasattr(layer_cache, "to_quantized")
            and layer_cache.offset >= quantized_kv_start
        ):
            prompt_cache[index] = layer_cache.to_quantized(
                group_size=kv_group_size,
                bits=int(kv_bits),
            )


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """Temporarily set the wired memory limit for generation."""
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for stream in streams:
                mx.synchronize(stream)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    cached_tokens: int = 0
    finish_reason: Optional[str] = None
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0
    is_draft: bool = False
    draft_text: str = ""
    text_already_printed: bool = False
    diffusion_step: int = 0
    diffusion_total_steps: int = 0
    diffusion_canvas_index: int = 0
    diffusion_block_complete: bool = False


class PromptCacheState:
    """Holds KV cache and token history across conversation turns."""

    def __init__(self):
        self.cache: Optional[List[Any]] = None
        self.token_ids: Optional[List[int]] = None

    def find_prefix_length(self, new_ids: list) -> int:
        """Return the number of leading tokens that match the cached ids."""
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len

    def update(self, token_ids: list, kv_cache: list):
        """Store the full token sequence and corresponding KV cache."""
        self.token_ids = list(token_ids)
        self.cache = kv_cache
