from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

from ..models import cache
from ..turboquant import TurboQuantKVCache, turboquant_enabled

DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_QUANTIZED_KV_START = 5000

# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


def set_generation_device(device_name: str):
    """Set the default device and rebuild the generation streams.

    Modules bind ``generation_stream`` by name at import time, so the new
    stream is also patched into any already-imported generation modules.
    """
    import sys

    device = mx.cpu if device_name == "cpu" else mx.gpu
    mx.set_default_device(device)
    stream = mx.new_thread_local_stream(device)
    global generation_stream
    generation_stream = stream
    for mod_name in (
        "mlx_vlm.generate.ar",
        "mlx_vlm.generate.diffusion",
        "mlx_vlm.generate.dispatch",
        "mlx_vlm.speculative.common",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "generation_stream"):
            mod.generation_stream = stream


def _policy_enabled(policy) -> bool:
    return bool(getattr(policy, "enabled", policy))


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
):
    if kv_bits is None:
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
                    return TurboQuantKVCache(bits=kv_bits)
                if entry.offset < quantized_kv_start:
                    return entry
                return TurboQuantKVCache.from_cache(entry, bits=kv_bits)
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
    if not mx.metal.is_available() or mx.default_device().type != mx.DeviceType.gpu:
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
