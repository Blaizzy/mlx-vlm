import argparse
import codecs
import contextlib
import functools
import json
import logging
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache as mlx_maybe_quantize_kv_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from . import apc as _apc
from .models import cache
from .prompt_utils import apply_chat_template
from .tokenizer_utils import make_streaming_detokenizer
from .turboquant import BatchTurboQuantKVCache, TurboQuantKVCache, turboquant_enabled
from .utils import (
    StoppingCriteria,
    ThinkingBudgetCriteria,
    group_images_by_shape,
    load,
    prepare_inputs,
)

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_VIDEO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"
DEFAULT_QUANTIZED_KV_START = 5000
DEFAULT_PREFILL_STEP_SIZE = 2048


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=DEFAULT_AUDIO,
        help="URL or path of the audio to process.",
    )
    parser.add_argument(
        "--video",
        type=str,
        nargs="+",
        default=DEFAULT_VIDEO,
        help="URL or path of the video to process.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames-per-second to sample from --video.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument("--verbose", action="store_false", help="Detailed output.")
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend. Fractional --kv-bits values use "
        "TurboQuant automatically.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model from Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model version to use (branch, tag, commit).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Enable activation quantization for QQLinear layers. "
        "Only supported for models quantized with 'nvfp4' or 'mxfp8' modes.",
    )
    parser.add_argument(
        "--processor-kwargs",
        type=json.loads,
        default={},
        help="Extra processor kwargs as JSON. "
        'Example: --processor-kwargs \'{"cropping": false, "max_patches": 3}\'',
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of tokens to process per prefill step. "
        "Lower values reduce peak memory usage but may be slower. "
        "Try 512 or 256 if you hit GPU memory errors during prefill.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Speculative drafter path or HF id (e.g. z-lab/Qwen3.5-4B-DFlash).",
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "mtp"],
        help="Drafter family. Supported: 'dflash' (Qwen3.5 DFlash), "
        "'mtp' (Gemma 4 Multi-Token Prediction / Assistant model). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in the chat template (e.g. for Qwen3.5).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing the end-of-thinking token.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=DEFAULT_THINKING_START_TOKEN,
        help="Token that marks the start of a thinking block (default: %(default)s).",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_TOKEN,
        help="Token that marks the end of a thinking block (default: %(default)s).",
    )

    return parser.parse_args()


def normalize_resize_shape(
    values: Optional[Sequence[int]],
) -> Optional[Tuple[int, int]]:
    if values is None:
        return None
    if not (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) in (1, 2)
        and all(type(value) is int for value in values)
    ):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    return (values[0], values[0]) if len(values) == 1 else tuple(values)


# A stream on the default device just for generation
generation_stream = mx.new_thread_local_stream(mx.default_device())


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

        # Skip the last layer (before final norm/LM head) — it's highly
        # sensitive to quantization in deep models (e.g. gemma-4-31b).
        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            prompt_cache[index] = quantize_entry(layer_cache)
        return

    mlx_maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=int(kv_bits),
    )


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
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
            for s in streams:
                mx.synchronize(s)
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


class PromptCacheState:
    """Holds KV cache and token history across conversation turns.

    Pass this to stream_generate via the ``prompt_cache_state`` kwarg to
    reuse the KV cache from previous turns.  Only the new tokens (after
    the common prefix) are processed, avoiding redundant prefill.
    """

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


def _prime_cached_prefix_rope_state(
    model: nn.Module,
    full_input_ids: mx.array,
    mask: Optional[mx.array],
    kwargs: Dict[str, Any],
) -> bool:
    """Prime Qwen-style mRoPE metadata before a cached-prefix trim.

    Qwen VL language models keep ``_rope_deltas`` on the model object and use
    it when continuing from a non-empty KV cache. If APC trims the prompt to
    only the uncached suffix, the suffix alone is not enough to recompute the
    original prompt's RoPE delta, so derive it from the full prompt first.
    """
    lm = getattr(model, "language_model", None)
    get_rope_index = getattr(lm, "get_rope_index", None)
    if not callable(get_rope_index):
        return True
    if not (hasattr(lm, "_rope_deltas") or hasattr(lm, "_position_ids")):
        return True
    try:
        position_ids, rope_deltas = get_rope_index(
            full_input_ids,
            kwargs.get("image_grid_thw", None),
            kwargs.get("video_grid_thw", None),
            mask,
        )
    except Exception as e:
        logger.warning(
            "Could not prime cached-prefix RoPE state; falling back to cold prefill: %s",
            e,
        )
        return False
    if hasattr(lm, "_position_ids"):
        lm._position_ids = position_ids
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = rope_deltas
    kwargs["rope_deltas"] = rope_deltas
    return True


def _speculative_walk(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    """Exact-greedy speculative-decoding walk.

    Accept drafted tokens up to the first mismatch with the target's
    greedy choice, then take the target's bonus at that position.
    Returns ``(accepted_count, new_tokens)`` with ``new_tokens``
    truncated to ``budget``.
    """
    n_draft = draft_tokens.shape[1]
    combined = mx.concatenate(
        [draft_tokens.reshape(-1), target_tokens.reshape(-1)]
    ).tolist()
    d = combined[:n_draft]
    t = combined[n_draft:]
    accepted = next((i for i in range(len(d)) if d[i] != t[i]), len(d))
    new_tokens = (d[:accepted] + [t[accepted]])[:budget]
    return accepted, new_tokens


def _speculative_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Per-sequence speculative walk for B > 1.

    Returns ``(accepted_list, new_tokens_list)`` where each entry
    corresponds to one sequence in the batch.
    """
    B = draft_tokens.shape[0]
    n_draft = draft_tokens.shape[1]
    combined = mx.concatenate(
        [draft_tokens.reshape(B, -1), target_tokens.reshape(B, -1)], axis=1
    ).tolist()
    accepted_list: List[int] = []
    new_tokens_list: List[List[int]] = []
    for i in range(B):
        d = combined[i][:n_draft]
        t = combined[i][n_draft:]
        acc = next((j for j in range(len(d)) if d[j] != t[j]), len(d))
        new = (d[:acc] + [t[acc]])[: budgets[i]]
        accepted_list.append(acc)
        new_tokens_list.append(new)
    return accepted_list, new_tokens_list


def _mtp_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
) -> Generator[Tuple[int, None], None, None]:
    """Gemma 4 MTP (Single-Position Multi-Token) speculative-decoding round loop.

    Mirrors ``_dflash_rounds`` but with three differences:
    (1) the drafter consumes the target's last-layer hidden + last-layer
    shared K/V per layer-type rather than concatenated multi-layer hiddens;
    (2) ``draft_block`` is autoregressive (K small forwards) rather than a
    single masked forward; (3) ``rollback_speculative_cache`` ignores
    ``gdn_states`` (Gemma 4 has no SSM/GDN state).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "MTP speculative decoding currently only supports gemma4."
        )

    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_model.reset(model)

    # Hidden from prefill is full prompt-length; reduce to a single slot.
    # The semantically-correct choice is the *last* prompt token's hidden:
    # the just-sampled bonus is the next-token prediction from that position,
    # so its embedding paired with that hidden is what the drafter expects.
    # (HF's literal ``[:, n_last_matches:n_last_matches+1]`` with ``n_matches=0``
    # on the first round picks position 0, which is BOS — they get away with
    # it because subsequent rounds slice into the per-call verify hidden, but
    # the round-1 acceptance is wasted. We don't replicate that quirk.)
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    kv_offset = int(prompt_cache[0].offset)
    draft_model.set_shared_kv(shared_kv_states, kv_offset)

    b = first_bonus
    emitted = 1  # caller already yielded the first bonus

    while emitted < max_tokens:
        bs = min(block_total, max_tokens - emitted + 1)
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b, hidden, None, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
            hidden_full = verify_out.hidden_states[-1]  # [B, bs, backbone]
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        accepted, new_tokens = _speculative_walk(
            draft_tokens, target_tokens, max_tokens - emitted
        )
        draft_model.accept_lens.append(accepted)

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        # Hidden for next round: pick the slot of the newly accepted bonus.
        hidden = hidden_full[:, accepted : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, None, accepted, bs)

        # Slice shared_kv_states to the post-rollback length and rebind.
        rejected = bs - (accepted + 1)
        next_shared_kv = {}
        for k, kv in verify_out.shared_kv_states.items():
            K, V = kv
            valid = K.shape[-2] - rejected
            if valid <= 0 or valid >= K.shape[-2]:
                next_shared_kv[k] = (
                    (K, V) if valid >= K.shape[-2] else (K[..., :1, :], V[..., :1, :])
                )
            else:
                next_shared_kv[k] = (K[..., :valid, :], V[..., :valid, :])
        kv_offset = int(prompt_cache[0].offset)
        draft_model.set_shared_kv(next_shared_kv, kv_offset)

        if emitted % 256 == 0:
            mx.clear_cache()


def _batch_cache_left_padding(prompt_cache: List[Any]) -> Optional[mx.array]:
    for cache_entry in prompt_cache:
        left_padding = getattr(cache_entry, "left_padding", None)
        if left_padding is not None:
            return left_padding
    return None


def _mtp_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    eos_token_ids: Optional[set] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batched Gemma 4 MTP round loop (B > 1).

    Mirrors ``_dflash_rounds_batch``: per-row state tracked by original
    index, continuous-batching filter on row finish. Differences vs DFlash
    batched: drafter consumes ``shared_kv_states`` (per-layer-type K/V
    snapshot) instead of multi-layer hidden capture, ``draft_block`` is
    autoregressive, and the per-round ``shared_kv`` snapshot is normalized
    back to the unbatched prefix-valid layout before each drafter rebind.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_model.reset(model)

    # First-round hidden: prefill output may have shape [B, L, H]; reduce
    # to a single slot per row (last prompt token's hidden — see comment in
    # ``_mtp_rounds`` for rationale).
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    # Per-row state. ``positions`` is the absolute position id of each
    # row's pending bonus (= row's logical KV length). All rows start at
    # ``L_prefill`` and advance by ``accepted_i + 1`` per round.
    offset0 = prompt_cache[0].offset
    if isinstance(offset0, mx.array):
        L_prefill = int(offset0.max().item())
        positions = [int(x) for x in offset0.tolist()]
    else:
        L_prefill = int(offset0)
        positions = [L_prefill] * B
    draft_model.set_shared_kv(
        shared_kv_states,
        kv_offset=L_prefill,
        position=mx.array(positions),
        left_padding=_batch_cache_left_padding(prompt_cache),
    )

    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = min(block_total, min(remaining))
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft (autoregressive K-step). hidden / shared_kv state was set
        # via set_shared_kv above; the drafter pulls it from there.
        draft_tokens = draft_model.draft_block(
            b_arr, hidden, None, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
            hidden_full = verify_out.hidden_states[-1]  # [B_active, bs, H]
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        # Walk per-row
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
        )
        for a in accepted_list:
            draft_model.accept_lens.append(a)

        max_a = max(accepted_list)
        accepted_arr = mx.array(accepted_list)

        # Per-row hidden: each row picks its own accepted slot from
        # hidden_full. Build [B_active, 1, H] with row-i's hidden at
        # position accepted_list[i].
        if max_a < bs - 1 or any(a < max_a for a in accepted_list):
            row_idx = mx.arange(n_active)
            col_idx = mx.array(accepted_list)
            # gather: hidden_full[row_idx, col_idx, :] -> [B_active, H]
            hidden = hidden_full[row_idx, col_idx, :][:, None, :]
        else:
            hidden = hidden_full[:, -1:, :]

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if eos_token_ids is not None and tok in eos_token_ids:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens and per-row positions
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]
            positions[orig] = positions[orig] + accepted_list[j] + 1

        # Rollback target cache (uniform trim by ``bs - max_a - 1`` plus
        # per-row tail-zero on rows that accepted less).
        if max_a < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, None, accepted_arr, bs)

        # Slice + tail-zero ``verify_out.shared_kv_states`` to match the
        # post-rollback target cache. ``set_shared_kv()`` will normalize the
        # resulting hybrid layout back into a prefix-valid drafter view.
        rejected_global = bs - (max_a + 1)
        next_shared_kv = {}
        for k, kv in verify_out.shared_kv_states.items():
            K, V = kv
            valid = K.shape[-2] - rejected_global
            if valid >= K.shape[-2]:
                K_next, V_next = K, V
            elif valid <= 0:
                K_next = K[..., :1, :]
                V_next = V[..., :1, :]
            else:
                K_next = K[..., :valid, :]
                V_next = V[..., :valid, :]
            # Per-row tail-zero on rows that accepted less than max_a.
            if any(a < max_a for a in accepted_list):
                # K_next/V_next shape: [B_active, H, valid, D]
                # For row i, zero positions [valid - max_a + accepted_i, valid).
                # (verify_start = valid - (max_a + 1), and tail begins at
                # verify_start + accepted_i + 1 = valid - max_a + accepted_i.)
                K_arr = mx.array(K_next)  # ensure materialized for slicing
                V_arr = mx.array(V_next)
                K_arr = mx.array(K_arr)
                V_arr = mx.array(V_arr)
                mask_rows = mx.arange(K_next.shape[-2])  # [valid]
                # Build per-row mask: True where position should be kept.
                # Shape [B_active, valid]. Row i keeps positions [0, valid - max_a + accepted_i).
                keep_lens = mx.array(
                    [valid - max_a + a for a in accepted_list], dtype=mx.int32
                )  # [B_active]
                keep_mask = mask_rows[None, :] < keep_lens[:, None]  # [B_active, valid]
                keep_f = keep_mask.astype(K_next.dtype)[:, None, :, None]  # broadcast
                K_next = K_next * keep_f
                V_next = V_next * keep_f
            next_shared_kv[k] = (K_next, V_next)

        # Continuous batching: filter finished sequences. Only safe when
        # the caches expose a .filter() method (e.g. BatchKVCache); the
        # plain KVCache / RotatingKVCache do not, so we keep all rows
        # in the batch and just stop emitting for finished rows. End the
        # round-loop when every row has finished.
        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)
        if all(finished[active_idx[j]] for j in range(n_active)):
            break
        if cache_filterable:
            keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
            if len(keep_slots) < n_active:
                keep_mx = mx.array(keep_slots, dtype=mx.int32)
                for c in prompt_cache:
                    c.filter(keep_mx)
                hidden = hidden[keep_mx]
                for k in next_shared_kv:
                    K_next, V_next = next_shared_kv[k]
                    next_shared_kv[k] = (K_next[keep_mx], V_next[keep_mx])
                active_idx = [active_idx[j] for j in keep_slots]

        # Re-bind drafter with new shared_kv and per-row positions.
        positions_active = [positions[active_idx[j]] for j in range(len(active_idx))]
        offset0 = prompt_cache[0].offset
        new_kv_offset = (
            int(offset0.max().item()) if isinstance(offset0, mx.array) else int(offset0)
        )
        draft_model.set_shared_kv(
            next_shared_kv,
            kv_offset=new_kv_offset,
            position=mx.array(positions_active),
            left_padding=_batch_cache_left_padding(prompt_cache),
        )

        if sum(emitted) % 256 == 0:
            mx.clear_cache()


def _dflash_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
) -> Generator[Tuple[int, None], None, None]:
    """DFlash speculative-decoding **round loop**.

    draft → verify → walk → rollback. ``generate_step`` is responsible
    for prefill, sampling the first bonus token, and packaging the
    captured hidden states into ``hidden``.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "Speculative decoding with a DFlash drafter currently only "
            "supports mlx_vlm.models.qwen3_5."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_cache = draft_model.reset(model)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        bs = min(block_total, max_tokens - emitted + 1)
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b, hidden, draft_cache, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens],
                axis=1,
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden)

        # Walk
        accepted, new_tokens = _speculative_walk(
            draft_tokens, target_tokens, max_tokens - emitted
        )
        draft_model.accept_lens.append(accepted)

        # Emit
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        if accepted < bs - 1:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, bs
                )

        if emitted % 256 == 0:
            mx.clear_cache()


def _dflash_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    Yields ``(tokens_list, None)`` where ``tokens_list[i]`` is the
    token for sequence ``i`` (or ``None`` if that sequence has nothing
    to emit this step).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement " "rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_cache = draft_model.reset(model)

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index

    def _reinit_drafter():
        """Cold-restart the drafter cache after a batch change."""
        nonlocal draft_cache
        draft_cache = draft_model.make_cache()

    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = min(block_total, min(remaining))
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft
        draft_tokens = draft_model.draft_block(
            b_arr, hidden, draft_cache, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        # Walk (per-sequence)
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
        )

        min_accepted = min(accepted_list)
        accepted_arr = mx.array(accepted_list)

        if min_accepted < bs - 1:
            max_a = int(accepted_arr.max().item())
            hidden = hidden_full[:, : max_a + 1, :]
        else:
            max_a = bs - 1
            hidden = hidden_full

        for a in accepted_list:
            draft_model.accept_lens.append(a)

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if min_accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted_arr, bs
                )

        # --- Continuous batching: filter out finished sequences ---
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            # Filter target caches (BatchKVCache supports this)
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            # Filter hidden
            hidden = hidden[keep_mx]
            # Update active index mapping
            active_idx = [active_idx[j] for j in keep_slots]
            # Cold-restart drafter for the new batch size
            _reinit_drafter()

        new_total = sum(emitted)
        if new_total // 256 > total_emitted // 256:
            mx.clear_cache()
        total_emitted = new_total


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    top_k: int = DEFAULT_TOP_K,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[float] = None,
    kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    draft_model: Optional[nn.Module] = None,
    draft_kind: str = "dflash",
    draft_block_size: Optional[int] = None,
    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,
    prompt_cache_checkpoint_len: Optional[int] = None,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): Minimum probability threshold relative to the
          highest-probability token.
        top_k (int, optional): Restrict sampling to the top-k tokens.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (float, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for uniform KV cache quantization.
        kv_quant_scheme (str): KV cache quantization backend.
        quantized_kv_start (int): Start index for quantized KV cache.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage.
        draft_model (nn.Module, optional): A drafter for speculative decoding.
          When set, the decode loop is replaced by the drafter's speculative
          loop (e.g. DFlash block-diffusion). VLM prefill with image/audio
          is supported via the same ``get_input_embeddings`` path the normal
          decoder uses; decode itself is text-only. ``temperature`` and
          ``sampler`` are respected; ``logprobs`` is always ``None`` on the
          speculative path.
        draft_block_size (int, optional): Override the drafter's configured
          block size.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
    )

    if sampler is None:
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    processors = make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Speculative decoding setup
    last_outputs = None
    if draft_model is not None:
        if draft_kind == "mtp":
            # MTP drafter consumes target's last-layer hidden + shared K/V
            # (per layer-type) rather than per-layer hidden captures.
            kwargs["return_hidden"] = True
            kwargs["return_shared_kv"] = True
        else:
            kwargs["capture_layer_ids"] = list(draft_model.config.target_layer_ids)
        prefill_step_size = None
        # Reset stale mRoPE state from any previous generation.
        lm = model.language_model if hasattr(model, "language_model") else model
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs, last_outputs

        with mx.stream(generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            last_outputs = outputs
            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs

    with mx.stream(generation_stream):
        # Get input embeddings (handles both multimodal and text-only)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )

        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        if getattr(model, "no_chunked_prefill", False):
            prefill_step_size = None
        checkpoint_len = (
            int(prompt_cache_checkpoint_len)
            if prompt_cache_checkpoint is not None
            and prompt_cache_checkpoint_len is not None
            else None
        )
        checkpoint_done = False
        should_chunk = (
            prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size
        ) or (
            checkpoint_len is not None and 0 < checkpoint_len < inputs_embeds.shape[1]
        )
        if prefill_step_size is not None and should_chunk:
            # Chunked prefill with embeddings
            total_tokens = inputs_embeds.shape[1]
            processed_tokens = 0
            with tqdm(total=total_tokens, desc="Prefill", unit="tok") as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    if (
                        checkpoint_len is not None
                        and not checkpoint_done
                        and processed_tokens < checkpoint_len
                        and processed_tokens + n_to_process > checkpoint_len
                    ):
                        n_to_process = checkpoint_len - processed_tokens
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        n_to_process=n_to_process,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    processed_tokens += n_to_process
                    if (
                        checkpoint_len is not None
                        and not checkpoint_done
                        and processed_tokens == checkpoint_len
                    ):
                        prompt_cache_checkpoint(processed_tokens, prompt_cache)
                        checkpoint_done = True
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

    mx.async_eval(y)

    # Speculative decoding
    if draft_model is not None:
        B = input_ids.shape[0]
        if draft_kind == "mtp":
            shared_kv_states = last_outputs.shared_kv_states
            hidden = last_outputs.hidden_states[-1]
            if B == 1:
                mx.eval(y)
                yield y.item(), logprobs
                yield from _mtp_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    shared_kv_states,
                    first_bonus=y.item(),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=input_ids.dtype,
                )
            else:
                mx.eval(y)
                # ``y`` is shape (B,) from sampler — no squeeze needed.
                first_bonus = y if y.ndim == 1 else y.reshape(-1)
                yield first_bonus.tolist(), logprobs
                # Surface EOS token IDs from the model config so per-row
                # stops are detected inside the round loop.
                eos = getattr(model.config, "eos_token_id", None)
                if isinstance(eos, int):
                    eos_set = {eos}
                elif eos is None:
                    eos_set = None
                else:
                    eos_set = set(int(x) for x in eos)
                yield from _mtp_rounds_batch(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    shared_kv_states,
                    first_bonus=first_bonus,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=input_ids.dtype,
                    eos_token_ids=eos_set,
                )
            return

        if draft_kind != "dflash":
            raise ValueError(
                f"Unknown draft_kind {draft_kind!r}. Supported: " "['dflash', 'mtp']"
            )
        hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
        if B == 1:
            mx.eval(y)
            yield y.item(), logprobs
            yield from _dflash_rounds(
                model,
                draft_model,
                prompt_cache,
                hidden,
                first_bonus=y.item(),
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
            )
        else:
            mx.eval(y)
            first_bonus = y.squeeze(-1)
            yield first_bonus.tolist(), logprobs
            yield from _dflash_rounds_batch(
                model,
                draft_model,
                prompt_cache,
                hidden,
                first_bonus=first_bonus,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_block_size=draft_block_size,
                token_dtype=input_ids.dtype,
            )
        return

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()

        if thinking_budget_criteria is not None:
            next_y = thinking_budget_criteria.apply_forced_token(next_y)
        y, logprobs = next_y, next_logprobs
        n += 1


def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    video: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image (Union[str, List[str]], optional): Image path(s) or URL(s).
        audio (Union[str, List[str]], optional): Audio file path(s).
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step. When set, enables chunked prefill which processes long prompts in
          smaller chunks to reduce peak memory usage.
        kwargs: Additional options passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[GenerationResult]: A generator producing GenerationResult objects
          containing the generated text, tokens, and statistics.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", DEFAULT_THINKING_END_TOKEN)
    thinking_start_token = kwargs.pop(
        "thinking_start_token", DEFAULT_THINKING_START_TOKEN
    )
    enable_thinking = kwargs.pop("enable_thinking", False)

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)
    vision_cache = kwargs.pop("vision_cache", None)
    prompt_cache_state = kwargs.pop("prompt_cache_state", None)
    apc_manager: Optional[_apc.APCManager] = kwargs.pop("apc_manager", None)
    apc_tenant: Optional[str] = kwargs.pop("apc_tenant", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            videos=video,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    # Vision feature caching: reuse cached image features across turns
    if vision_cache is not None and image is not None and pixel_values is not None:
        cached = vision_cache.get(image)
        if cached is not None:
            kwargs["cached_image_features"] = cached
        elif hasattr(model, "encode_image"):
            features = model.encode_image(pixel_values)
            mx.eval(features)
            vision_cache.put(image, features)
            kwargs["cached_image_features"] = features

    # Prompt cache reuse: skip common prefix from previous turn
    reused_prefix_len = 0
    full_input_ids_list = input_ids.flatten().tolist()
    apc_blocks_in_use: List[_apc.APCBlock] = []
    apc_extra_hash = 0
    apc_mode: Optional[str] = None

    if apc_manager is not None:
        apc_mode = _apc.model_apc_mode(model.language_model)
        if apc_mode is None:
            apc_manager = None

    if apc_manager is not None:
        image_hash = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=image)
        apc_extra_hash = _apc.tenant_scoped_hash(apc_tenant, image_hash)

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)
        if prefix_len > 0 and prefix_len < input_ids.shape[1]:
            if _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                reused_prefix_len = prefix_len
                # Trim to only new tokens
                input_ids = input_ids[:, prefix_len:]
                # Only skip vision if no image tokens in the new (trimmed) tokens
                image_token_id = getattr(
                    model.config, "image_token_id", None
                ) or getattr(model.config, "image_token_index", None)
                new_ids = input_ids.flatten().tolist()
                has_image_in_new = (
                    image_token_id is not None and image_token_id in new_ids
                )
                if not has_image_in_new:
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                # Reuse the saved KV cache (trimmed to prefix length)
                kv_cache = prompt_cache_state.cache
                # Trim cache to prefix_len in case it includes generated tokens
                for c in kv_cache:
                    if hasattr(c, "keys") and c.keys is not None:
                        cached_len = c.keys.shape[2]
                        if cached_len > prefix_len:
                            c.keys = c.keys[:, :, :prefix_len, :]
                            c.values = c.values[:, :, :prefix_len, :]
                            if hasattr(c, "offset"):
                                c.offset = prefix_len
                kwargs["prompt_cache"] = kv_cache

    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    if apc_manager is not None and reused_prefix_len == 0:
        if apc_mode == "exact":
            exact_prompt_cache, exact_prefix_len = apc_manager.lookup_exact_cache(
                full_input_ids_list,
                extra_hash=apc_extra_hash,
            )
            if (
                exact_prompt_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < input_ids.shape[1]
                and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
            ):
                reused_prefix_len = exact_prefix_len
                input_ids = input_ids[:, exact_prefix_len:]
                image_token_id = getattr(
                    model.config, "image_token_id", None
                ) or getattr(model.config, "image_token_index", None)
                new_ids = input_ids.flatten().tolist()
                if image_token_id is None or image_token_id not in new_ids:
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                kwargs["prompt_cache"] = exact_prompt_cache
        else:
            matched_blocks, prefix_len = apc_manager.lookup_prefix(
                full_input_ids_list, extra_hash=apc_extra_hash
            )
            exact_prompt_cache = None
            exact_prefix_len = 0
            if prefix_len < input_ids.shape[1]:
                exact_prompt_cache, exact_prefix_len = apc_manager.lookup_exact_cache(
                    full_input_ids_list,
                    extra_hash=apc_extra_hash,
                    min_prefix_tokens=prefix_len,
                )
            disk_prompt_cache = None
            disk_prefix_len = 0
            if max(prefix_len, exact_prefix_len) < input_ids.shape[1]:
                disk_prompt_cache, disk_prefix_len = (
                    apc_manager.lookup_prefix_disk_cache(
                        full_input_ids_list,
                        extra_hash=apc_extra_hash,
                        min_prefix_tokens=max(prefix_len, exact_prefix_len),
                        allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
                    )
                )
            if (
                disk_prefix_len > max(prefix_len, exact_prefix_len)
                and disk_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = disk_prefix_len
                    input_ids = input_ids[:, disk_prefix_len:]
                    image_token_id = getattr(
                        model.config, "image_token_id", None
                    ) or getattr(model.config, "image_token_index", None)
                    new_ids = input_ids.flatten().tolist()
                    if image_token_id is None or image_token_id not in new_ids:
                        pixel_values = None
                        kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = disk_prompt_cache
            elif (
                exact_prefix_len > prefix_len and exact_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = exact_prefix_len
                    input_ids = input_ids[:, exact_prefix_len:]
                    image_token_id = getattr(
                        model.config, "image_token_id", None
                    ) or getattr(model.config, "image_token_index", None)
                    new_ids = input_ids.flatten().tolist()
                    if image_token_id is None or image_token_id not in new_ids:
                        pixel_values = None
                        kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = exact_prompt_cache
            elif prefix_len > 0 and prefix_len < input_ids.shape[1]:
                if _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    apc_blocks_in_use = matched_blocks
                    reused_prefix_len = prefix_len
                    input_ids = input_ids[:, prefix_len:]
                    image_token_id = getattr(
                        model.config, "image_token_id", None
                    ) or getattr(model.config, "image_token_index", None)
                    new_ids = input_ids.flatten().tolist()
                    if image_token_id is None or image_token_id not in new_ids:
                        pixel_values = None
                        kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = _apc.make_warm_kv_cache(
                        matched_blocks,
                        min_capacity_tokens=prefix_len + input_ids.shape[1] + 1,
                    )
                else:
                    apc_manager.release(matched_blocks)
            elif matched_blocks:
                # Full match (no new tokens to compute) — release; fall through to normal path
                apc_manager.release(matched_blocks)

    if thinking_budget is not None:
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        enable_thinking = enable_thinking and (
            thinking_start_token_id in input_ids.flatten().tolist()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    # Ensure we have a prompt_cache we can track for reuse.
    if "prompt_cache" not in kwargs:
        kwargs["prompt_cache"] = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=kwargs.get("max_kv_size", None),
        )
    tracked_cache = kwargs["prompt_cache"]

    total_prompt_tokens = reused_prefix_len + input_ids.size

    with wired_limit(model, [generation_stream]):
        detokenizer = make_streaming_detokenizer(processor)
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)
        exact_checkpoint_len = None
        exact_checkpoint = None
        if apc_manager is not None and apc_mode == "exact" and reused_prefix_len == 0:
            exact_checkpoint_len = max(
                1,
                len(full_input_ids_list) - apc_manager.exact_cache_guard_tokens,
            )

            def exact_checkpoint(prefix_len: int, prompt_cache: List[Any]) -> None:
                apc_manager.store_exact_cache(
                    full_input_ids_list[:prefix_len],
                    prompt_cache,
                    extra_hash=apc_extra_hash,
                )

        gen = generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            prompt_cache_checkpoint=exact_checkpoint,
            prompt_cache_checkpoint_len=exact_checkpoint_len,
            **kwargs,
        )
        tic = time.perf_counter()

        generated_tokens = []
        for n, (token, logprobs) in enumerate(gen):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = total_prompt_tokens / prompt_time
                tic = time.perf_counter()
                if (
                    apc_manager is not None
                    and apc_mode == "exact"
                    and reused_prefix_len == 0
                ):
                    try:
                        apc_manager.store_exact_cache(
                            full_input_ids_list,
                            tracked_cache,
                            extra_hash=apc_extra_hash,
                        )
                    except Exception as e:
                        logger.warning("APC exact-cache store failed: %s", e)

            generated_tokens.append(token)

            # Check thinking budget and force token if needed
            if thinking_criteria is not None:
                thinking_criteria(token)

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=n + 1,
                total_tokens=total_prompt_tokens + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=n + 1,
            total_tokens=total_prompt_tokens + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )

        # Save cache state for potential reuse on next turn
        all_ids: Optional[List[int]] = None
        if prompt_cache_state is not None:
            all_ids = full_input_ids_list + [
                t.item() if hasattr(t, "item") else t for t in generated_tokens
            ]
            prompt_cache_state.update(all_ids, tracked_cache)

        # APC: harvest new blocks from the post-generation KV state.
        if apc_manager is not None and apc_mode == "block":
            try:
                if all_ids is None:
                    all_ids = full_input_ids_list + [
                        t.item() if hasattr(t, "item") else t for t in generated_tokens
                    ]
                # Snapshot keys/values up to the live offset for each layer.
                layer_keys: List[mx.array] = []
                layer_values: List[mx.array] = []
                ok = True
                for c in tracked_cache:
                    k = getattr(c, "keys", None)
                    v = getattr(c, "values", None)
                    off = getattr(c, "offset", None)
                    if k is None or v is None or off is None:
                        ok = False
                        break
                    layer_keys.append(k[..., :off, :])
                    layer_values.append(v[..., :off, :])
                if ok and layer_keys:
                    new_blocks = apc_manager.store_kv_blocks(
                        all_ids,
                        layer_keys,
                        layer_values,
                        extra_hash=apc_extra_hash,
                        skip_first_n_tokens=reused_prefix_len,
                    )
                    apc_manager.release(apc_blocks_in_use + new_blocks)
                else:
                    apc_manager.release(apc_blocks_in_use)
            except Exception as e:
                logger.warning("APC store failed: %s", e)
                apc_manager.release(apc_blocks_in_use)

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    video: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> GenerationResult:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if video is not None:
            files.extend(video if isinstance(video, list) else [video])

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in stream_generate(
        model, processor, prompt, image, audio, video, **kwargs
    ):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return GenerationResult(
                text=text,
                token=None,
                logprobs=None,
                prompt_tokens=0,
                generation_tokens=0,
                total_tokens=0,
                prompt_tps=0.0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
    )


@dataclass
class BatchGenerationResult:
    """
    Result of batch generation with optional image size tracking.

    Attributes:
        texts: Generated text for each sample
        tokens: Last generated token for each sample
        logprobs: Log probabilities for each sample
        prompt_tokens: Number of prompt tokens per sample
        generation_tokens: Number of generated tokens per sample
        total_tokens: Total tokens (prompt + generation) per sample
        prompt_tps: Prompt tokens per second per sample
        generation_tps: Generation tokens per second per sample
        peak_memory: Peak memory usage in GB
        image_sizes: Original (height, width) for each image (for tracking)
    """

    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0
    image_sizes: Optional[List[Tuple[int, int]]] = None


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([list(p) + [0] * (max_length - len(p)) for p in prompts])


_SEQUENCE_ALIGNED_PROMPT_KWARGS = {
    "attention_mask",
    "decoder_inputs_embeds",
    "deepstack_visual_embeds",
    "visual_pos_masks",
    "per_layer_inputs",
    "full_text_row_masked_out_mask",
    "position_ids",
    "pos_hw",
}

APC_PRIVATE_PROMPT_KEYS = ("_apc_tenant", "_apc_image_hash")


def _prompt_kwarg_row(v: mx.array, row_idx: int, batch_size: int) -> mx.array:
    if v.shape[0] == batch_size:
        return v[row_idx : row_idx + 1]
    return v[:1]


def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> List[dict]:
    """Normalize batched prompt kwargs into one dict per batch row.

    ``model.get_input_embeddings()`` commonly returns batch-sized tensors
    (notably ``inputs_embeds``). ``BatchGenerator.insert()`` stores prompt
    kwargs per sequence, so passing the same batched dict for every row causes
    the prompt builder to concatenate those batched tensors ``batch_size``
    times, effectively squaring the batch dimension.
    """
    if batch_size <= 1:
        return [prompt_kwargs or {}]

    rows = [{} for _ in range(batch_size)]
    for k, v in (prompt_kwargs or {}).items():
        if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
            for i in range(batch_size):
                rows[i][k] = _prompt_kwarg_row(v, i, batch_size)
        else:
            for row in rows:
                row[k] = v
    return rows


def _is_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, sequence_length: int
) -> bool:
    return (
        key in _SEQUENCE_ALIGNED_PROMPT_KWARGS
        and v.ndim >= 2
        and v.shape[1] == sequence_length
    )


def _pad_sequence_aligned_prompt_kwarg(
    v: mx.array, target_length: int, *, left: bool
) -> mx.array:
    pad = target_length - v.shape[1]
    if pad <= 0:
        return v
    pad_shape = (v.shape[0], pad) + tuple(v.shape[2:])
    pad_v = mx.zeros(pad_shape, dtype=v.dtype)
    parts = [pad_v, v] if left else [v, pad_v]
    return mx.concatenate(parts, axis=1)


def _merge_prefill_prompt_kwargs(
    prompt_kwargs_list: List[Optional[dict]],
    input_ids: List[List[int]],
) -> Tuple[mx.array, dict]:
    """Batch per-row prompt kwargs for a left-padded prefill forward."""
    lengths = [len(ids) for ids in input_ids]
    max_length = max(lengths)

    row_embeds: List[mx.array] = []
    embed_dtype = None
    embed_dim = None
    for kw, length in zip(prompt_kwargs_list, lengths):
        if not kw or kw.get("inputs_embeds") is None:
            raise ValueError("inputs_embeds is required")
        embeds = kw["inputs_embeds"]  # [1, length, D]
        embed_dtype = embeds.dtype
        embed_dim = embeds.shape[-1]
        if length < max_length:
            pad = mx.zeros(
                (embeds.shape[0], max_length - length, embed_dim),
                dtype=embed_dtype,
            )
            embeds = mx.concatenate([pad, embeds], axis=1)
        row_embeds.append(embeds)
    inputs_embeds = mx.concatenate(row_embeds, axis=0)

    merged_kwargs: dict = {}
    per_row_keys: dict = {}
    batch_size = len(prompt_kwargs_list)
    for i, (kw, length) in enumerate(zip(prompt_kwargs_list, lengths)):
        if not kw:
            continue
        for k, v in kw.items():
            if k == "inputs_embeds" or k in APC_PRIVATE_PROMPT_KEYS:
                continue
            if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                row_v = _prompt_kwarg_row(v, i, batch_size)
                if _is_sequence_aligned_prompt_kwarg(k, row_v, length):
                    row_v = _pad_sequence_aligned_prompt_kwarg(
                        row_v, max_length, left=True
                    )
                per_row_keys.setdefault(k, []).append(row_v)
            elif k not in merged_kwargs:
                merged_kwargs[k] = v
    for k, vs in per_row_keys.items():
        merged_kwargs[k] = mx.concatenate(vs, axis=0)

    return inputs_embeds, merged_kwargs


def _extend_cache(cache_a, cache_b):
    """Extend cache_a with cache_b along the batch dimension."""
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a


def _make_cache(
    model,
    left_padding,
    kv_bits=None,
    kv_group_size=64,
    kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* is set, a quantized batch cache is used instead of
    ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.

    *kv_quant_scheme* selects the quantization backend:
    - ``"uniform"`` → ``BatchQuantizedKVCache`` (``mx.quantize``)
    - ``"turboquant"`` or fractional *kv_bits* → ``BatchTurboQuantKVCache``
    """
    use_turbo = kv_bits is not None and turboquant_enabled(kv_bits, kv_quant_scheme)

    def _make_quant_cache(lp):
        if use_turbo:
            return BatchTurboQuantKVCache(lp, bits=kv_bits)
        return cache.BatchQuantizedKVCache(
            lp, group_size=kv_group_size, bits=int(kv_bits)
        )

    def to_batch_cache(c, quantize=True):
        if isinstance(c, cache.KVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ChunkedKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.SimpleKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        elif isinstance(c, tuple):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        n = len(model_cache)
        # Skip quantizing the last layer — it's sensitive to quantization
        return [
            to_batch_cache(c, quantize=(i < n - 1 if n > 2 else True))
            for i, c in enumerate(model_cache)
        ]
    else:
        if kv_bits is not None:
            n = len(model.layers)
            return [
                (
                    _make_quant_cache(left_padding)
                    if i < n - 1 or n <= 2
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [cache.BatchKVCache(left_padding) for _ in model.layers]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Tuple[int, int]]] = None


class GenerationBatch:
    """
    Batched token generator with double-buffered pipelining.

    Manages the generation phase after prompt processing, with KV caches,
    sampling, and stop detection for multiple sequences. Uses async_eval
    to overlap GPU computation with CPU processing (decode-ahead pattern).
    """

    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[List[Tuple[int, float]]] = None

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        top_logprobs_k: int = 0,
        token_context: Optional[List[List[int]]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.compute_logprobs = True
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        self.token_context = [list(ctx) for ctx in (token_context or [])]

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self._next_lps = None
        self._next_top_idx = None
        self._next_top_lp = None

        # Per-sequence MRoPE delta
        self._rope_deltas = None

    def __len__(self):
        return len(self.uids)

    def _step(self):
        """Perform one generation step with double buffering."""
        self._current_tokens = self._next_tokens
        self._current_lps = self._next_lps
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = self._rope_deltas

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if self.logits_processors and any(self.logits_processors):
            last_tokens = inputs.tolist()
            if not self.token_context:
                self.token_context = [[] for _ in self.uids]
            for i, token in enumerate(last_tokens):
                self.token_context[i].append(token)

            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    if hasattr(processor, "process_last_token"):
                        sample_logits = processor.process_last_token(
                            last_tokens[i], sample_logits
                        )
                    else:
                        sample_logits = processor(
                            mx.array(self.token_context[i]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)

        self._next_tokens = sampled
        prev_top_idx = self._next_top_idx
        prev_top_lp = self._next_top_lp

        eval_targets = [self._next_tokens]
        if self.compute_logprobs:
            self._next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
            eval_targets.append(self._next_lps)
        else:
            self._next_lps = None

        k = self.top_logprobs_k
        if k > 0:
            # argsort ascending; take last K columns and reverse for descending.
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self._next_top_idx = top_idx
            self._next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self._next_top_idx = None
            self._next_top_lp = None

        mx.async_eval(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            mx.eval(*to_eval)
            top_idx_list = prev_top_idx.tolist() if prev_top_idx is not None else None
            top_lp_list = prev_top_lp.tolist() if prev_top_lp is not None else None
            return (
                inputs.tolist(),
                self._current_lps.tolist(),
                top_idx_list,
                top_lp_list,
            )
        else:
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

    def extend(self, other: "GenerationBatch"):
        """Extend this batch with another generation batch."""
        self_was_empty = len(self.uids) == 0
        self.uids.extend(other.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, other.prompt_cache)
        self.max_tokens.extend(other.max_tokens)
        self._num_tokens.extend(other._num_tokens)
        self.token_context.extend(other.token_context)
        self.logits_processors.extend(other.logits_processors)

        if self._current_tokens is None:
            self._current_tokens = other._current_tokens
            self._current_lps = other._current_lps
        elif other._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, other._current_tokens]
            )
            if self._current_lps is not None and other._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, other._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = other._next_tokens
            self._next_lps = other._next_lps
            self._next_top_idx = other._next_top_idx
            self._next_top_lp = other._next_top_lp
        elif other._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, other._next_tokens])
            if self._next_lps is not None and other._next_lps is not None:
                self._next_lps = mx.concatenate([self._next_lps, other._next_lps])

            if (
                self._next_top_idx is not None
                and other._next_top_idx is not None
                and self._next_top_idx.shape[-1] == other._next_top_idx.shape[-1]
            ):
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, other._next_top_idx]
                )
                self._next_top_lp = mx.concatenate(
                    [self._next_top_lp, other._next_top_lp]
                )
            else:
                self._next_top_idx = None
                self._next_top_lp = None

        if self_was_empty:
            self._rope_deltas = other._rope_deltas
        elif (self._rope_deltas is None) != (other._rope_deltas is None):
            raise RuntimeError(
                "extend() mixes MRoPE and non-MRoPE batches; both sides must "
                "carry rope_deltas or neither side may."
            )
        elif self._rope_deltas is not None:
            self._rope_deltas = mx.concatenate([self._rope_deltas, other._rope_deltas])

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        if self.token_context:
            self.token_context = [self.token_context[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            self._rope_deltas = None
            self.token_context = []
            self.logits_processors = []
        else:
            keep_arr = mx.array(keep, mx.int32)
            for c in self.prompt_cache:
                c.filter(keep_arr)
            if self._next_tokens is not None:
                self._next_tokens = self._next_tokens[keep_arr]
            if self._next_lps is not None:
                self._next_lps = self._next_lps[keep_arr]
            if self._next_top_idx is not None:
                self._next_top_idx = self._next_top_idx[keep_arr]
                self._next_top_lp = self._next_top_lp[keep_arr]
            if self._rope_deltas is not None:
                self._rope_deltas = self._rope_deltas[keep_arr]

    def next(self) -> List[Response]:
        """Generate the next batch of tokens."""
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            top_lp = None
            if top_idx_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                )
            )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls, model, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ):
        """Create an empty generation batch."""
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch.token_context = []
        batch.logits_processors = []
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        return batch


class PromptProcessingBatch:
    """
    Handles VLM prompt processing with inputs_embeds and chunked prefill.

    Processes prompt tokens incrementally (one chunk per step) to allow
    interleaving with generation for continuous batching. Transitions to
    a GenerationBatch when prompt processing is complete.
    """

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        input_ids: List[List[int]],
        max_tokens: List[int],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        warm_cache: Optional[List[Any]] = None,
        apc_meta: Optional[List[dict]] = None,
        apc_manager: Optional["_apc.APCManager"] = None,
        right_pad_per_row: Optional[List[int]] = None,
        suffix_lens: Optional[List[int]] = None,
        apc_mode: Optional[str] = None,
    ):
        self.model = model
        self.uids = uids
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        # ``input_ids`` here are the per-row prefill inputs — for warm-start
        # rows this is the suffix, for cold rows the full prompt. When
        # ``right_pad_per_row`` is set the rows are right-padded (used in
        # mixed warm/cold prefill so suffix RoPE positions align). Otherwise
        # we left-pad as before.
        self._right_pad_per_row = right_pad_per_row
        self._suffix_lens = suffix_lens or lengths
        self._left_padding_per_row: List[int]

        if right_pad_per_row is not None:
            # Right-pad each row to max_length (so the last `pad[i]` cells are
            # right-pad and need to be rolled into left-pad by finalize()).
            left_padding = [0] * len(input_ids)
            self._input_ids = _right_pad_prompts(input_ids, max_length=max_length)
        else:
            left_padding = [max_length - l for l in lengths]
            self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._left_padding_per_row = list(left_padding)
        self._total_prompt_tokens = sum(lengths)
        self._processed_prompt_columns = 0

        self.logits_processors = logits_processors or []
        self._token_context = (
            [list(ids) for ids in input_ids]
            if self.logits_processors and any(self.logits_processors)
            else []
        )
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs or {}
        self._prompt_length_aware_keys: List[str] = []
        if self._prompt_kwargs and self._inputs_embeds is not None:
            prompt_batch = self._inputs_embeds.shape[0]
            prompt_len = self._inputs_embeds.shape[1]
            for k, v in self._prompt_kwargs.items():
                if (
                    isinstance(v, mx.array)
                    and v.ndim >= 2
                    and v.shape[0] == prompt_batch
                    and v.shape[1] == prompt_len
                ):
                    self._prompt_length_aware_keys.append(k)

        # APC metadata used for post-prefill block harvest (per-row).
        self._apc_meta = apc_meta or []
        self._apc_manager = apc_manager
        self._apc_mode = apc_mode
        self._apc_harvest_enabled = True

        if warm_cache is not None:
            self.prompt_cache = warm_cache
        else:
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
            )

        # Declare per-row right-padding on each cache so finalize() can roll
        # it into left-padding once the prefill forward pass is complete.
        if right_pad_per_row is not None and any(right_pad_per_row):
            for c in self.prompt_cache:
                prepare = getattr(c, "prepare", None)
                if not callable(prepare):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with prepare()"
                    )
                prepare(right_padding=right_pad_per_row, lengths=self._suffix_lens)

    def __len__(self):
        return len(self.uids)

    def _release_apc_meta_blocks(self):
        if self._apc_manager is None:
            return
        for meta in self._apc_meta:
            if meta is not None:
                self._apc_manager.release(meta.get("apc_blocks", []))

    def needs_processing(self):
        """True if prompt needs chunked processing before generate()."""
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return self._next_apc_checkpoint_column() is not None
        if self._next_apc_checkpoint_column() is not None:
            return True
        return self._inputs_embeds.shape[1] > self.prefill_step_size

    def _apc_checkpoint_column_for_meta(
        self, batch_idx: int, meta: dict
    ) -> Optional[int]:
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if (
            self._apc_mode != "exact"
            or checkpoint_len <= 0
            or meta.get("checkpoint_done")
        ):
            return None
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if checkpoint_len <= prefix_len:
            meta["checkpoint_done"] = True
            return None
        if self._right_pad_per_row is not None:
            suffix_checkpoint = checkpoint_len - prefix_len
            if suffix_checkpoint >= self._suffix_lens[batch_idx]:
                return None
            return suffix_checkpoint
        return self._left_padding_per_row[batch_idx] + checkpoint_len

    def _next_apc_checkpoint_column(self) -> Optional[int]:
        if (
            self._apc_manager is None
            or self._apc_mode != "exact"
            or not self._apc_meta
            or self._inputs_embeds is None
        ):
            return None
        start = self._processed_prompt_columns
        end = start + self._inputs_embeds.shape[1]
        next_col: Optional[int] = None
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            col = self._apc_checkpoint_column_for_meta(batch_idx, meta)
            if col is None or col <= start or col >= end:
                continue
            next_col = col if next_col is None else min(next_col, col)
        return next_col

    def _row_real_tokens_processed(self, batch_idx: int) -> int:
        meta = self._apc_meta[batch_idx]
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if self._right_pad_per_row is not None:
            suffix_done = min(
                self._suffix_lens[batch_idx],
                max(0, self._processed_prompt_columns),
            )
            return prefix_len + suffix_done
        real_done = (
            self._processed_prompt_columns - self._left_padding_per_row[batch_idx]
        )
        return prefix_len + min(self._suffix_lens[batch_idx], max(0, real_done))

    def _store_apc_exact_checkpoints(self) -> None:
        if self._apc_manager is None or self._apc_mode != "exact":
            return
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None or meta.get("checkpoint_done"):
                continue
            checkpoint_len = int(meta.get("checkpoint_len") or 0)
            if checkpoint_len <= 0:
                continue
            if self._row_real_tokens_processed(batch_idx) != checkpoint_len:
                continue
            prompt_cache = _apc.extract_prompt_cache_from_batch(
                self.prompt_cache,
                batch_idx,
            )
            if prompt_cache is None:
                continue
            self._apc_manager.store_exact_cache(
                meta["full_input_ids"][:checkpoint_len],
                prompt_cache,
                extra_hash=meta.get("extra_hash", 0),
            )
            meta["checkpoint_done"] = True

    def _prompt_kwargs_for_step(self, n: Optional[int] = None) -> dict:
        if n is None or not self._prompt_length_aware_keys:
            return self._prompt_kwargs
        out = dict(self._prompt_kwargs)
        for k in self._prompt_length_aware_keys:
            out[k] = out[k][:, :n, ...]
        return out

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        n = min(step, self._inputs_embeds.shape[1] - 1)
        checkpoint_col = self._next_apc_checkpoint_column()
        if checkpoint_col is not None:
            n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **prompt_kwargs,
        )
        mx.eval([c.state for c in self.prompt_cache])
        self._processed_prompt_columns += n
        self._store_apc_exact_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = self._prompt_kwargs[k][:, n:, ...]
        mx.clear_cache()
        return n

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **self._prompt_kwargs,
        )
        logits = output.logits if hasattr(output, "logits") else output
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            # Per-row last *real* token sits at index (seq - 1 - right_pad[i]).
            seq = logits.shape[1]
            last_idx = mx.array(
                [seq - 1 - p for p in self._right_pad_per_row], dtype=mx.int32
            )[:, None, None]
            last_idx = mx.broadcast_to(last_idx, (logits.shape[0], 1, logits.shape[-1]))
            logits = mx.take_along_axis(logits, last_idx, axis=1).squeeze(1)
        else:
            logits = logits[:, -1, :]
        if self.logits_processors and any(self.logits_processors):
            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    sample_logits = processor(
                        mx.array(self._token_context[i]), sample_logits
                    )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = sampler(logprobs)

        # Roll any right-padding into left-padding so the cache decoded by
        # GenerationBatch sees a canonical layout.
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            for c in self.prompt_cache:
                finalize = getattr(c, "finalize", None)
                if not callable(finalize):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with finalize()"
                    )
                finalize()
        if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
            c0 = self.prompt_cache[0] if self.prompt_cache else None
            if c0 is not None:
                off = getattr(c0, "offset", None)
                lp = getattr(c0, "left_padding", None)
                logger.warning(
                    "post-prefill cache[0]: _idx=%s offset=%s left_padding=%s right_pad_per_row=%s suffix_lens=%s",
                    getattr(c0, "_idx", None),
                    off.tolist() if hasattr(off, "tolist") else off,
                    lp.tolist() if hasattr(lp, "tolist") else lp,
                    self._right_pad_per_row,
                    self._suffix_lens,
                )

        gen_batch = GenerationBatch(
            model=self.model,
            uids=list(self.uids),
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=list(self.max_tokens),
            top_logprobs_k=top_logprobs_k,
            token_context=[list(ctx) for ctx in self._token_context],
            logits_processors=list(self.logits_processors),
        )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs:
            gen_batch._next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

        # Prime top-K buffers so the first token can emit top_logprobs too.
        if top_logprobs_k > 0:
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch._next_top_idx = top_idx
            gen_batch._next_top_lp = top_lp

        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = self._capture_rope_deltas(language_model, len(gen_batch.uids))
        if rope_deltas is not None:
            # Normalize to shape (B, 1) so extend/filter stay consistent.
            if rope_deltas.ndim == 0:
                rope_deltas = rope_deltas.reshape(1, 1)
            elif rope_deltas.ndim == 1:
                rope_deltas = rope_deltas[:, None]
            # When a warm-start batch reuses the model's cached _rope_deltas
            # (computed during a previous prefill with a smaller batch), the
            # batch dim won't match this prompt batch's row count. Realign
            # so extend()/filter() down the line stay consistent with the
            # generation batch's row count.
            target_b = first_tokens.shape[0]
            if rope_deltas.shape[0] != target_b:
                if rope_deltas.shape[0] == 1:
                    rope_deltas = mx.broadcast_to(
                        rope_deltas, (target_b, rope_deltas.shape[1])
                    )
                elif rope_deltas.shape[0] < target_b:
                    pad = target_b - rope_deltas.shape[0]
                    rope_deltas = mx.concatenate(
                        [
                            rope_deltas,
                            mx.broadcast_to(
                                rope_deltas[-1:],
                                (pad, rope_deltas.shape[1]),
                            ),
                        ],
                        axis=0,
                    )
                else:
                    rope_deltas = rope_deltas[:target_b]
            gen_batch._rope_deltas = rope_deltas

        # APC: harvest the post-prefill K/V into hashed blocks. Done after the
        # final prefill forward but before the cache references are released
        # so the block tensors snapshot the prompt prefix.
        if (
            self._apc_manager is not None
            and self._apc_meta
            and self._apc_harvest_enabled
        ):
            try:
                for batch_idx, meta in enumerate(self._apc_meta):
                    if meta is None:
                        continue
                    if self._apc_mode == "exact":
                        prompt_cache = _apc.extract_prompt_cache_from_batch(
                            self.prompt_cache,
                            batch_idx,
                        )
                        if prompt_cache is not None:
                            self._apc_manager.store_exact_cache(
                                meta["full_input_ids"],
                                prompt_cache,
                                extra_hash=meta.get("extra_hash", 0),
                            )
                        self._apc_manager.release(meta.get("apc_blocks", []))
                    else:
                        new_blocks = _apc.harvest_blocks_from_batch_cache(
                            self._apc_manager,
                            self.prompt_cache,
                            batch_idx,
                            meta["full_input_ids"],
                            extra_hash=meta.get("extra_hash", 0),
                            skip_first_n_tokens=meta.get("prefix_len", 0),
                        )
                        self._apc_manager.release(
                            meta.get("apc_blocks", []) + new_blocks
                        )
            except Exception as e:
                logger.warning("APC harvest failed during batched prefill: %s", e)
                # Best effort — release any acquired prefix blocks.
                for meta in self._apc_meta:
                    if meta is not None:
                        self._apc_manager.release(meta.get("apc_blocks", []))

        self.uids = []
        self.prompt_cache = []
        self._token_context = []
        self.logits_processors = []
        self._apc_meta = []
        return gen_batch

    @property
    def total_prompt_tokens(self):
        return self._total_prompt_tokens

    @staticmethod
    def _capture_rope_deltas(language_model, B: int):
        if not hasattr(language_model, "_rope_deltas"):
            return None
        rope_deltas = language_model._rope_deltas
        if rope_deltas is None:
            return mx.zeros((B, 1), dtype=mx.int32)
        if rope_deltas.ndim == 0:
            rope_deltas = rope_deltas.reshape(1, 1)
        elif rope_deltas.ndim == 1:
            rope_deltas = rope_deltas[:, None]
        # Falcon OCR emits a singleton meant to broadcast across rows.
        if rope_deltas.shape[0] == 1 and B > 1:
            rope_deltas = mx.broadcast_to(rope_deltas, (B, 1))
        if rope_deltas.shape[0] != B:
            raise RuntimeError(
                f"_rope_deltas shape {rope_deltas.shape} does not match prefill batch size {B}"
            )
        return rope_deltas


class BatchGenerator:
    """
    Continuous batching with separate prompt processing and generation phases.

    next() returns (prompt_responses, generation_responses) where:
    - prompt_responses is currently always [] (reserved for progress tracking)
    - generation_responses is a list of GenerationBatch.Response objects
    """

    def __init__(
        self,
        model,
        processor,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        prompt_cache=None,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        stream=None,
        apc_manager: Optional["_apc.APCManager"] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        # APC: opt-out for KV-quantized caches. Plain KV models use block APC;
        # mixed/custom cache models use exact prompt-cache snapshots.
        self.apc_mode = None
        if apc_manager is not None and kv_bits is not None:
            apc_manager = None
        if apc_manager is not None:
            self.apc_mode = _apc.model_apc_mode(model)
            if self.apc_mode is None:
                apc_manager = None
        self.apc_manager = apc_manager
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self._stream = stream or generation_stream

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
        )
        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [self._stream]))

    # ---------------- APC integration helpers ----------------
    # Keys that are APC-only metadata; stripped from ``prompt_kwargs`` before
    # the merged kwargs are passed to the language model forward.
    _APC_PRIVATE_KEYS = APC_PRIVATE_PROMPT_KEYS

    def _apc_extra_hash(self, prompt_kwargs: dict) -> int:
        """Salt for the APC hash chain."""
        if self.apc_manager is None:
            return 0
        if prompt_kwargs is None:
            prompt_kwargs = {}
        img = prompt_kwargs.get("_apc_image_hash")
        if img is None:
            pixel_values = prompt_kwargs.get("pixel_values")
            img = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=None)
        tenant = prompt_kwargs.get("_apc_tenant")
        return _apc.tenant_scoped_hash(tenant, img)

    def _apc_pick_for(self, sequence) -> Optional[dict]:
        """Look up an APC prefix for ``sequence``. Returns dict with matched
        blocks + suffix metadata when there is a usable hit, else None.
        """
        if self.apc_manager is None:
            return None
        uid, ids_list, max_toks, prompt_kwargs, lps = sequence
        if not ids_list or len(ids_list) < 2:
            return None
        # v1/v2: don't trim a prefix that contains image tokens — re-running
        # vision merging on the suffix is the cheap path here.
        image_token_id = getattr(self.model.config, "image_token_id", None) or getattr(
            self.model.config, "image_token_index", None
        )
        extra_hash = self._apc_extra_hash(prompt_kwargs or {})
        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            exact_cache, exact_prefix_len = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
            )
            if (
                exact_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < len(ids_list)
            ):
                if (
                    image_token_id is not None
                    and image_token_id in ids_list[:exact_prefix_len]
                ):
                    return None
                return {
                    "matched_blocks": [],
                    "warm_cache": exact_cache,
                    "prefix_len": exact_prefix_len,
                    "extra_hash": extra_hash,
                    "full_input_ids": list(ids_list),
                }
            return None
        matched, prefix_len = self.apc_manager.lookup_prefix(
            ids_list, extra_hash=extra_hash
        )
        exact_cache = None
        exact_prefix_len = 0
        if prefix_len < len(ids_list):
            exact_cache, exact_prefix_len = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=prefix_len,
            )
        warm_cache = None
        disk_prefix_len = 0
        if max(prefix_len, exact_prefix_len) < len(ids_list):
            warm_cache, disk_prefix_len = self.apc_manager.lookup_prefix_disk_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=max(prefix_len, exact_prefix_len),
                allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
            )
        if disk_prefix_len > max(
            prefix_len, exact_prefix_len
        ) and disk_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if (
                image_token_id is not None
                and image_token_id in ids_list[:disk_prefix_len]
            ):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": warm_cache,
                "prefix_len": disk_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if exact_prefix_len > prefix_len and exact_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if (
                image_token_id is not None
                and image_token_id in ids_list[:exact_prefix_len]
            ):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": exact_cache,
                "prefix_len": exact_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if prefix_len > 0 and prefix_len < len(ids_list):
            if image_token_id is not None and image_token_id in ids_list[:prefix_len]:
                self.apc_manager.release(matched)
                return None
            return {
                "matched_blocks": matched,
                "prefix_len": prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if matched:
            self.apc_manager.release(matched)
        return None

    def _build_mixed_prompt_batch(
        self, sequences: List[tuple]
    ) -> Optional["PromptProcessingBatch"]:
        """Build a multi-row PromptProcessingBatch admitting ``sequences``.

        Each row is independently looked up in APC. Warm rows have their
        suffixes prefilled against pre-populated K/V; cold rows prefill from
        scratch in the same batch. Right-padding aligns RoPE positions
        across rows with different prefix/suffix lengths.

        Returns ``None`` if APC is disabled (in which case the caller should
        use the cold-only path).
        """
        if self.apc_manager is None:
            return None

        picks: List[Optional[dict]] = [self._apc_pick_for(s) for s in sequences]
        any_warm = any(p is not None for p in picks)
        if not any_warm:
            return None  # caller falls back to cold-only path

        uids = [s[0] for s in sequences]
        full_ids = [list(s[1]) for s in sequences]
        max_tokens_list = [s[2] for s in sequences]
        prompt_kwargs_list = [s[3] for s in sequences]
        logits_processors = [s[4] for s in sequences]

        # Per-row prefix length and suffix tokens
        prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
        suffix_ids_list = [full_ids[i][prefix_lens[i] :] for i in range(len(sequences))]
        suffix_lens = [len(s) for s in suffix_ids_list]

        max_suffix_len = max(suffix_lens)
        right_pad_per_row = [max_suffix_len - s for s in suffix_lens]

        # Source inputs_embeds: every row's prompt_kwargs holds the full-prompt
        # embeddings. Slice to suffix per-row, right-pad to max_suffix_len, stack.
        suffix_embeds_per_row: List[mx.array] = []
        for i, kw in enumerate(prompt_kwargs_list):
            if kw is None or kw.get("inputs_embeds") is None:
                raise ValueError("APC mixed prefill requires precomputed inputs_embeds")
            full = kw["inputs_embeds"]  # [1, full_len, D]
            suff = full[:, prefix_lens[i] :, :]
            pad = right_pad_per_row[i]
            if pad > 0:
                pad_emb = mx.zeros(
                    (suff.shape[0], pad, suff.shape[-1]), dtype=suff.dtype
                )
                suff = mx.concatenate([suff, pad_emb], axis=1)
            suffix_embeds_per_row.append(suff)
        inputs_embeds = mx.concatenate(suffix_embeds_per_row, axis=0)

        # Merge prompt-side kwargs (excluding inputs_embeds, which we've just
        # rebuilt). Per-batch tensors get concatenated across rows; scalars
        # take the first row's value (matches the existing cold-only path).
        # APC-private keys (e.g. tenant salt) are dropped — they're consumed
        # in _apc_extra_hash, never forwarded to the model.
        merged_kwargs: dict = {}
        per_row_keys: dict = {}
        batch_size = len(prompt_kwargs_list)
        for i, kw in enumerate(prompt_kwargs_list):
            if not kw:
                continue
            full_len = len(full_ids[i])
            prefix_len = prefix_lens[i]
            right_pad = right_pad_per_row[i]
            for k, v in kw.items():
                if k == "inputs_embeds" or k in self._APC_PRIVATE_KEYS:
                    continue
                if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                    row_v = _prompt_kwarg_row(v, i, batch_size)
                    if _is_sequence_aligned_prompt_kwarg(k, row_v, full_len):
                        row_v = row_v[:, prefix_len:, ...]
                        row_v = _pad_sequence_aligned_prompt_kwarg(
                            row_v,
                            max_suffix_len,
                            left=False,
                        )
                    per_row_keys.setdefault(k, []).append(row_v)
                elif k not in merged_kwargs:
                    merged_kwargs[k] = v
        for k, vs in per_row_keys.items():
            merged_kwargs[k] = mx.concatenate(vs, axis=0)

        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            row_caches = [
                p["warm_cache"] if p is not None else self.model.make_cache()
                for p in picks
            ]
            warm_cache, _ = _apc.make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
            )
            if warm_cache is None:
                return None
        else:
            # Build the multi-row warm cache (zeros for cold rows, K/V for warm).
            num_layers = (
                len(self.model.make_cache())
                if hasattr(self.model, "make_cache")
                else len(self.model.layers)
            )
            warm_cache, _ = _apc.make_warm_batch_kv_cache_multi(
                picks, num_layers=num_layers
            )

        apc_meta = [
            {
                "full_input_ids": full_ids[i],
                "prefix_len": prefix_lens[i],
                "extra_hash": (
                    picks[i]["extra_hash"]
                    if picks[i]
                    else self._apc_extra_hash(prompt_kwargs_list[i] or {})
                ),
                "apc_blocks": picks[i].get("matched_blocks", []) if picks[i] else [],
                "checkpoint_len": (
                    max(
                        1,
                        len(full_ids[i]) - self.apc_manager.exact_cache_guard_tokens,
                    )
                    if apc_mode == "exact"
                    else 0
                ),
            }
            for i in range(len(sequences))
        ]

        return PromptProcessingBatch(
            model=self.model,
            uids=uids,
            input_ids=suffix_ids_list,
            max_tokens=max_tokens_list,
            inputs_embeds=inputs_embeds,
            prompt_kwargs=merged_kwargs,
            logits_processors=logits_processors,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            kv_quant_scheme=self.kv_quant_scheme,
            warm_cache=warm_cache,
            apc_meta=apc_meta,
            apc_manager=self.apc_manager,
            right_pad_per_row=right_pad_per_row,
            suffix_lens=suffix_lens,
            apc_mode=apc_mode,
        )

    def _build_apc_meta_for_cold(
        self,
        input_ids_list: List[List[int]],
        prompt_kwargs_list: List[Optional[dict]],
    ) -> Optional[List[Optional[dict]]]:
        """Build per-row harvest metadata for a cold-prefill batch so the
        produced K/V are added to APC after prefill.
        """
        if self.apc_manager is None:
            return None
        meta: List[Optional[dict]] = []
        for ids_list, kw in zip(input_ids_list, prompt_kwargs_list):
            extra_hash = self._apc_extra_hash(kw or {})
            meta.append(
                {
                    "full_input_ids": list(ids_list),
                    "prefix_len": 0,
                    "extra_hash": extra_hash,
                    "apc_blocks": [],
                    "checkpoint_len": (
                        max(
                            1,
                            len(ids_list) - self.apc_manager.exact_cache_guard_tokens,
                        )
                        if getattr(self, "apc_mode", "block") == "exact"
                        else 0
                    ),
                }
            )
        return meta

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompts,
        max_tokens: Union[List[int], int, None] = None,
        prompt_kwargs: Optional[List[dict]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
    ):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if logits_processors is None:
            logits_processors = [self.logits_processors] * len(prompts)
        elif len(logits_processors) != len(prompts):
            raise ValueError("Insufficient number of logits_processors provided")

        for p, m, kw, lp in zip(prompts, max_tokens, prompt_kwargs, logits_processors):
            self._unprocessed_sequences.append((self.uid_count, p, m, kw, lp))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def remove(self, uid) -> bool:
        """Remove a sequence from the batch by uid."""
        with mx.stream(self._stream):
            # Waiting in the queue.
            for i, (seq_uid, _, _, _, _) in enumerate(self._unprocessed_sequences):
                if seq_uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            # Being prefilled
            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding.
            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    @property
    def unprocessed_prompts(self):
        """Backward-compatible alias for server flush logic."""
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self):
        """True if there are prompts waiting or being processed."""
        return len(self._unprocessed_sequences) > 0 or self._prompt_batch is not None

    @property
    def has_work(self):
        """True if there is any remaining work."""
        return (
            len(self._generation_batch) > 0
            or self._prompt_batch is not None
            or len(self._unprocessed_sequences) > 0
        )

    def stats(self):
        """Return accumulated batch statistics."""
        stats = BatchStats()
        stats.prompt_tokens = self._prompt_tokens_counter
        stats.prompt_time = self._prompt_time_counter
        stats.prompt_tps = (
            self._prompt_tokens_counter / self._prompt_time_counter
            if self._prompt_time_counter > 0
            else 0
        )
        stats.generation_tokens = self._gen_tokens_counter
        stats.peak_memory = mx.get_peak_memory() / 1e9
        return stats

    def _next(self, **kwargs):
        generation_responses = []
        prompt_responses = []

        # Decode-first: always emit a generation step before touching prefill.
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
                self._prompt_tokens_counter += n
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            self._prompt_time_counter += time.perf_counter() - tic
            self._generation_batch.extend(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            # Take up to prefill_batch_size pending sequences. If APC is on
            # and at least one of them has a prefix hit, build a mixed
            # warm/cold PromptProcessingBatch with right-padded suffixes so
            # warm and cold rows prefill in a single forward pass.
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
                logger.warning(
                    "APC admit n=%d (pending=%d)",
                    n,
                    len(self._unprocessed_sequences),
                )
            mixed = self._build_mixed_prompt_batch(sequences)
            if mixed is not None:
                self._unprocessed_sequences = self._unprocessed_sequences[n:]
                self._prompt_batch = mixed
                self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens
                if self._prompt_batch.needs_processing():
                    tic = time.perf_counter()
                    nstep = self._prompt_batch.prompt_step()
                    self._prompt_time_counter += time.perf_counter() - tic
                else:
                    tic = time.perf_counter()
                    gen_batch = self._prompt_batch.generate(
                        self.sampler,
                        self.tokenizer.stopping_criteria,
                        compute_logprobs=self.compute_logprobs,
                        top_logprobs_k=self.top_logprobs_k,
                    )
                    self._prompt_time_counter += time.perf_counter() - tic
                    self._generation_batch.extend(gen_batch)
                    self._prompt_batch = None
                    mx.clear_cache()
                return prompt_responses, generation_responses

            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            prompt_kwargs_list = [s[3] for s in sequences]
            logits_processors = [s[4] for s in sequences]

            inputs_embeds, merged_kwargs = _merge_prefill_prompt_kwargs(
                prompt_kwargs_list, input_ids
            )

            # APC: also harvest cold-prefill prefixes so future requests hit.
            apc_meta = self._build_apc_meta_for_cold(input_ids, prompt_kwargs_list)

            self._prompt_batch = PromptProcessingBatch(
                model=self.model,
                uids=uids,
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                logits_processors=logits_processors,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                apc_meta=apc_meta,
                apc_manager=self.apc_manager,
                apc_mode=self.apc_mode,
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                self._prompt_time_counter += time.perf_counter() - tic
                self._generation_batch.extend(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses

    def next(self, **kwargs):
        with mx.stream(self._stream):
            return self._next(**kwargs)


def batch_generate(
    model,
    processor,
    images: Union[str, List[str]] = None,
    audios: Union[str, List[str]] = None,
    prompts: List[str] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs,
):
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       audios (Union[str, List[str]]): Audio files (not yet supported for batching).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from .utils import process_image

    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = _generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )

    processed_images = []
    image_sizes_original = []
    for img in images:
        if isinstance(img, str):
            pil_img = process_image(img, None, image_processor)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = img
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts = [None] * len(prompts)
    all_image_sizes = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in indices
                ]

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **group_kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response


def _clone_or_share_logits_processor(processor):
    if hasattr(processor, "clone"):
        return processor.clone()
    warnings.warn(
        "Sharing logits processor across batch entries because it does not "
        "implement clone(). Stateful logits processors should implement clone() "
        "to avoid shared state across sequences.",
        RuntimeWarning,
        stacklevel=2,
    )
    return processor


def _generate_batch(
    model,
    processor,
    prompts: List[str],
    images: List = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[str], BatchStats]:

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)
    logits_processors = kwargs.pop("logits_processors", None)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            p,
            num_images=num_images_list[i],
        )
        for i, p in enumerate(prompts)
    ]

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    if getattr(model, "no_chunked_prefill", False):
        kwargs.pop("prefill_step_size", None)
        kwargs["prefill_step_size"] = None

    # Use batch_size for prefill and completion to ensure consistent processing
    gen = BatchGenerator(
        model.language_model,
        processor,
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
        compute_logprobs=False,
        **kwargs,
    )

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **data_kwargs
    )

    gen_kwargs = {**data_kwargs, **embedding_output.to_dict()}

    if logits_processors and all(
        callable(processor) for processor in logits_processors
    ):
        logits_processors = [
            [_clone_or_share_logits_processor(p) for p in logits_processors]
            for _ in range(batch_size)
        ]

    uids = gen.insert(
        input_ids.tolist(),
        max_tokens,
        prompt_kwargs=_split_prompt_kwargs_per_row(gen_kwargs, batch_size),
        logits_processors=logits_processors,
    )
    results = {uid: [] for uid in uids}

    tic = time.perf_counter()
    while gen.has_work:
        _, generation_responses = gen.next()
        for r in generation_responses:
            if r.finish_reason != "stop":
                results[r.uid].append(r.token)
    total_time = time.perf_counter() - tic

    gen.close()

    detokenizer = processor.detokenizer
    texts = []
    for uid in uids:
        detokenizer.reset()
        for t in results[uid]:
            detokenizer.add_token(t)
        detokenizer.finalize()
        texts.append(detokenizer.text)

    stats = gen.stats()
    stats.generation_time = total_time - stats.prompt_time
    if stats.generation_time > 0:
        stats.generation_tps = stats.generation_tokens / stats.generation_time
    return texts, stats


def main():
    args = parse_arguments()
    if isinstance(args.image, str):
        args.image = [args.image]
    if isinstance(args.audio, str):
        args.audio = [args.audio]
    if isinstance(args.video, str):
        args.video = [args.video]

    model, processor = load(
        args.model,
        args.adapter_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
    )
    config = model.config

    draft_model = None
    if args.draft_model is not None:
        from .speculative.drafters import load_drafter

        print(f"Loading drafter ({args.draft_kind or 'auto'}): {args.draft_model}")
        draft_model, resolved_kind = load_drafter(
            args.draft_model, kind=args.draft_kind
        )
        if args.draft_kind is None:
            print(f"  → auto-detected --draft-kind={resolved_kind!r}.")
        elif resolved_kind != args.draft_kind:
            print(
                f"  → drafter requires --draft-kind={resolved_kind!r}; "
                f"using {resolved_kind!r} instead of {args.draft_kind!r}."
            )
        args.draft_kind = resolved_kind

    prompt = args.prompt

    num_images = len(args.image) if args.image is not None else 0
    num_audios = len(args.audio) if args.audio is not None else 0

    chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    if args.video:
        chat_template_kwargs["video"] = args.video
        chat_template_kwargs["fps"] = args.fps

    prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=num_images,
        num_audios=num_audios,
        **chat_template_kwargs,
    )

    kwargs = {}

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Add processor kwargs from JSON
    if args.processor_kwargs:
        kwargs.update(args.processor_kwargs)

    # Add thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    if args.chat:
        from .vision_cache import VisionFeatureCache

        vision_cache = VisionFeatureCache()
        chat = []
        if args.system:
            chat.append({"role": "system", "content": args.system})
        while user := input("User:"):
            chat.append({"role": "user", "content": user})
            prompt = apply_chat_template(
                processor,
                config,
                chat,
                num_images=num_images,
                num_audios=num_audios,
                **chat_template_kwargs,
            )
            response = ""
            print("Assistant:", end="")
            stream_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "vision_cache": vision_cache,
                **kwargs,
            }
            if args.resize_shape is not None:
                stream_kwargs["resize_shape"] = args.resize_shape
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size

            for chunk in stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                **stream_kwargs,
            ):
                response += chunk.text
                print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            print()

    else:
        gen_kwargs = {
            "image": args.image,
            "audio": args.audio,
            "video": args.video,
            "fps": args.fps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "verbose": args.verbose,
            "max_kv_size": args.max_kv_size,
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "kv_quant_scheme": getattr(
                args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME
            ),
            "quantized_kv_start": args.quantized_kv_start,
            **kwargs,
        }
        if args.resize_shape is not None:
            gen_kwargs["resize_shape"] = args.resize_shape
        if args.prefill_step_size is not None:
            gen_kwargs["prefill_step_size"] = args.prefill_step_size
        if draft_model is not None:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["draft_kind"] = args.draft_kind
            if args.draft_block_size is not None:
                gen_kwargs["draft_block_size"] = args.draft_block_size

        result = generate(
            model,
            processor,
            prompt,
            **gen_kwargs,
        )
        if not args.verbose:
            print(result.text)

        if draft_model is not None:
            lens = getattr(draft_model, "accept_lens", None) or []
            if lens:
                mean_accept = round(sum(lens) / len(lens), 2)
                print(
                    f"Speculative decoding: {mean_accept} accepted tokens over {len(lens)} rounds"
                )


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.generate ...` directly is deprecated."
        " Use `mlx_vlm generate` or `python -m mlx_vlm generate` instead."
    )
    main()
