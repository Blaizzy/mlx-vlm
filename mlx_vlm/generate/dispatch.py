import argparse
import codecs
import contextlib
import inspect
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache as mlx_maybe_quantize_kv_cache
from transformers import PreTrainedTokenizer

from .. import apc as _apc
from ..models import cache
from ..prompt_utils import apply_chat_template, thinking_template_kwargs
from ..speculative.utils import format_speculative_stats
from ..tokenizer_utils import make_streaming_detokenizer
from ..turboquant import TurboQuantKVCache, turboquant_enabled
from ..utils import StoppingCriteria, ThinkingBudgetCriteria, load, prepare_inputs
from .image import (
    DEFAULT_IMAGE_GUIDANCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STEPS,
    DEFAULT_IMAGE_TASK,
    run_image_generation_cli,
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
DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH = 64
DEFAULT_DIFFUSION_MAX_DENOISING_STEPS = 48


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
        "--output-modality",
        type=str,
        choices=("text", "image"),
        default="text",
        help="Generate text with a VLM or generate an image with a supported image model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for image generation.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("generate", "edit"),
        default=DEFAULT_IMAGE_TASK,
        help="Image task to run when --output-modality image is selected.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help=(
            "Image size as WIDTHxHEIGHT. Generation defaults to "
            f"{DEFAULT_IMAGE_SIZE}; editing defaults to the first reference image size."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_IMAGE_STEPS,
        help="Number of image inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "PRNG seed for reproducible sampling and diffusion canvas init. "
            "Image generation/editing defaults to a random 32-bit seed."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_IMAGE_GUIDANCE,
        help="Classifier-free guidance for image generation/editing.",
    )
    parser.add_argument(
        "--prompt-expansion-model",
        type=str,
        default=None,
        help=(
            "Text model path or Hugging Face repo used to expand plain image "
            "prompts into Ideogram 4 JSON captions."
        ),
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
        "--max-long-side-pixel",
        type=_parse_positive_int,
        default=None,
        help="Maximum long-side pixels for MiniMax M3 image/video preprocessing.",
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
        "--max-denoising-steps",
        type=int,
        default=None,
        help=(
            "Maximum denoising steps for diffusion generation. "
            "Default: the checkpoint's generation config (typically "
            f"{DEFAULT_DIFFUSION_MAX_DENOISING_STEPS}). Adaptive stopping "
            "usually converges canvases earlier; set lower to hard-cap "
            "throughput."
        ),
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=None,
        help="Block length for diffusion text generation.",
    )
    parser.add_argument(
        "--num-to-transfer",
        type=int,
        default=None,
        help="Target number of masked tokens to transfer per diffusion denoising step.",
    )
    parser.add_argument(
        "--max-transfer-per-step",
        type=int,
        default=None,
        help="Maximum confident masked tokens to transfer per denoising step.",
    )
    parser.add_argument(
        "--editing-threshold",
        type=float,
        default=None,
        help="Confidence threshold for diffusion post-fill token edits.",
    )
    parser.add_argument(
        "--max-post-steps",
        type=int,
        default=None,
        help="Maximum diffusion post-fill editing steps per block.",
    )
    parser.add_argument(
        "--stability-steps",
        type=int,
        default=None,
        help="Stop post-fill refinement after this many stable no-edit steps.",
    )
    parser.add_argument(
        "--diffusion-full-canvas",
        action="store_true",
        help=(
            "Use the checkpoint canvas length for diffusion generation even when "
            "--max-tokens requests a partial block."
        ),
    )
    parser.add_argument(
        "--diffusion-min-canvas-length",
        type=int,
        default=None,
        help=(
            "Minimum active canvas length for diffusion partial blocks. "
            f"Default: {DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH}."
        ),
    )
    parser.add_argument(
        "--diffusion-max-canvas-length",
        type=int,
        default=None,
        help=(
            "Maximum active canvas length for diffusion generation. Default: the "
            "checkpoint canvas length; set lower to trade quality for "
            "throughput."
        ),
    )
    parser.add_argument(
        "--diffusion-sampler",
        choices=["entropy-bound", "confidence-threshold"],
        default="entropy-bound",
        help="Canvas update sampler for diffusion generation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Token probability threshold for diffusion confidence transfer. "
            "Default: 0.9 for confidence-threshold sampling; masked-diffusion "
            "models use their checkpoint reference defaults."
        ),
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=None,
        help="Lowest token probability threshold for masked diffusion transfer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty factor for previously generated tokens.",
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for repetition penalty.",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Additive penalty for tokens that already appeared.",
    )
    parser.add_argument(
        "--presence-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for presence penalty.",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Additive penalty scaled by token frequency.",
    )
    parser.add_argument(
        "--frequency-context-size",
        type=int,
        default=DEFAULT_REPETITION_CONTEXT_SIZE,
        help="Number of recent generated tokens used for frequency penalty.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detailed output (use --no-verbose to print only the final result).",
    )
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
        "--lazy-load",
        action="store_true",
        help=(
            "Skip eager parameter materialization during model load. This can "
            "reduce startup memory for very large checkpoints, but the first "
            "generation step will still need to materialize the weights it uses."
        ),
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
        "--gen-kwargs",
        type=json.loads,
        default={},
        help="Extra generation kwargs as JSON. "
        "Example: --gen-kwargs '{\"custom_arg\": true}'",
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
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, Inferact/MiniMax-M3-EAGLE3)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family. Supported: 'dflash' (Qwen3.5 DFlash), "
        "'eagle3' (Speculators/SGLang EAGLE-3), "
        "'mtp' (native/assistant Multi-Token Prediction). "
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
        "--thinking-mode",
        type=str,
        default=None,
        choices=["enabled", "disabled", "adaptive"],
        help="Model chat template thinking mode, when supported.",
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


def normalize_max_long_side_pixel(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_long_side_pixel must be a positive integer")
    return value


def _parse_positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "max_long_side_pixel must be a positive integer"
        ) from exc
    try:
        return normalize_max_long_side_pixel(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
    cached_tokens: int = 0
    # Populated only on the terminal chunk yielded by ``stream_generate``:
    # ``"stop"`` for eos/stop-sequence, ``"length"`` for max_tokens.
    finish_reason: Optional[str] = None


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


def _cache_offset_value(c) -> Optional[int]:
    offset = getattr(c, "offset", None)
    if offset is None:
        return None
    try:
        return int(offset.item() if hasattr(offset, "item") else offset)
    except (TypeError, ValueError):
        return None


def _trim_prompt_cache_entry_to(c, prefix_len: int):
    if hasattr(c, "caches"):
        for sub_cache in c.caches:
            _trim_prompt_cache_entry_to(sub_cache, prefix_len)
        return
    if isinstance(c, tuple):
        for sub_cache in c:
            _trim_prompt_cache_entry_to(sub_cache, prefix_len)
        return

    offset = _cache_offset_value(c)
    if offset is None or offset <= prefix_len:
        return

    trim = getattr(c, "trim", None)
    if callable(trim):
        trim(offset - prefix_len)
        return

    keys = getattr(c, "keys", None)
    values = getattr(c, "values", None)
    if keys is not None and values is not None:
        c.keys = keys[..., :prefix_len, :]
        c.values = values[..., :prefix_len, :]
        if hasattr(c, "offset"):
            c.offset = prefix_len


def _config_token_id(config, name: str):
    return getattr(config, f"{name}_token_id", None) or getattr(
        config, f"{name}_token_index", None
    )


def _drop_unused_multimodal_inputs(model, new_ids: list, pixel_values, kwargs: dict):
    config = getattr(model, "config", None)
    image_token_id = _config_token_id(config, "image")
    video_token_id = _config_token_id(config, "video")
    has_image = image_token_id is not None and image_token_id in new_ids
    has_video = video_token_id is not None and video_token_id in new_ids

    if not has_image:
        kwargs.pop("cached_image_features", None)
        if not has_video or "pixel_values_videos" in kwargs:
            pixel_values = None

    if video_token_id is not None and not has_video:
        kwargs.pop("pixel_values_videos", None)
        kwargs.pop("video_grid_thw", None)
        kwargs.pop("cached_video_features", None)

    return pixel_values


def _encode_image_for_vision_cache(model, pixel_values, kwargs: dict):
    encode_image = getattr(model, "encode_image", None)
    if not callable(encode_image):
        return None

    encode_kwargs = {}
    try:
        parameters = inspect.signature(encode_image).parameters
    except (TypeError, ValueError):
        return encode_image(pixel_values)

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    for name in ("image_grid_thw", "image_position_ids"):
        if name in kwargs and (accepts_kwargs or name in parameters):
            encode_kwargs[name] = kwargs[name]

    return encode_image(pixel_values, **encode_kwargs)


def _encode_video_for_vision_cache(model, pixel_values, kwargs: dict):
    encode_video = getattr(model, "encode_video", None)
    if not callable(encode_video):
        return None

    encode_kwargs = {}
    try:
        parameters = inspect.signature(encode_video).parameters
    except (TypeError, ValueError):
        return encode_video(pixel_values)

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if "video_grid_thw" in kwargs and (
        accepts_kwargs or "video_grid_thw" in parameters
    ):
        encode_kwargs["video_grid_thw"] = kwargs["video_grid_thw"]

    return encode_video(pixel_values, **encode_kwargs)


def _video_cache_key(video):
    if isinstance(video, list):
        return ["video", *video]
    return f"video:{video}"


from .common import GenerationResult, generation_stream, wired_limit
from .diffusion import (
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    DiffusionOutputHandler,
    diffusion_kwargs_from_args,
    is_diffusion_model,
    is_masked_diffusion_model,
    stream_diffusion_generate_from_kwargs,
)


def is_masked_diffusion_text_model(model: nn.Module) -> bool:
    return is_masked_diffusion_model(model)


def _use_masked_diffusion_text_path(model: nn.Module, kwargs: Dict[str, Any]) -> bool:
    if not is_masked_diffusion_text_model(model):
        return False

    config = getattr(model, "config", None)
    if getattr(config, "default_generation_mode", None) != "ar":
        return True

    generation_mode = kwargs.get("generation_mode")
    if generation_mode is not None:
        return generation_mode != "ar"

    return False


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


from .ar import generate_step


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
    verbose = kwargs.pop("verbose", False)

    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", DEFAULT_THINKING_END_TOKEN)
    thinking_start_token = kwargs.pop(
        "thinking_start_token", DEFAULT_THINKING_START_TOKEN
    )
    enable_thinking = kwargs.pop("enable_thinking", False)
    if model.config.model_type in {"minimax_m3", "minimax_m3_vl"}:
        if thinking_start_token == DEFAULT_THINKING_START_TOKEN:
            thinking_start_token = "<mm:think>"
        if thinking_end_token == DEFAULT_THINKING_END_TOKEN:
            thinking_end_token = "</mm:think>"

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4", "gemma4_unified"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    max_long_side_pixel = normalize_max_long_side_pixel(
        kwargs.pop("max_long_side_pixel", None)
    )
    image_token_index = getattr(model.config, "image_token_index", None)
    vision_cache = kwargs.pop("vision_cache", None)
    prompt_cache_state = kwargs.pop("prompt_cache_state", None)
    apc_manager: Optional[_apc.APCManager] = kwargs.pop("apc_manager", None)
    apc_tenant: Optional[str] = kwargs.pop("apc_tenant", None)
    image = image or None
    audio = audio or None
    video = video or None

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        processor_kwargs = {}
        if max_long_side_pixel is not None:
            processor_kwargs["max_long_side_pixel"] = max_long_side_pixel
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            videos=video,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **processor_kwargs,
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

    if _use_masked_diffusion_text_path(model, kwargs):
        if image is not None or audio is not None or video is not None:
            raise ValueError("Diffusion text generation models are text-only.")

        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)
        top_p = kwargs.get("top_p", DEFAULT_TOP_P)
        top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        max_denoising_steps = kwargs.get("max_denoising_steps")
        if max_denoising_steps is None:
            config = getattr(model, "config", None)
            max_denoising_steps = kwargs.get(
                "steps", getattr(config, "default_diffusion_steps", 32)
            )
        config = getattr(model, "config", None)
        # Sampler knobs resolve as: explicit kwarg > config default_diffusion_*
        # attribute > the model generate()'s own reference defaults (omitted
        # here). Forcing shared defaults broke checkpoints whose reference
        # generation differs (e.g. LLaDA2.0 corrupts with editing enabled).
        tuned_kwargs = {}
        for key, config_attr in (
            ("threshold", "default_diffusion_threshold"),
            ("min_threshold", "default_diffusion_min_threshold"),
            ("editing_threshold", "default_diffusion_editing_threshold"),
            ("num_to_transfer", "default_diffusion_num_to_transfer"),
            ("max_transfer_per_step", "default_diffusion_max_transfer_per_step"),
            ("max_post_steps", "default_diffusion_max_post_steps"),
            ("stability_steps", "default_diffusion_stability_steps"),
        ):
            value = kwargs.get(key)
            if value is None:
                value = getattr(config, config_attr, None)
            if value is not None:
                tuned_kwargs[key] = value

        generation_stats = {}
        handled_generation_kwargs = {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "max_denoising_steps",
            "steps",
            "block_length",
            "threshold",
            "min_threshold",
            "editing_threshold",
            "max_post_steps",
            "num_to_transfer",
            "max_transfer_per_step",
            "stability_steps",
        }
        model_generate_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in handled_generation_kwargs
        }
        tic = time.perf_counter()
        generated = model.language_model.generate(
            input_ids,
            temperature=temperature,
            block_length=kwargs.get("block_length", 32),
            steps=max_denoising_steps,
            gen_length=max_tokens,
            top_p=None if top_p is None or top_p >= 1.0 else top_p,
            top_k=None if top_k is None or top_k <= 0 else top_k,
            eos_early_stop=True,
            visualize=verbose,
            tokenizer=tokenizer,
            skip_special_tokens=skip_special_tokens,
            stats=generation_stats,
            **tuned_kwargs,
            **model_generate_kwargs,
        )
        mx.eval(generated)
        total_time = time.perf_counter() - tic
        prompt_time = generation_stats.get("prompt_time", 0.0)
        prompt_tps = input_ids.size / prompt_time if prompt_time > 0 else 0.0
        generation_time = max(total_time - prompt_time, 1e-9)
        generated_tokens = generated[0].tolist()
        text = tokenizer.decode(
            generated_tokens, skip_special_tokens=skip_special_tokens
        )

        yield GenerationResult(
            text=text,
            token=generated_tokens[-1] if generated_tokens else None,
            logprobs=None,
            prompt_tokens=input_ids.size,
            generation_tokens=len(generated_tokens),
            total_tokens=input_ids.size + len(generated_tokens),
            prompt_tps=prompt_tps,
            generation_tps=len(generated_tokens) / generation_time,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=(
                "stop"
                if generated_tokens
                and tokenizer.stopping_criteria(generated_tokens[-1])
                else "length"
            ),
            text_already_printed=bool(generation_stats.get("text_already_printed")),
        )
        return

    # Vision feature caching: reuse cached image features across turns
    if vision_cache is not None and image is not None and pixel_values is not None:
        cached = vision_cache.get(image)
        if cached is not None:
            kwargs["cached_image_features"] = cached
        else:
            features = _encode_image_for_vision_cache(model, pixel_values, kwargs)
            if features is not None:
                mx.eval(features)
                vision_cache.put(image, features)
                kwargs["cached_image_features"] = features

    if (
        vision_cache is not None
        and video is not None
        and kwargs.get("pixel_values_videos", None) is not None
    ):
        cache_key = _video_cache_key(video)
        cached = vision_cache.get(cache_key)
        if cached is not None:
            kwargs["cached_video_features"] = cached
        else:
            features = _encode_video_for_vision_cache(
                model,
                kwargs["pixel_values_videos"],
                kwargs,
            )
            if features is not None:
                mx.eval(features)
                vision_cache.put(cache_key, features)
                kwargs["cached_video_features"] = features

    # Prompt cache reuse: skip common prefix from previous turn
    reused_prefix_len = 0
    full_input_ids_list = input_ids.flatten().tolist()
    apc_blocks_in_use: List[_apc.APCBlock] = []
    apc_extra_hash = 0
    apc_mode: Optional[str] = None

    multimodal_token_ids = _apc.multimodal_token_ids_from_config(model.config)
    apc_safe_prefix_min = _apc.media_safe_prefix_min(
        full_input_ids_list,
        multimodal_token_ids,
    )
    apc_safe_prefix_lookup_min = max(0, apc_safe_prefix_min - 1)

    def _apc_suffix_is_text_only(prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            full_input_ids_list,
            prefix_len,
            multimodal_token_ids,
        )

    def _apc_prefix_has_media_tokens(prefix_len: int) -> bool:
        return _apc.prefix_contains_media_tokens(
            full_input_ids_list,
            prefix_len,
            multimodal_token_ids,
        )

    if is_diffusion_model(model):
        yield from stream_diffusion_generate_from_kwargs(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            mask,
            skip_special_token_ids,
            kwargs,
        )
        return

    if apc_manager is not None:
        apc_mode = _apc.model_apc_mode(model.language_model)
        if apc_mode is None:
            apc_manager = None

    if apc_manager is not None:
        image_hash = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=image)
        video_hash = _apc.hash_video_payload(
            pixel_values=kwargs.get("pixel_values_videos"),
            video_grid_thw=kwargs.get("video_grid_thw"),
            video_ref=video,
        )
        payload_hash = _apc.hash_multimodal_payload(image_hash, video_hash)
        apc_extra_hash = _apc.tenant_scoped_hash(apc_tenant, payload_hash)

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)
        if prefix_len > 0 and prefix_len < input_ids.shape[1]:
            if _apc_suffix_is_text_only(prefix_len) and _prime_cached_prefix_rope_state(
                model, input_ids, mask, kwargs
            ):
                reused_prefix_len = prefix_len
                # Trim to only new tokens
                input_ids = input_ids[:, prefix_len:]
                new_ids = input_ids.flatten().tolist()
                pixel_values = _drop_unused_multimodal_inputs(
                    model, new_ids, pixel_values, kwargs
                )
                # Reuse the saved KV cache (trimmed to prefix length)
                kv_cache = prompt_cache_state.cache
                # Trim cache to prefix_len in case it includes generated tokens
                for c in kv_cache:
                    _trim_prompt_cache_entry_to(c, prefix_len)
                kwargs["prompt_cache"] = kv_cache

    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    if apc_manager is not None and reused_prefix_len == 0:
        if apc_mode == "exact":
            exact_prompt_cache, exact_prefix_len = apc_manager.lookup_exact_cache(
                full_input_ids_list,
                extra_hash=apc_extra_hash,
                min_prefix_tokens=apc_safe_prefix_lookup_min,
            )
            if (
                exact_prompt_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < input_ids.shape[1]
                and _apc_suffix_is_text_only(exact_prefix_len)
                and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
            ):
                reused_prefix_len = exact_prefix_len
                input_ids = input_ids[:, exact_prefix_len:]
                new_ids = input_ids.flatten().tolist()
                pixel_values = _drop_unused_multimodal_inputs(
                    model, new_ids, pixel_values, kwargs
                )
                kwargs["prompt_cache"] = exact_prompt_cache
        else:
            matched_blocks, prefix_len = apc_manager.lookup_prefix(
                full_input_ids_list, extra_hash=apc_extra_hash
            )
            if prefix_len > 0 and _apc_prefix_has_media_tokens(prefix_len):
                apc_manager.release(matched_blocks)
                matched_blocks = []
                prefix_len = 0
            exact_prompt_cache = None
            exact_prefix_len = 0
            if prefix_len < input_ids.shape[1]:
                exact_prompt_cache, exact_prefix_len = apc_manager.lookup_exact_cache(
                    full_input_ids_list,
                    extra_hash=apc_extra_hash,
                    min_prefix_tokens=max(prefix_len, apc_safe_prefix_lookup_min),
                )
            disk_prompt_cache = None
            disk_prefix_len = 0
            if max(prefix_len, exact_prefix_len) < input_ids.shape[1]:
                disk_prompt_cache, disk_prefix_len = (
                    apc_manager.lookup_prefix_disk_cache(
                        full_input_ids_list,
                        extra_hash=apc_extra_hash,
                        min_prefix_tokens=max(
                            prefix_len,
                            exact_prefix_len,
                            apc_safe_prefix_lookup_min,
                        ),
                        allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
                    )
                )
            if (
                disk_prefix_len > max(prefix_len, exact_prefix_len)
                and disk_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _apc_suffix_is_text_only(
                    disk_prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = disk_prefix_len
                    input_ids = input_ids[:, disk_prefix_len:]
                    new_ids = input_ids.flatten().tolist()
                    pixel_values = _drop_unused_multimodal_inputs(
                        model, new_ids, pixel_values, kwargs
                    )
                    kwargs["prompt_cache"] = disk_prompt_cache
            elif (
                exact_prefix_len > prefix_len and exact_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _apc_suffix_is_text_only(
                    exact_prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = exact_prefix_len
                    input_ids = input_ids[:, exact_prefix_len:]
                    new_ids = input_ids.flatten().tolist()
                    pixel_values = _drop_unused_multimodal_inputs(
                        model, new_ids, pixel_values, kwargs
                    )
                    kwargs["prompt_cache"] = exact_prompt_cache
            elif prefix_len > 0 and prefix_len < input_ids.shape[1]:
                if _apc_suffix_is_text_only(
                    prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    apc_blocks_in_use = matched_blocks
                    reused_prefix_len = prefix_len
                    input_ids = input_ids[:, prefix_len:]
                    new_ids = input_ids.flatten().tolist()
                    pixel_values = _drop_unused_multimodal_inputs(
                        model, new_ids, pixel_values, kwargs
                    )
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
            exact_checkpoint_len = _apc.adjust_prefix_to_text_suffix_boundary(
                full_input_ids_list,
                len(full_input_ids_list) - apc_manager.exact_cache_guard_tokens,
                multimodal_token_ids,
                max_prefix_tokens=len(full_input_ids_list) - 1,
            )
            if exact_checkpoint_len <= 0:
                exact_checkpoint_len = None

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
            verbose=verbose,
            **kwargs,
        )
        tic = time.perf_counter()

        generated_tokens = []
        finish_reason: Optional[str] = None
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
                finish_reason = "stop"
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
                cached_tokens=reused_prefix_len,
            )
        else:
            # generate_step exhausted its budget without stopping_criteria firing.
            finish_reason = "length"

        if not generated_tokens:
            prompt_time = time.perf_counter() - tic
            prompt_tps = total_prompt_tokens / prompt_time if prompt_time > 0 else 0.0
            yield GenerationResult(
                text="",
                token=None,
                logprobs=None,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=0,
                total_tokens=total_prompt_tokens,
                prompt_tps=prompt_tps,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                cached_tokens=reused_prefix_len,
                finish_reason="length",
            )
            return

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
            cached_tokens=reused_prefix_len,
            finish_reason=finish_reason,
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
    diffusion_output = DiffusionOutputHandler(model, kwargs, verbose)

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
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
    ):
        if response.is_draft:
            diffusion_output.handle_draft(response)
            last_response = response
            continue

        if (
            verbose
            and not response.text_already_printed
            and not diffusion_output.handle_text(response.text)
        ):
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if last_response is None:
        return GenerationResult(text=text, peak_memory=mx.get_peak_memory() / 1e9)

    if verbose:
        diffusion_output.finish(text)
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
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
        cached_tokens=last_response.cached_tokens,
        finish_reason=last_response.finish_reason,
        diffusion_canvas_tokens=last_response.diffusion_canvas_tokens,
        diffusion_denoising_steps=last_response.diffusion_denoising_steps,
        diffusion_work_tokens=last_response.diffusion_work_tokens,
        diffusion_canvas_tps=last_response.diffusion_canvas_tps,
        diffusion_work_tps=last_response.diffusion_work_tps,
    )


def main():
    args = parse_arguments()

    if getattr(args, "output_modality", "text") == "image":
        run_image_generation_cli(args)
        return

    if getattr(args, "seed", None) is not None:
        mx.random.seed(args.seed)

    diffusion_arg_defaults = {
        "max_denoising_steps": None,
        "diffusion_full_canvas": False,
        "diffusion_min_canvas_length": None,
        "diffusion_max_canvas_length": None,
        "diffusion_sampler": "entropy-bound",
        "threshold": None,
        "min_threshold": None,
        "block_length": None,
        "num_to_transfer": None,
        "max_transfer_per_step": None,
        "editing_threshold": None,
        "max_post_steps": None,
        "stability_steps": None,
        "gen_kwargs": {},
    }
    for name, default in diffusion_arg_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)

    if isinstance(args.image, str):
        args.image = [args.image]
    if isinstance(args.audio, str):
        args.audio = [args.audio]
    if isinstance(args.video, str):
        args.video = [args.video]

    model, processor = load(
        args.model,
        args.adapter_path,
        lazy=args.lazy_load,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
    )
    config = model.config

    draft_model = None
    if args.draft_model is not None:
        from ..speculative.drafters import load_drafter, validate_drafter_compatibility

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
        try:
            validate_drafter_compatibility(model, draft_model, args.draft_kind)
        except ValueError as e:
            print(
                "Speculative drafter is incompatible with the target model; "
                f"falling back to autoregressive generation. {e}"
            )
            draft_model = None
            args.draft_kind = None

    prompt = args.prompt

    if args.system:
        prompt = [{"role": "system", "content": args.system}] + (
            prompt if isinstance(prompt, list) else [prompt]
        )

    num_images = len(args.image) if args.image is not None else 0
    num_audios = len(args.audio) if args.audio is not None else 0

    thinking_mode = getattr(args, "thinking_mode", None)
    chat_template_kwargs = thinking_template_kwargs(
        config,
        enable_thinking=args.enable_thinking,
        thinking_mode=thinking_mode,
    )
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

    # Add generation kwargs from JSON
    if args.gen_kwargs:
        kwargs.update(args.gen_kwargs)

    # Add thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if thinking_mode == "enabled":
        kwargs["enable_thinking"] = True
    elif thinking_mode == "disabled":
        kwargs["enable_thinking"] = False
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        thinking_start_token = args.thinking_start_token
        thinking_end_token = args.thinking_end_token
        if config.model_type in {"minimax_m3", "minimax_m3_vl"}:
            if thinking_start_token == DEFAULT_THINKING_START_TOKEN:
                thinking_start_token = "<mm:think>"
            if thinking_end_token == DEFAULT_THINKING_END_TOKEN:
                thinking_end_token = "</mm:think>"
        kwargs["thinking_end_token"] = thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = thinking_start_token

    if args.chat:
        from ..vision_cache import VisionFeatureCache

        vision_cache = VisionFeatureCache()
        is_masked_text_diffusion = is_masked_diffusion_text_model(model)
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
                "repetition_penalty": args.repetition_penalty,
                "repetition_context_size": args.repetition_context_size,
                "presence_penalty": args.presence_penalty,
                "presence_context_size": args.presence_context_size,
                "frequency_penalty": args.frequency_penalty,
                "frequency_context_size": args.frequency_context_size,
                "vision_cache": vision_cache,
                **kwargs,
            }
            if args.resize_shape is not None:
                stream_kwargs["resize_shape"] = args.resize_shape
            if args.max_long_side_pixel is not None:
                stream_kwargs["max_long_side_pixel"] = args.max_long_side_pixel
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size
            if is_masked_text_diffusion:
                if args.max_denoising_steps is not None:
                    stream_kwargs["max_denoising_steps"] = args.max_denoising_steps
                if args.block_length is not None:
                    stream_kwargs["block_length"] = args.block_length
                if args.num_to_transfer is not None:
                    stream_kwargs["num_to_transfer"] = args.num_to_transfer
                if args.max_transfer_per_step is not None:
                    stream_kwargs["max_transfer_per_step"] = args.max_transfer_per_step
                if args.threshold is not None:
                    stream_kwargs["threshold"] = args.threshold
                if args.min_threshold is not None:
                    stream_kwargs["min_threshold"] = args.min_threshold
                if args.editing_threshold is not None:
                    stream_kwargs["editing_threshold"] = args.editing_threshold
                if args.max_post_steps is not None:
                    stream_kwargs["max_post_steps"] = args.max_post_steps
                if args.stability_steps is not None:
                    stream_kwargs["stability_steps"] = args.stability_steps
            stream_kwargs.update(diffusion_kwargs_from_args(args, config))

            diffusion_output = DiffusionOutputHandler(model, stream_kwargs, True)
            for chunk in stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                args.video,
                **stream_kwargs,
            ):
                if chunk.is_draft:
                    diffusion_output.handle_draft(chunk)
                    continue
                response += chunk.text
                if not diffusion_output.handle_text(chunk.text):
                    print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            diffusion_output.finish(response)
            print()

    else:
        gen_kwargs = {
            "image": args.image,
            "audio": args.audio,
            "video": args.video,
            "fps": args.fps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "repetition_penalty": args.repetition_penalty,
            "repetition_context_size": args.repetition_context_size,
            "presence_penalty": args.presence_penalty,
            "presence_context_size": args.presence_context_size,
            "frequency_penalty": args.frequency_penalty,
            "frequency_context_size": args.frequency_context_size,
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
        if args.max_long_side_pixel is not None:
            gen_kwargs["max_long_side_pixel"] = args.max_long_side_pixel
        if args.prefill_step_size is not None:
            gen_kwargs["prefill_step_size"] = args.prefill_step_size
        if is_masked_diffusion_text_model(model):
            if args.max_denoising_steps is not None:
                gen_kwargs["max_denoising_steps"] = args.max_denoising_steps
            if args.block_length is not None:
                gen_kwargs["block_length"] = args.block_length
            if args.num_to_transfer is not None:
                gen_kwargs["num_to_transfer"] = args.num_to_transfer
            if args.max_transfer_per_step is not None:
                gen_kwargs["max_transfer_per_step"] = args.max_transfer_per_step
            if args.threshold is not None:
                gen_kwargs["threshold"] = args.threshold
            if args.min_threshold is not None:
                gen_kwargs["min_threshold"] = args.min_threshold
            if args.editing_threshold is not None:
                gen_kwargs["editing_threshold"] = args.editing_threshold
            if args.max_post_steps is not None:
                gen_kwargs["max_post_steps"] = args.max_post_steps
            if args.stability_steps is not None:
                gen_kwargs["stability_steps"] = args.stability_steps
        gen_kwargs.update(diffusion_kwargs_from_args(args, config))
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
            stats = format_speculative_stats(draft_model)
            if stats is not None:
                print(stats)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.generate ...` directly is deprecated."
        " Use `mlx_vlm generate` or `python -m mlx_vlm generate` instead."
    )
    main()
