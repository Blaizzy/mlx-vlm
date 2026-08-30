import argparse
import codecs
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .. import apc as _apc
from ..kv_quant import from_legacy as kv_quant_from_legacy
from ..models import cache
from ..prompt_utils import apply_chat_template
from ..speculative.utils import format_speculative_stats
from ..tokenizer_utils import make_streaming_detokenizer
from ..utils import (
    StoppingCriteria,
    ThinkingBudgetCriteria,
    load,
    prepare_inputs,
    should_add_special_tokens,
)
from .common import (
    DEFAULT_DIFFUSION_MAX_DENOISING_STEPS,
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    GenerationResult,
    generation_stream,
    wired_limit,
)
from .image import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STEPS,
    DEFAULT_IMAGE_TASK,
    run_image_generation_cli,
)
from .video_generation import DEFAULT_VIDEO_STEPS, run_video_generation_cli

logger = logging.getLogger("mlx_vlm.generate")

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_VIDEO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_SEED = 0
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text, an image, or a video with a supported model."
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
        choices=("text", "image", "video"),
        default="text",
        help=(
            "Generate text with a VLM, an image with a supported image model, "
            "or a video with a supported video model."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for image or video generation.",
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
            "Output size as WIDTHxHEIGHT. Image generation defaults to "
            f"{DEFAULT_IMAGE_SIZE}; image editing defaults to the first reference "
            "image size, and video uses the model default when omitted."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of inference steps. Defaults to "
            f"{DEFAULT_IMAGE_STEPS} for images and {DEFAULT_VIDEO_STEPS} for videos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "PRNG seed for reproducible sampling and diffusion canvas init. "
            "Image and video generation default to a random 32-bit seed."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=("t2va", "fl2va", "ref2va"),
        default=None,
        help=(
            "Video-generation workflow. Inferred from --image/--last-image or "
            "--reference when omitted."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Requested number of generated video frames.",
    )
    parser.add_argument(
        "--last-image",
        type=str,
        default=None,
        help="Last-frame conditioning image for FL2VA video generation.",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=None,
        metavar="KIND=PATH",
        help=(
            "Ordered Ref2VA reference; KIND is image, video, or audio. Repeat "
            "the argument to preserve semantic reference order."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
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
        "--video-max-frames",
        type=int,
        default=16,
        help="Cap on frames sent when video falls back to ordered images "
        "(long clips are re-sampled evenly to this count).",
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
        default="confidence-threshold",
        help=(
            "Canvas update sampler for diffusion generation. Use entropy-bound "
            "for reference-style denoising; confidence-threshold is faster for "
            "quantized block-diffusion checkpoints."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Token probability threshold for diffusion confidence transfer. "
            f"Default: {DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD:g} for "
            "confidence-threshold sampling; "
            "masked-diffusion models use their checkpoint reference defaults."
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
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling: keep the smallest set of tokens whose "
        "probabilities sum to this. 1.0 disables it.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Keep only the k most probable tokens. 0 disables it.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=DEFAULT_MIN_P,
        help="Drop tokens whose probability is below this fraction of the "
        "most probable token's. 0 disables it.",
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
        default=False,
        help=(
            "Enable detailed output and progress bars. By default only the final "
            "result is printed."
        ),
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
        "--kv-key-bits",
        type=float,
        default=None,
        help="Override the TurboQuant key bit-width (defaults to floor(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-value-bits",
        type=float,
        default=None,
        help="Override the TurboQuant value bit-width (defaults to ceil(--kv-bits)).",
    )
    parser.add_argument(
        "--kv-key-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for keys only.",
    )
    parser.add_argument(
        "--kv-value-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=None,
        help="Override the KV quantization backend for values only.",
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
        "--expert-cache-gb",
        type=float,
        default=None,
        help="For an mlx_vlm.moe_offload checkpoint, bound the resident routed-"
        "expert set to this many GB (default: 70%% of the GPU's recommended "
        "working set). Ignored for a normal, non-offloaded checkpoint.",
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
        help="Speculative drafter path or HF id (e.g. z-lab/Qwen3.5-4B-DFlash).",
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "eagle3", "mtp"],
        help="Drafter family. Supported: 'dflash' (Qwen3.5 DFlash), "
        "'eagle3' (Speculators/SGLang EAGLE-3), "
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
        help=(
            "Enable thinking in the chat template. Templates that use "
            "thinking_mode receive thinking_mode='enabled'."
        ),
    )
    parser.add_argument(
        "--thinking-mode",
        choices=("enabled", "disabled", "adaptive"),
        default=None,
        help=(
            "Set the chat-template thinking mode when supported. "
            "Choices: enabled, disabled, adaptive."
        ),
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


from .diffusion import (
    DEFAULT_DIFFUSION_CONFIDENCE_THRESHOLD,
    DEFAULT_DIFFUSION_MIN_CANVAS_LENGTH,
    DiffusionOutputHandler,
    diffusion_kwargs_from_args,
    is_diffusion_model,
    stream_diffusion_generate_from_kwargs,
)
from .types import GenerateKwargs, ProcessorLike, Unpack


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
    # ``generate_step`` prepares embeddings after APC has trimmed the input to
    # the uncached suffix. Preserve the full-prompt positions explicitly so a
    # Qwen-style embedding helper cannot replace them with suffix-local 0..N
    # positions. The language model slices these arrays at its cache offset.
    kwargs["position_ids"] = position_ids
    kwargs["rope_deltas"] = rope_deltas
    return True


def _cache_fully_retained(c: Any) -> bool:
    """Whether ``c`` still holds its whole sequence from position 0.

    Only such a cache can be rolled back to an earlier prefix. Note this is not
    ``is_trimmable()``: ``BufferedRotatingKVCache`` reports itself trimmable even
    after it has evicted early tokens, and trimming it then would only clamp to
    its retained window and desync ``offset`` from the flat layers.
    """
    children = getattr(c, "caches", None)
    if children is not None:  # CacheList
        return all(_cache_fully_retained(x) for x in children)
    start_position = getattr(c, "start_position", None)
    if start_position is not None:  # Buffered/Chunked: evicts by advancing start
        return int(start_position) == 0
    max_size = getattr(c, "max_size", None)
    if max_size is not None:  # RotatingKVCache: reusable until the window wraps
        return int(getattr(c, "offset", 0) or 0) <= int(max_size)
    return True  # KVCache / QuantizedKVCache retain the whole sequence


def _prefix_cache_trim_amount(kv_cache: List[Any], prefix_len: int) -> Optional[int]:
    """Trailing tokens to drop so ``kv_cache`` keeps only its first ``prefix_len``.

    The old reuse path sliced key/value arrays directly --
    ``keys[..., :prefix_len, :]`` -- which is only valid for a flat cache. On a
    rotating (sliding-window) cache it slices a ring buffer by a logical length,
    taking an arbitrary rotation of the window and leaving the ring index stale:
    silent output corruption, or a broadcast crash once speculative decoding wraps
    the cache in ``BufferedRotatingKVCache``. Returns the number of tokens to drop
    (``0`` when the whole cache is reusable), or ``None`` when an entry has already
    evicted part of the prefix and the caller must cold-prefill instead.
    """
    cached_len = max((int(getattr(c, "offset", 0) or 0) for c in kv_cache), default=0)
    n_drop = max(0, cached_len - prefix_len)
    if n_drop and not all(_cache_fully_retained(c) for c in kv_cache):
        return None
    return n_drop


from .ar import generate_step


def stream_generate(
    model: nn.Module,
    processor: ProcessorLike | PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str], None] = None,
    audio: Union[str, List[str], None] = None,
    video: Union[str, List[str], None] = None,
    **kwargs: Unpack[GenerateKwargs],
) -> Generator[GenerationResult, None, None]:
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
    # Preserve only explicitly supplied sequence tensors as semantic APC
    # inputs. Tensors produced by prepare_inputs span the complete prompt and
    # therefore change whenever text is appended, even when the old token
    # prefix is identical.
    custom_inputs_embeds = kwargs.get("inputs_embeds")
    custom_mask = kwargs.get("mask")

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

    add_special_tokens = should_add_special_tokens(model.config.model_type, processor)

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
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

    if is_diffusion_model(model, kwargs):
        yield from stream_diffusion_generate_from_kwargs(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            mask,
            skip_special_token_ids,
            kwargs,
            skip_special_tokens=skip_special_tokens,
            verbose=verbose,
        )
        return

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
    apc_coordinator: Optional[_apc.APCCoordinator] = None

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

    if apc_manager is not None:
        apc_coordinator = _apc.APCCoordinator(apc_manager, model.language_model)
        if not apc_coordinator.enabled:
            apc_coordinator = None
            apc_manager = None

    if apc_manager is not None:
        image_hash = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=image)
        audio_features = kwargs.get("input_features")
        video_features = kwargs.get("pixel_values_videos")
        apc_extra_hash = _apc.semantic_extra_hash(
            tenant=apc_tenant,
            image_hash=image_hash,
            media={
                "audio": audio_features if audio_features is not None else audio,
                "video": video_features if video_features is not None else video,
                "embeddings": custom_inputs_embeds,
                "masks": custom_mask,
            },
            model=model,
            processor=processor,
        )

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)
        kv_cache = prompt_cache_state.cache
        # None => a cache can't be trimmed back to the shared prefix (wrapped
        # rotating window); reusing it would corrupt state, so cold-prefill instead.
        n_drop = _prefix_cache_trim_amount(kv_cache, prefix_len)
        if (
            0 < prefix_len < input_ids.shape[1]
            and n_drop is not None
            and _apc_suffix_is_text_only(prefix_len)
            and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
        ):
            # Drop cached tokens past the shared prefix via each cache's own trim().
            for c in kv_cache:
                if n_drop:
                    c.trim(n_drop)
            reused_prefix_len = prefix_len
            # Trim to only new tokens
            input_ids = input_ids[:, prefix_len:]
            pixel_values = None
            kwargs.pop("cached_image_features", None)
            kwargs["prompt_cache"] = kv_cache

    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    if apc_manager is not None and reused_prefix_len == 0:
        plan = apc_coordinator.lookup(
            full_input_ids_list,
            extra_hash=apc_extra_hash,
            safe_lookup_min=apc_safe_prefix_lookup_min,
            suffix_is_text_only=_apc_suffix_is_text_only,
            prefix_has_media=_apc_prefix_has_media_tokens,
        )
        if plan is not None:
            plen = plan["prefix_len"]
            warm_cache = plan.get("warm_cache")
            matched_blocks = plan.get("matched_blocks") or []
            primed = _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
            if primed:
                reused_prefix_len = plen
                input_ids = input_ids[:, plen:]
                pixel_values = None
                kwargs.pop("cached_image_features", None)
                apc_blocks_in_use = matched_blocks
                _quant_policy = kv_quant_from_legacy(
                    kwargs.get("kv_bits"),
                    kwargs.get("kv_quant_scheme"),
                    kwargs.get("kv_group_size", 64),
                    kwargs.get("kv_key_bits"),
                    kwargs.get("kv_value_bits"),
                    kwargs.get("kv_key_scheme"),
                    kwargs.get("kv_value_scheme"),
                )
                _quant_cfg = (
                    _quant_policy.to_config() if _quant_policy is not None else None
                )
                kwargs["prompt_cache"] = apc_coordinator.materialize_single(
                    plan,
                    min_capacity_tokens=plen + input_ids.shape[1] + 1,
                    kv_quant_config=_quant_cfg,
                )
            elif warm_cache is None and matched_blocks:
                apc_coordinator.release_hit(plan)

    if thinking_budget is not None:
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        prompt_preopens_thinking = (
            thinking_start_token_id in input_ids.flatten().tolist()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
            prompt_preopens_thinking=prompt_preopens_thinking,
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
        if (
            apc_coordinator is not None
            and apc_coordinator.is_checkpoint
            and reused_prefix_len == 0
        ):
            exact_checkpoint_len = apc_coordinator.checkpoint_len(
                full_input_ids_list, multimodal_token_ids
            )
            if exact_checkpoint_len <= 0:
                exact_checkpoint_len = None

            def exact_checkpoint(prefix_len: int, prompt_cache: List[Any]) -> None:
                apc_coordinator.store_checkpoint(
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
        if apc_coordinator is not None and not apc_coordinator.is_checkpoint:
            try:
                if all_ids is None:
                    all_ids = full_input_ids_list + [
                        t.item() if hasattr(t, "item") else t for t in generated_tokens
                    ]
                apc_coordinator.commit(
                    tracked_cache,
                    all_ids,
                    extra_hash=apc_extra_hash,
                    skip_first_n_tokens=reused_prefix_len,
                    blocks_in_use=apc_blocks_in_use,
                )
            except Exception as e:
                logger.warning("APC store failed: %s", e)
                apc_coordinator.manager.release(apc_blocks_in_use)

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: nn.Module,
    processor: ProcessorLike | PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str], None] = None,
    audio: Union[str, List[str], None] = None,
    video: Union[str, List[str], None] = None,
    verbose: bool = False,
    **kwargs: Unpack[GenerateKwargs],
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

    clean_output = getattr(processor, "clean_output", None)
    if callable(clean_output):
        text = clean_output(text)

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
    if getattr(args, "output_modality", "text") == "video":
        run_video_generation_cli(args)
        return

    if getattr(args, "seed", None) is not None:
        mx.random.seed(args.seed)

    diffusion_arg_defaults = {
        "max_denoising_steps": None,
        "diffusion_full_canvas": False,
        "diffusion_min_canvas_length": None,
        "diffusion_max_canvas_length": None,
        "diffusion_sampler": "confidence-threshold",
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
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
        expert_cache_gb=args.expert_cache_gb,
        max_kv_size=args.max_kv_size,
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

    # Processors without native video support used to drop --video silently:
    # the frames were loaded, the processor ignored the kwarg, and the model
    # hallucinated an answer with no visual input at all. Fall back to sending
    # sampled frames as ordered images (see generate/video.py).
    gen_kwargs_extra = {}
    video_prompt = None
    if args.video:
        from .video import (
            pair_adjacent_frames,
            processor_handles_video,
            resolve_video_inputs,
            sample_video_frames,
            timestamped_frame_messages,
        )

        if not processor_handles_video(processor):
            max_frames = max(2, getattr(args, "video_max_frames", 16) or 16)
            pair_hook = getattr(model, "prepare_video_frame_pairs", None)
            if pair_hook is not None:
                frames, frame_fps = sample_video_frames(args.video, args.fps or 2.0)
                anchors, first_frames, second_frames = pair_adjacent_frames(
                    frames, max_frames
                )
                gen_kwargs_extra.update(pair_hook(processor, second_frames))
                still_count = len(args.image or [])
                args.image = (args.image or []) + first_frames
                user_text = (
                    " ".join(args.prompt)
                    if isinstance(args.prompt, list)
                    else str(args.prompt)
                )
                msgs = timestamped_frame_messages(
                    user_text,
                    args.system,
                    still_count,
                    [a / max(frame_fps, 1e-6) for a in anchors],
                )
                _tok = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                video_prompt = _tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                args.video = None
            else:
                resolution = resolve_video_inputs(
                    processor,
                    args.video,
                    images=args.image,
                    fps=args.fps or 2.0,
                    max_frames=max_frames,
                )
                print(
                    f"{processor.__class__.__name__} has no native video "
                    f"support; sending {resolution.selected_count} of "
                    f"{resolution.sampled_count} sampled "
                    f"frames as ordered images."
                )
                args.image = resolution.images
                args.video = resolution.videos or None

    num_images = len(args.image) if args.image is not None else 0
    num_audios = len(args.audio) if args.audio is not None else 0

    chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    if args.thinking_mode is not None:
        chat_template_kwargs["thinking_mode"] = args.thinking_mode
    if args.video:
        chat_template_kwargs["video"] = args.video
        chat_template_kwargs["fps"] = args.fps

    if video_prompt is not None:
        prompt = video_prompt
    else:
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
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    if args.chat:
        from ..vision_cache import VisionFeatureCache

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
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
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
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size
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
            **gen_kwargs_extra,
            "image": args.image,
            "audio": args.audio,
            "video": args.video,
            "fps": args.fps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
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
            "kv_key_bits": getattr(args, "kv_key_bits", None),
            "kv_value_bits": getattr(args, "kv_value_bits", None),
            "kv_key_scheme": getattr(args, "kv_key_scheme", None),
            "kv_value_scheme": getattr(args, "kv_value_scheme", None),
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
