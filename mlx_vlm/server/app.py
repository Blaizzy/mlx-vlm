import asyncio
import gc
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from threading import Lock
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import CacheNotFound, RepositoryNotFoundError

from .. import apc as _apc
from ..generate import (
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from ..generate.edit_image import load_image_edit_model
from ..generate.image import is_image_generation_model, load_image_generation_model
from ..structured import build_json_schema_logits_processor
from ..tool_parsers import _infer_tool_parser_from_processor
from ..version import __version__
from ..vision_cache import VisionFeatureCache
from .anthropic import register_routes as register_anthropic_routes
from .audio import register_routes as register_audio_routes
from .generation import (
    GenerationArguments,
    PromptTooLongError,
    ResponseGenerator,
    ServerMetricsStore,
    _build_metrics_envelope,
    get_configured_context_limit,
    get_kv_group_size,
    get_kv_quant_scheme,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_server_enable_thinking,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
    get_top_logprobs_k,
)
from .openai import register_routes as register_openai_routes
from .responses_state import _split_thinking as _split_thinking_text
from .runtime import ModelCacheRegistry, runtime
from .schemas import ChatLogprobContent, ModelsResponse, TopLogprob

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
SERVER_API_KEY_ENV = "MLX_VLM_SERVER_API_KEY"

logger = logging.getLogger("mlx_vlm.server")


def _server_api_key() -> Optional[str]:
    key = os.environ.get(SERVER_API_KEY_ENV)
    return key if key else None


def _require_management_api_key(request: Request) -> None:
    api_key = _server_api_key()
    if api_key is None:
        return

    expected = f"Bearer {api_key}"
    supplied = request.headers.get("Authorization", "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _cache_group_for_cache(cache: dict) -> str:
    model_kind = cache.get("model_kind")
    if model_kind == "image_generation":
        return "image_generation"
    if model_kind == "image_edit":
        return "image_edit"
    if model_kind == "audio_tts":
        return "tts"
    if model_kind == "audio_stt":
        return "stt"
    if model_kind == "audio":
        return "audio"
    return "text_generation"


def _model_cache_registry() -> ModelCacheRegistry:
    cache = runtime.model_cache
    if isinstance(cache, ModelCacheRegistry):
        return cache

    registry = ModelCacheRegistry()
    if isinstance(cache, dict) and cache:
        registry.set(_cache_group_for_cache(cache), cache)
    runtime.model_cache = registry
    return registry


def _server_runtime_snapshot() -> dict:
    registry = _model_cache_registry()
    default_cache = registry.for_kind("text_generation")
    processor = default_cache.get("processor")
    config = default_cache.get("config")
    text_config = getattr(config, "text_config", None)
    native_context_size = getattr(text_config, "max_position_embeddings", None)
    configured_context_limit = get_configured_context_limit()
    effective_context_limit = (
        min(native_context_size, configured_context_limit)
        if native_context_size is not None and configured_context_limit is not None
        else configured_context_limit or native_context_size
    )
    queue_depth = 0
    if runtime.response_generator is not None and hasattr(
        runtime.response_generator, "requests"
    ):
        try:
            queue_depth = runtime.response_generator.requests.qsize()
        except Exception:
            queue_depth = 0
    audio_queue_depth = 0
    if runtime.audio_queue is not None and hasattr(runtime.audio_queue, "qsize"):
        try:
            audio_queue_depth = runtime.audio_queue.qsize()
        except Exception:
            audio_queue_depth = 0
    return {
        "loaded_model": default_cache.get("model_path", None),
        "loaded_adapter": default_cache.get("adapter_path", None),
        "loaded_models": {
            group: {
                "model": cache.get("model_path"),
                "adapter": cache.get("adapter_path"),
                "model_kind": cache.get("model_kind"),
            }
            for group, cache in registry.items()
        },
        "model_kind": default_cache.get("model_kind", "text_generation"),
        "loaded_context_size": native_context_size,
        "configured_context_limit": configured_context_limit,
        "effective_context_limit": effective_context_limit,
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(processor) if processor else None
        ),
        "continuous_batching_enabled": runtime.response_generator is not None,
        "request_queue_depth": queue_depth,
        "audio_queue_depth": audio_queue_depth,
        "apc": (
            {"enabled": False}
            if runtime.apc_manager is None
            else {"enabled": True, **runtime.apc_manager.stats_snapshot()}
        ),
    }


_DISABLED_REASONING_EFFORTS = {"none", "off", "disabled", "false", "0"}


def _request_field_is_set(request, field_name: str) -> bool:
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None:
        return field_name in fields_set
    return getattr(request, field_name, None) is not None


def _reasoning_effort_enabled(effort) -> Tuple[Optional[bool], Optional[str]]:
    if effort is None:
        return None, None
    normalized = str(effort).strip().lower()
    if not normalized:
        return None, None
    return normalized not in _DISABLED_REASONING_EFFORTS, normalized


def _standard_reasoning_control(
    request,
) -> Tuple[Optional[bool], Optional[str], bool]:
    """Normalize OpenAI reasoning fields into a thinking-mode decision."""
    if _request_field_is_set(request, "reasoning"):
        reasoning = getattr(request, "reasoning", None)
        if hasattr(reasoning, "model_dump"):
            reasoning = reasoning.model_dump(exclude_none=True)
        if isinstance(reasoning, dict):
            enabled, effort = _reasoning_effort_enabled(reasoning.get("effort"))
            return (True if enabled is None else enabled), effort, True
        if isinstance(reasoning, bool):
            return reasoning, None, True
        enabled, effort = _reasoning_effort_enabled(reasoning)
        if enabled is not None:
            return enabled, effort, True

    if _request_field_is_set(request, "reasoning_effort"):
        enabled, effort = _reasoning_effort_enabled(
            getattr(request, "reasoning_effort", None)
        )
        if enabled is not None:
            return enabled, effort, True

    return None, None, False


def _build_gen_args(
    request, processor=None, tenant_id: Optional[str] = None
) -> GenerationArguments:
    """Build GenerationArguments from an OpenAIRequest or ChatRequest."""
    max_tokens = getattr(request, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(request, "max_output_tokens", None)
    if max_tokens is None:
        max_tokens = get_server_max_tokens()
    logit_bias = getattr(request, "logit_bias", None)
    if logit_bias is not None and isinstance(logit_bias, dict):
        logit_bias = {int(k): v for k, v in logit_bias.items()}
    standard_reasoning, reasoning_effort, has_standard_reasoning = (
        _standard_reasoning_control(request)
    )
    server_enable_thinking = get_server_enable_thinking()
    if _request_field_is_set(request, "enable_thinking"):
        enable_thinking = bool(getattr(request, "enable_thinking", False))
        template_reasoning = enable_thinking
    elif has_standard_reasoning:
        enable_thinking = bool(standard_reasoning)
        template_reasoning = enable_thinking
    else:
        enable_thinking = server_enable_thinking
        # Preserve a model template's native default when the server default is
        # off and the request did not express a reasoning preference.
        template_reasoning = True if server_enable_thinking else None
    default_temperature = _model_config_field_or_default(
        processor, "temperature", DEFAULT_TEMPERATURE
    )
    default_top_p = _model_config_field_or_default(processor, "top_p", DEFAULT_TOP_P)
    default_top_k = _model_config_field_or_default(processor, "top_k", 0)
    if _model_config_field_or_default(processor, "do_sample", None) is False:
        default_temperature = 0.0
    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=_request_field_or_default(
            request, "temperature", default_temperature
        ),
        top_p=_request_field_or_default(request, "top_p", default_top_p),
        top_k=_request_field_or_default(request, "top_k", default_top_k),
        min_p=getattr(request, "min_p", 0.0),
        seed=getattr(request, "seed", None),
        logprobs=bool(getattr(request, "logprobs", False)),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        repetition_context_size=_request_field_or_default(
            request,
            "repetition_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        presence_penalty=getattr(request, "presence_penalty", None),
        presence_context_size=_request_field_or_default(
            request,
            "presence_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        frequency_penalty=getattr(request, "frequency_penalty", None),
        frequency_context_size=_request_field_or_default(
            request,
            "frequency_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        max_denoising_steps=_request_field_or_default(
            request, "max_denoising_steps", None
        ),
        block_length=_request_field_or_default(request, "block_length", None),
        num_to_transfer=_request_field_or_default(request, "num_to_transfer", None),
        max_transfer_per_step=_request_field_or_default(
            request, "max_transfer_per_step", None
        ),
        editing_threshold=_request_field_or_default(request, "editing_threshold", None),
        max_post_steps=_request_field_or_default(request, "max_post_steps", None),
        stability_steps=_request_field_or_default(request, "stability_steps", None),
        diffusion_full_canvas=_request_field_or_default(
            request, "diffusion_full_canvas", None
        ),
        diffusion_min_canvas_length=_request_field_or_default(
            request, "diffusion_min_canvas_length", None
        ),
        diffusion_max_canvas_length=_request_field_or_default(
            request, "diffusion_max_canvas_length", None
        ),
        diffusion_sampler=_request_field_or_default(request, "diffusion_sampler", None),
        threshold=_request_field_or_default(request, "threshold", None),
        min_threshold=_request_field_or_default(request, "min_threshold", None),
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        reasoning=template_reasoning,
        reasoning_effort=reasoning_effort,
        thinking_budget=_request_field_or_default(
            request, "thinking_budget", get_server_thinking_budget()
        ),
        thinking_start_token=_request_field_or_default(
            request, "thinking_start_token", get_server_thinking_start_token()
        ),
        thinking_end_token=_request_field_or_default(
            request, "thinking_end_token", get_server_thinking_end_token()
        ),
        tenant_id=tenant_id,
    )
    if processor is not None:
        args.logits_processors = _build_structured_logits_processors(request, processor)
    return args


def _request_field_or_default(request, field_name: str, default):
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None and field_name not in fields_set:
        return default
    value = getattr(request, field_name, default)
    return default if value is None else value


def _model_config_field_or_default(processor, field_name: str, default):
    config = runtime.model_cache.get("config")
    if config is None and processor is not None:
        config = getattr(processor, "config", None)
    return getattr(config, field_name, default)


def _read_tenant_id(http_request) -> Optional[str]:
    """Pull a per-tenant APC salt from the request headers.

    Honoured headers (in order): ``X-APC-Tenant``, ``X-Tenant-Id``.
    """
    if http_request is None or not hasattr(http_request, "headers"):
        return None
    h = http_request.headers
    return h.get("x-apc-tenant") or h.get("x-tenant-id") or None


async def _preflight_stream_context_budget(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    images: Optional[List] = None,
    audio: Optional[List] = None,
    videos: Optional[List] = None,
    args: GenerationArguments,
):
    """Reject over-budget streaming requests before the HTTP stream starts."""
    if runtime.response_generator is None:
        return
    try:
        validate_kwargs = {"images": images, "audio": audio, "args": args}
        if videos is not None:
            validate_kwargs["videos"] = videos
        await asyncio.to_thread(
            runtime.response_generator.validate_context_budget,
            prompt,
            **validate_kwargs,
        )
    except PromptTooLongError as e:
        runtime.metrics.record_failure(
            endpoint=endpoint,
            model=model,
            stream=True,
            error=str(e),
        )
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=400, detail=str(e))


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _extract_response_format_schema(request) -> Optional[Union[str, dict]]:
    response_format = _as_plain_dict(getattr(request, "response_format", None))

    text_config = _as_plain_dict(getattr(request, "text", None))
    if response_format is None and isinstance(text_config, dict):
        response_format = _as_plain_dict(text_config.get("format"))

    if response_format is None:
        return None

    format_type = response_format.get("type")
    if format_type in (None, "text"):
        return None
    if format_type in ("json_object", "object"):
        return {"type": "object"}
    if format_type != "json_schema":
        raise ValueError(f"Unsupported response_format type: {format_type!r}")

    json_schema = _as_plain_dict(response_format.get("json_schema"))
    if json_schema is None:
        # Responses API text.format places schema directly on the format object.
        json_schema = response_format

    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    if schema is None:
        raise ValueError("response_format json_schema must include a schema field")
    return schema


def _build_structured_logits_processors(request, processor):
    schema = _extract_response_format_schema(request)
    if schema is None:
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    logits_processor = _server_package_attr(
        "build_json_schema_logits_processor",
        build_json_schema_logits_processor,
    )(tokenizer, schema)
    return [logits_processor]


def _count_thinking_tag_tokens(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    if (
        thinking_start_token
        and thinking_end_token
        and thinking_start_token in text
        and thinking_end_token in text
    ):
        return 2
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content)."""
    return _split_thinking_text(text, thinking_start_token, thinking_end_token)


def _decode_token(tokenizer, token_id: int) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Shared mutable server runtime state.
runtime.metrics = ServerMetricsStore()


def _server_package_attr(name, fallback=None):
    package = sys.modules.get(__package__)
    if package is not None and hasattr(package, name):
        return getattr(package, name)
    if fallback is not None:
        return fallback
    return globals()[name]


def __getattr__(name):
    legacy_runtime_attrs = {
        "model_cache": "model_cache",
        "response_generator": "response_generator",
        "apc_manager": "apc_manager",
        "server_metrics": "metrics",
    }
    if name in legacy_runtime_attrs:
        return getattr(runtime, legacy_runtime_attrs[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_audio_model(model_path: str):
    from mlx_audio.utils import load_model

    return load_model(model_path)


@asynccontextmanager
async def lifespan(app):
    model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    adapter_path = os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
    if model_path:
        logger.info("Pre-loading language model: %s", model_path)
        get_cached_model(model_path, adapter_path, model_kind="text_generation")
        kv_bits = os.environ.get("KV_BITS")
        kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
        if kv_bits:
            logger.info("KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme)
        logger.info("Language model ready, continuous batching enabled.")

    preload_models = (
        (
            os.environ.pop("MLX_VLM_PRELOAD_IMAGE_MODEL", None),
            None,
            "image_generation",
            "image generation model",
        ),
        (
            os.environ.pop("MLX_VLM_PRELOAD_TTS_MODEL", None),
            None,
            "audio_tts",
            "text-to-speech model",
        ),
        (
            os.environ.pop("MLX_VLM_PRELOAD_STT_MODEL", None),
            None,
            "audio_stt",
            "speech-to-text model",
        ),
    )
    for preload_model_path, preload_adapter_path, model_kind, label in preload_models:
        if not preload_model_path:
            continue
        logger.info("Pre-loading %s: %s", label, preload_model_path)
        get_cached_model(
            preload_model_path,
            preload_adapter_path,
            model_kind=model_kind,
        )
        logger.info("%s ready.", label.capitalize())
    try:
        yield
    finally:
        if runtime.audio_queue is not None:
            runtime.audio_queue.stop_and_join()
            runtime.audio_queue = None


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once


_INHERIT_ADAPTER = object()


def _unload_model_cache_group(cache_group: str) -> bool:
    registry = _model_cache_registry()
    cache = registry.for_kind(cache_group)
    if not cache:
        return False

    logger.info(
        "Unloading %s model: %s (adapter=%s)",
        cache_group,
        cache.get("model_path"),
        cache.get("adapter_path"),
    )

    response_generator = cache.get("response_generator")
    if response_generator is not None:
        logger.info("Stopping response generator.")
        response_generator.stop_and_join()
        if runtime.response_generator is response_generator:
            runtime.response_generator = None

    apc_manager = cache.get("apc_manager")
    if apc_manager is not None:
        apc_manager.clear()
        if runtime.apc_manager is apc_manager:
            runtime.apc_manager = None

    if "vision_cache" in cache:
        cache["vision_cache"].clear()

    registry.pop(cache_group)
    gc.collect()
    mx.clear_cache()
    return True


def _audio_model_kind(model_kind: str) -> bool:
    return model_kind in ("audio", "audio_tts", "audio_stt")


def _audio_cache_group(model_kind: str) -> str:
    if model_kind == "audio_tts":
        return "tts"
    if model_kind == "audio_stt":
        return "stt"
    return "audio"


def get_cached_model(
    model_path: str,
    adapter_path=_INHERIT_ADAPTER,
    *,
    model_kind: str = "auto",
):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    Also creates/updates the ResponseGenerator for continuous batching.
    """
    load_as_edit = model_kind == "image_edit"
    load_as_audio = _audio_model_kind(model_kind)
    load_as_image = model_kind == "image_generation" or (
        model_kind == "auto" and is_image_generation_model(model_path)
    )
    if load_as_edit:
        cache_group = "image_edit"
        effective_model_kind = "image_edit"
    elif load_as_audio:
        cache_group = _audio_cache_group(model_kind)
        effective_model_kind = model_kind
    elif load_as_image:
        cache_group = "image_generation"
        effective_model_kind = "image_generation"
    else:
        cache_group = "text_generation"
        effective_model_kind = "text_generation" if model_kind == "auto" else model_kind

    registry = _model_cache_registry()
    if adapter_path is _INHERIT_ADAPTER:
        cached_cache = registry.for_kind(cache_group)
        cached = cached_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path, effective_model_kind)
    cached_cache = registry.for_kind(cache_group)

    # Return from cache if already loaded and matches the requested paths
    if cached_cache and cached_cache.get("cache_key") == cache_key:
        if cache_group == "text_generation":
            runtime.response_generator = cached_cache.get("response_generator")
            runtime.apc_manager = cached_cache.get("apc_manager")
        logger.debug("Using cached model: %s (adapter=%s)", model_path, adapter_path)
        return (
            cached_cache["model"],
            cached_cache["processor"],
            cached_cache["config"],
        )

    # If this kind has a different model cached, clear only that cache group.
    if cached_cache:
        logger.info("New %s model requested; clearing its existing cache.", cache_group)
        _unload_model_cache_group(cache_group)

    if load_as_edit:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image edit models.",
            )
        logger.info("Loading image edit model: %s", model_path)
        try:
            model = load_image_edit_model(model_path)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image edit model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load image edit model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "family", "image_edit"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_edit",
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    if load_as_image:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image generation models.",
            )
        logger.info("Loading image generation model: %s", model_path)
        try:
            model = load_image_generation_model(model_path)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image generation model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load image generation model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "family", "image_generation"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_generation",
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    if load_as_audio:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for audio models.",
            )
        logger.info("Loading audio model: %s", model_path)
        try:
            model = _server_package_attr("load_audio_model", load_audio_model)(
                model_path
            )
        except RepositoryNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model not found: {model_path!r} is not a known "
                    "Hugging Face repo or local path"
                ),
            ) from e
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported audio model: {e}"
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load audio model: {e}"
            ) from e
        config = SimpleNamespace(
            model_type=getattr(model, "model_type", "audio"),
            text_config=None,
        )
        cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": model_kind,
            "generation_lock": Lock(),
        }
        registry.set(cache_group, cache)
        return model, None, config

    vision_cache_size = int(os.environ.get("MLX_VLM_VISION_CACHE_SIZE", "20"))
    vision_cache = VisionFeatureCache(max_size=vision_cache_size)

    # APC: build a shared block pool if opted in via env var.
    runtime.apc_manager = _apc.from_env(model_namespace=model_path)

    # KV cache quantization (uniform or TurboQuant)
    kv_bits = get_quantized_kv_bits(model_path)
    kv_group_size = get_kv_group_size()
    quantized_kv_start = get_quantized_kv_start()
    kv_quant_scheme = get_kv_quant_scheme()

    response_generator = ResponseGenerator(
        model_path=model_path,
        adapter_path=adapter_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
        top_logprobs_k=get_top_logprobs_k(),
        apc_manager=runtime.apc_manager,
    )
    try:
        model, processor, config = response_generator.wait_until_ready()
    except Exception:
        response_generator.stop_and_join()
        vision_cache.clear()
        raise

    # Dry-run APC layout when the shared pool is enabled (log-only; never blocks serve).
    if runtime.apc_manager is not None:
        try:
            _apc.self_check_model_apc(model, kv_bits=kv_bits)
        except Exception as exc:
            logger.warning("APC self-check raised unexpectedly: %s", exc)

    cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
        "model_kind": "text_generation",
        "response_generator": response_generator,
        "apc_manager": runtime.apc_manager,
    }
    registry.set(cache_group, cache)
    runtime.response_generator = response_generator
    runtime.apc_manager = cache["apc_manager"]

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    unloaded_any = False
    if runtime.audio_queue is not None:
        is_audio_worker = getattr(
            runtime.audio_queue, "is_worker_thread", lambda: False
        )
        if not is_audio_worker():
            logger.info("Stopping audio request queue.")
            runtime.audio_queue.stop_and_join()
            runtime.audio_queue = None
            unloaded_any = True

    registry = _model_cache_registry()
    for cache_group, _ in list(registry.items()):
        unloaded_any = _unload_model_cache_group(cache_group) or unloaded_any

    runtime.response_generator = None
    runtime.apc_manager = None
    gc.collect()
    mx.clear_cache()
    if unloaded_any:
        logger.info("Model caches cleared.")
    return unloaded_any


_protocol_deps = SimpleNamespace(
    INHERIT_ADAPTER=_INHERIT_ADAPTER,
    get_cached_model=lambda *args, **kwargs: _server_package_attr("get_cached_model")(
        *args, **kwargs
    ),
    generate=lambda *args, **kwargs: _server_package_attr("generate")(*args, **kwargs),
    stream_generate=lambda *args, **kwargs: _server_package_attr("stream_generate")(
        *args, **kwargs
    ),
    apply_chat_template=lambda *args, **kwargs: _server_package_attr(
        "apply_chat_template"
    )(*args, **kwargs),
    infer_tool_parser_from_processor=lambda *args, **kwargs: _server_package_attr(
        "_infer_tool_parser_from_processor"
    )(*args, **kwargs),
    load_tool_module=lambda *args, **kwargs: _server_package_attr("load_tool_module")(
        *args, **kwargs
    ),
    build_gen_args=_build_gen_args,
    read_tenant_id=_read_tenant_id,
    preflight_stream_context_budget=_preflight_stream_context_budget,
    as_plain_dict=_as_plain_dict,
    split_thinking=_split_thinking,
    count_thinking_tag_tokens=_count_thinking_tag_tokens,
    make_logprob_content=_make_logprob_content,
    build_metrics_envelope=_build_metrics_envelope,
)
register_anthropic_routes(app, _protocol_deps)
register_openai_routes(app, _protocol_deps)
register_audio_routes(app, _protocol_deps)


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    required_files = {"config.json", "tokenizer_config.json"}

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        has_weights = "model.safetensors.index.json" in file_names or any(
            file_name.endswith(".safetensors") for file_name in file_names
        )
        return required_files.issubset(file_names) and has_weights

    # Scan the cache directory for downloaded mlx models when it exists.
    try:
        hf_cache_info = _server_package_attr("scan_cache_dir", scan_cache_dir)()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]
    except CacheNotFound:
        downloaded_models = []

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]
    loaded_models = {
        cache.get("model_path")
        for cache in _model_cache_registry().values()
        if cache.get("model_path")
    }
    loaded_model = _model_cache_registry().get("model_path")
    if loaded_model:
        loaded_models.add(loaded_model)
    for loaded in sorted(loaded_models):
        if all(model["id"] != loaded for model in models):
            models.append(
                {"id": loaded, "object": "model", "created": int(time.time())}
            )

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"mlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check(request: Request):
    """
    Check if the server is healthy and what model is loaded.
    """
    _require_management_api_key(request)
    runtime = _server_runtime_snapshot()
    return {
        "status": "healthy",
        "loaded_model": runtime["loaded_model"],
        "loaded_adapter": runtime["loaded_adapter"],
        "loaded_models": runtime["loaded_models"],
        "loaded_context_size": runtime["loaded_context_size"],
        "configured_context_limit": runtime["configured_context_limit"],
        "effective_context_limit": runtime["effective_context_limit"],
        "loaded_tool_parser": runtime["loaded_tool_parser"],
        "continuous_batching_enabled": runtime["continuous_batching_enabled"],
        "apc_enabled": runtime["apc"]["enabled"],
    }


@app.get("/metrics")
@app.get("/v1/metrics", include_in_schema=False)
async def metrics_endpoint(request: Request):
    _require_management_api_key(request)
    payload = runtime.metrics.snapshot()
    payload["server"] = _server_runtime_snapshot()
    return payload


@app.get("/v1/cache/stats")
@app.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats(request: Request):
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    _require_management_api_key(request)
    if runtime.apc_manager is None:
        return {"enabled": False}
    snap = runtime.apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@app.post("/v1/cache/reset")
@app.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset(request: Request):
    _require_management_api_key(request)
    if runtime.apc_manager is None:
        return {"enabled": False}
    runtime.apc_manager.clear()
    return {"enabled": True, "status": "cleared"}


@app.post("/unload")
async def unload_model_endpoint(request: Request):
    """
    Unload the currently loaded model from memory.
    """
    _require_management_api_key(request)
    snapshot = _server_runtime_snapshot()
    unloaded_info = {
        "model_name": snapshot["loaded_model"],
        "adapter_name": snapshot["loaded_adapter"],
        "models": snapshot["loaded_models"],
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
