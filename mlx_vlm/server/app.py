import asyncio
import gc
import logging
import os
import sys
from contextlib import asynccontextmanager
from threading import Lock
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import scan_cache_dir

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
from .generation import (
    GenerationArguments,
    PromptTooLongError,
    ResponseGenerator,
    ServerMetricsStore,
    get_configured_context_limit,
    get_kv_group_size,
    get_kv_quant_scheme,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_server_enable_thinking,
    get_server_max_tokens,
    get_top_logprobs_k,
)
from .openai import register_routes as register_openai_routes
from .runtime import runtime
from .schemas import ChatLogprobContent, ModelsResponse, TopLogprob

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080

logger = logging.getLogger("mlx_vlm.server")


def _server_runtime_snapshot() -> dict:
    config = runtime.model_cache.get("config")
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
    return {
        "loaded_model": runtime.model_cache.get("model_path", None),
        "loaded_adapter": runtime.model_cache.get("adapter_path", None),
        "model_kind": runtime.model_cache.get("model_kind", "text_generation"),
        "loaded_context_size": native_context_size,
        "configured_context_limit": configured_context_limit,
        "effective_context_limit": effective_context_limit,
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(runtime.model_cache.get("processor"))
            if runtime.model_cache.get("processor")
            else None
        ),
        "continuous_batching_enabled": runtime.response_generator is not None,
        "request_queue_depth": queue_depth,
        "apc": (
            {"enabled": False}
            if runtime.apc_manager is None
            else {"enabled": True, **runtime.apc_manager.stats_snapshot()}
        ),
    }


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
    enable_thinking = _request_field_or_default(
        request,
        "enable_thinking",
        get_server_enable_thinking(),
    )
    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=getattr(request, "temperature", DEFAULT_TEMPERATURE),
        top_p=getattr(request, "top_p", DEFAULT_TOP_P),
        top_k=getattr(request, "top_k", 0),
        min_p=getattr(request, "min_p", 0.0),
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
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        thinking_budget=getattr(request, "thinking_budget", None),
        thinking_start_token=getattr(request, "thinking_start_token", None),
        thinking_end_token=getattr(request, "thinking_end_token", None),
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
    args: GenerationArguments,
):
    """Reject over-budget streaming requests before the HTTP stream starts."""
    if runtime.response_generator is None:
        return
    try:
        await asyncio.to_thread(
            runtime.response_generator.validate_context_budget,
            prompt,
            images,
            audio,
            args,
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


def _count_thinking_tag_tokens(text: str) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(text: str) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content)."""
    # Handle <|channel>thought...<channel|> format (gemma4)
    # Also handle partial tag: text starting with "thought\n" (continuation)
    if "<|channel>thought" in text or (
        "<channel|>" in text and text.lstrip().startswith("thought")
    ):
        parts = text.split("<channel|>", 1)
        if len(parts) == 2:
            reasoning = (
                parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
            )
            content = parts[1].strip()
            return reasoning or None, content
        reasoning = parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
        return reasoning or None, ""
    # Handle <think>...</think> format (qwen3.5 etc)
    # Also handle partial: output starts with thinking text + </think> (no opening tag)
    if "<think>" in text or "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
            return reasoning or None, content
        return parts[0].replace("<think>", "").strip(), ""
    return None, text


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


@asynccontextmanager
async def lifespan(app):
    model_path = os.environ.pop("MLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        adapter_path = os.environ.pop("MLX_VLM_PRELOAD_ADAPTER", None)
        logger.info("Pre-loading model: %s", model_path)
        get_cached_model(model_path, adapter_path)
        kv_bits = os.environ.get("KV_BITS")
        kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
        if kv_bits:
            logger.info("KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme)
        logger.info("Model ready, continuous batching enabled.")
    yield


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
    if adapter_path is _INHERIT_ADAPTER:
        cached = runtime.model_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path, model_kind)

    # Return from cache if already loaded and matches the requested paths
    if runtime.model_cache.get("cache_key") == cache_key:
        print(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return (
            runtime.model_cache["model"],
            runtime.model_cache["processor"],
            runtime.model_cache["config"],
        )

    # If cache exists but doesn't match, clear it
    if runtime.model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    load_as_edit = model_kind == "image_edit"
    if load_as_edit:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image edit models.",
            )
        print(f"Loading image edit model from: {model_path}")
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
        runtime.response_generator = None
        runtime.apc_manager = None
        runtime.model_cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_edit",
            "generation_lock": Lock(),
        }
        return model, None, config

    load_as_image = model_kind == "image_generation" or (
        model_kind == "auto" and is_image_generation_model(model_path)
    )
    if load_as_image:
        if adapter_path is not None:
            raise HTTPException(
                status_code=400,
                detail="Adapters are not supported for image generation models.",
            )
        print(f"Loading image generation model from: {model_path}")
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
        runtime.response_generator = None
        runtime.apc_manager = None
        runtime.model_cache = {
            "cache_key": cache_key,
            "model_path": model_path,
            "adapter_path": None,
            "model": model,
            "processor": None,
            "config": config,
            "model_kind": "image_generation",
            "generation_lock": Lock(),
        }
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

    runtime.response_generator = ResponseGenerator(
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
        model, processor, config = runtime.response_generator.wait_until_ready()
    except Exception:
        runtime.response_generator.stop_and_join()
        runtime.response_generator = None
        vision_cache.clear()
        raise

    runtime.model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    if not runtime.model_cache:
        return False

    print(
        f"Unloading model: {runtime.model_cache.get('model_path')}, Adapter: {runtime.model_cache.get('adapter_path')}"
    )

    # Stop the ResponseGenerator if running
    if runtime.response_generator is not None:
        print("Stopping ResponseGenerator...")
        runtime.response_generator.stop_and_join()
        runtime.response_generator = None

    # Drop APC blocks for the previous model
    if runtime.apc_manager is not None:
        runtime.apc_manager.clear()
        runtime.apc_manager = None

    # Clear vision cache before dropping references
    if "vision_cache" in runtime.model_cache:
        runtime.model_cache["vision_cache"].clear()
    runtime.model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


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
)
register_anthropic_routes(app, _protocol_deps)
register_openai_routes(app, _protocol_deps)


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

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = _server_package_attr("scan_cache_dir", scan_cache_dir)()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"mlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    runtime = _server_runtime_snapshot()
    return {
        "status": "healthy",
        "loaded_model": runtime["loaded_model"],
        "loaded_adapter": runtime["loaded_adapter"],
        "loaded_context_size": runtime["loaded_context_size"],
        "configured_context_limit": runtime["configured_context_limit"],
        "effective_context_limit": runtime["effective_context_limit"],
        "loaded_tool_parser": runtime["loaded_tool_parser"],
        "continuous_batching_enabled": runtime["continuous_batching_enabled"],
        "apc_enabled": runtime["apc"]["enabled"],
    }


@app.get("/metrics")
@app.get("/v1/metrics", include_in_schema=False)
async def metrics_endpoint():
    payload = runtime.metrics.snapshot()
    payload["server"] = _server_runtime_snapshot()
    return payload


@app.get("/v1/cache/stats")
@app.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats():
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    if runtime.apc_manager is None:
        return {"enabled": False}
    snap = runtime.apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@app.post("/v1/cache/reset")
@app.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset():
    if runtime.apc_manager is None:
        return {"enabled": False}
    runtime.apc_manager.clear()
    return {"enabled": True, "status": "cleared"}


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": runtime.model_cache.get("model_path", None),
        "adapter_name": runtime.model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
