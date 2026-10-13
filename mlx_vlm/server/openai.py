import asyncio
import base64
import binascii
import copy
import gc
import json
import logging
import random
import time
import uuid
from contextlib import aclosing
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ..generate import generate, stream_generate
from ..generate.edit_image import ImageEditRequest as CoreImageEditRequest
from ..generate.edit_image import edit_image
from ..generate.image import ImageGenerationRequest as CoreImageGenerationRequest
from ..generate.image import generate_image, parse_size
from ..generate.video import resolve_video_inputs
from ..prompt_utils import apply_chat_template
from ..tools import (
    _infer_tool_parser_from_processor,
    _prepare_chat_tool_choice,
    load_tool_module,
    process_tool_calls,
)
from ..utils import prepare_inputs
from . import compaction
from .compacting_generation import CompactingGeneration
from .generation import (
    GenerationMetrics,
    PromptTooLongError,
    _build_metrics_envelope,
    _count_prompt_tokens,
)
from .request_normalization import (
    _chat_message_to_prompt,
    _normalize_instruction_messages,
)
from .responses_state import (
    ToolCallStreamState,
    _normalize_response_input,
    _response_chain_items,
    _response_items_to_chat,
    _response_output_items_from_text,
    _response_tool_registry,
    _sequence_response_events,
)
from .responses_state import _sse_event as _response_sse_event
from .responses_state import (
    _store_response,
    finish_content_streams,
    make_response_stream_state,
    prompt_has_open_thinking,
    response_store,
    response_store_lock,
    strip_protocol_markers,
)
from .runtime import runtime
from .schemas import (
    ChatChoice,
    ChatLogprobs,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
    CompactRequest,
    ContentPartOutputText,
    GenerationTimings,
    ImageEditRequest,
    ImageEditResponse,
    ImageEditResponseData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationResponseData,
    InputAudio,
    OpenAIRequest,
    OpenAIResponse,
    OpenAIUsage,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputTextDoneEvent,
    StreamingTimings,
    UsageStats,
)

logger = logging.getLogger("mlx_vlm.server")

_INHERIT_ADAPTER = None
get_cached_model = None
_build_gen_args = None
_read_tenant_id = None
_preflight_stream_context_budget = None
_split_thinking = None
_count_thinking_tag_tokens = None
_make_logprob_content = None
_AUDIO_REFERENCE_PREFIXES = ("http://", "https://", "file://", "/", "./", "../")
_AUDIO_REFERENCE_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm")
_MISSING_INPUT_DETAIL = (
    "Request must include at least one non-empty message content or media input."
)


def _has_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _content_has_effective_input(content: Any) -> bool:
    if _has_non_empty_text(content):
        return True
    if content is None:
        return False
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in ("text", "input_text", "output_text"):
                if _has_non_empty_text(item.get("text")) or _has_non_empty_text(
                    item.get("content")
                ):
                    return True
            elif item_type in ("image_url", "input_image"):
                image = item.get("image_url") or item.get("file_id")
                if isinstance(image, dict):
                    image = image.get("url")
                if image:
                    return True
            elif item_type == "input_audio":
                input_audio = item.get("input_audio")
                if isinstance(input_audio, dict) and input_audio.get("data"):
                    return True
            elif _content_has_effective_input(item.get("content")):
                return True
        return False
    if isinstance(content, dict):
        return _content_has_effective_input([content])
    return bool(str(content).strip())


def _message_has_effective_input(message: Any) -> bool:
    if hasattr(message, "model_dump"):
        message = message.model_dump(exclude_none=True)
    if not isinstance(message, dict):
        return False
    return (
        _content_has_effective_input(message.get("content"))
        or bool(message.get("tool_calls"))
        or _has_non_empty_text(message.get("reasoning_content"))
        or _has_non_empty_text(message.get("reasoning"))
    )


def _ensure_effective_input(messages, *, images=None, audio=None):
    if any(image for image in (images or [])) or any(item for item in (audio or [])):
        return
    if any(_message_has_effective_input(message) for message in messages or []):
        return
    raise HTTPException(status_code=400, detail=_MISSING_INPUT_DETAIL)


def _runtime_cache_get(key, default=None, *, kind=None):
    cache = runtime.model_cache
    try:
        return cache.get(key, default, kind=kind)
    except TypeError:
        return cache.get(key, default)


def _looks_like_audio_reference(value: str) -> bool:
    return value.startswith(_AUDIO_REFERENCE_PREFIXES) or value.lower().endswith(
        _AUDIO_REFERENCE_SUFFIXES
    )


def _adapter_path_or_inherit(request):
    return (
        request.adapter_path
        if "adapter_path" in request.model_fields_set
        else _INHERIT_ADAPTER
    )


def _decode_input_audio_data(input_audio: InputAudio):
    data = input_audio["data"]
    if not isinstance(data, str):
        return data

    stripped = data.strip()
    if stripped.startswith("data:"):
        prefix, separator, encoded = stripped.partition(",")
        if (
            separator == ","
            and ";base64" in prefix
            and prefix.startswith("data:audio/")
        ):
            try:
                return BytesIO(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="input_audio data URI is not valid base64 audio",
                ) from exc
        return data

    if _looks_like_audio_reference(stripped):
        return data

    try:
        return BytesIO(base64.b64decode(stripped, validate=True))
    except (binascii.Error, ValueError):
        return data


def _extract_video_reference(item):
    item_type = item.get("type")
    if item_type == "video":
        return item.get("video")
    if item_type == "input_video":
        video = item.get("video") or item.get("video_url")
    elif item_type == "video_url":
        video = item.get("video_url")
    else:
        return None
    return video.get("url") if isinstance(video, dict) else video


def _final_chat_chunk(
    request_id: str,
    model: str,
    finish_reason: str,
    predicted_per_second: Optional[float] = None,
) -> ChatStreamChunk:
    return ChatStreamChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
        timings=StreamingTimings(predicted_per_second=predicted_per_second),
        choices=[
            ChatStreamChoice(
                finish_reason=finish_reason,
                delta=ChatMessage(role="assistant"),
            )
        ],
    )


def _chat_usage_chunk(
    request_id: str,
    model: str,
    metrics: GenerationMetrics,
    prompt_tokens: int,
    output_tokens: int,
) -> ChatStreamChunk:
    return ChatStreamChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
        usage=UsageStats.from_metrics(metrics, prompt_tokens, output_tokens),
        choices=[],
        timings=GenerationTimings.from_metrics(metrics, prompt_tokens, output_tokens),
    )


def register_routes(app, deps):
    global _INHERIT_ADAPTER
    global get_cached_model, _build_gen_args, _read_tenant_id
    global _preflight_stream_context_budget, _split_thinking
    global _count_thinking_tag_tokens, _make_logprob_content
    global generate, stream_generate, apply_chat_template
    global _infer_tool_parser_from_processor, load_tool_module

    _INHERIT_ADAPTER = deps.INHERIT_ADAPTER
    get_cached_model = deps.get_cached_model
    generate = deps.generate
    stream_generate = deps.stream_generate
    apply_chat_template = deps.apply_chat_template
    _infer_tool_parser_from_processor = deps.infer_tool_parser_from_processor
    load_tool_module = deps.load_tool_module
    _build_gen_args = deps.build_gen_args
    _read_tenant_id = deps.read_tenant_id
    _preflight_stream_context_budget = deps.preflight_stream_context_budget
    _split_thinking = deps.split_thinking
    _count_thinking_tag_tokens = deps.count_thinking_tag_tokens
    _make_logprob_content = deps.make_logprob_content

    app.post("/responses/input_tokens")(responses_input_tokens_endpoint)
    app.post("/responses/compact")(responses_compact_endpoint)
    app.post("/v1/responses/compact", include_in_schema=False)(
        responses_compact_endpoint
    )
    app.post("/v1/responses/input_tokens", include_in_schema=False)(
        responses_input_tokens_endpoint
    )
    app.get("/responses/{response_id}")(responses_retrieve_endpoint)
    app.get("/v1/responses/{response_id}", include_in_schema=False)(
        responses_retrieve_endpoint
    )
    app.delete("/responses/{response_id}")(responses_delete_endpoint)
    app.delete("/v1/responses/{response_id}", include_in_schema=False)(
        responses_delete_endpoint
    )
    app.post("/responses/{response_id}/cancel")(responses_cancel_endpoint)
    app.post("/v1/responses/{response_id}/cancel", include_in_schema=False)(
        responses_cancel_endpoint
    )
    app.get("/responses/{response_id}/input_items")(responses_input_items_endpoint)
    app.get("/v1/responses/{response_id}/input_items", include_in_schema=False)(
        responses_input_items_endpoint
    )
    app.post("/responses")(responses_endpoint)
    app.post("/v1/responses", include_in_schema=False)(responses_endpoint)
    app.post("/chat/completions", response_model=None)(chat_completions_endpoint)
    app.post("/v1/chat/completions", response_model=None, include_in_schema=False)(
        chat_completions_endpoint
    )
    app.post("/images/generations", response_model=ImageGenerationResponse)(
        images_generations_endpoint
    )
    app.post(
        "/v1/images/generations",
        response_model=ImageGenerationResponse,
        include_in_schema=False,
    )(images_generations_endpoint)
    app.post("/images/edits", response_model=ImageEditResponse)(images_edits_endpoint)
    app.post(
        "/v1/images/edits",
        response_model=ImageEditResponse,
        include_in_schema=False,
    )(images_edits_endpoint)


# OpenAI compatile endpoints


def _resolve_image_size(image_request: ImageGenerationRequest) -> Tuple[int, int]:
    if image_request.width is not None or image_request.height is not None:
        if image_request.width is None or image_request.height is None:
            raise HTTPException(
                status_code=400,
                detail="Both width and height are required when either is set.",
            )
        return image_request.width, image_request.height
    try:
        return parse_size(image_request.size or "512x512")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _resolve_optional_image_size(
    image_request: ImageEditRequest,
) -> Tuple[int | None, int | None]:
    if image_request.width is not None or image_request.height is not None:
        if image_request.width is None or image_request.height is None:
            raise HTTPException(
                status_code=400,
                detail="Both width and height are required when either is set.",
            )
        return image_request.width, image_request.height
    if image_request.size is None:
        return None, None
    try:
        return parse_size(image_request.size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _indexed_output_path(path: Path, index: int, count: int) -> Path:
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    if count <= 1:
        return path
    return path.with_name(f"{path.stem}-{index + 1:02d}{path.suffix}")


def _image_output_path(
    image_request: ImageGenerationRequest,
    *,
    index: int,
    count: int,
    seed: int,
) -> Path | None:
    if image_request.output_path:
        return _indexed_output_path(
            Path(image_request.output_path).expanduser(), index, count
        )
    if image_request.output_dir:
        directory = Path(image_request.output_dir).expanduser()
        return directory / f"image-{seed}.png"
    if image_request.response_format == "path":
        return Path("outputs") / f"image-{seed}.png"
    return None


def _image_edit_paths(image_request: ImageEditRequest) -> tuple[str, ...]:
    if isinstance(image_request.image, str):
        return (image_request.image,)
    return tuple(image_request.image)


def _image_edit_output_path(
    image_request: ImageEditRequest,
    *,
    index: int,
    count: int,
    seed: int,
) -> Path | None:
    if image_request.output_path:
        return _indexed_output_path(
            Path(image_request.output_path).expanduser(), index, count
        )
    if image_request.output_dir:
        directory = Path(image_request.output_dir).expanduser()
        return directory / f"edit-{seed}.png"
    if image_request.response_format == "path":
        return Path("outputs") / f"edit-{seed}.png"
    return None


async def images_generations_endpoint(request: Request):
    request_start = time.perf_counter()
    body = await request.json()
    image_request = ImageGenerationRequest(**body)
    if not image_request.prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")

    width, height = _resolve_image_size(image_request)
    created = int(time.time())
    base_seed = (
        int(image_request.seed)
        if image_request.seed is not None
        else random.randrange(2**32)
    )

    runtime.metrics.begin_request(
        endpoint="/v1/images/generations",
        model=image_request.model,
        stream=False,
    )
    try:
        model, _, _ = get_cached_model(
            image_request.model, model_kind="image_generation"
        )
        generation_lock = _runtime_cache_get("generation_lock", kind="image_generation")

        def _generate_all():
            results = []
            lock = generation_lock
            if lock is None:

                class _NullLock:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                lock = _NullLock()
            with lock:
                for index in range(image_request.n):
                    seed = base_seed + index
                    output_path = _image_output_path(
                        image_request,
                        index=index,
                        count=image_request.n,
                        seed=seed,
                    )
                    extra = {}
                    if image_request.auto_json_caption is not None:
                        extra["auto_json_caption"] = image_request.auto_json_caption
                    if image_request.prompt_expansion_model is not None:
                        extra["prompt_expansion_model"] = (
                            image_request.prompt_expansion_model
                        )
                    core_request = CoreImageGenerationRequest(
                        prompt=image_request.prompt,
                        seed=seed,
                        steps=image_request.steps,
                        width=width,
                        height=height,
                        guidance=image_request.guidance,
                        output_format=image_request.output_format,
                        extra=extra,
                    )
                    result = generate_image(
                        model,
                        core_request,
                        output_path=output_path,
                    )
                    results.append(result)
            return results

        results = _generate_all()
        data = []
        for result in results:
            item = ImageGenerationResponseData(
                width=result.width,
                height=result.height,
                seed=result.seed,
                path=str(result.path) if result.path is not None else None,
                revised_prompt=result.metadata.get("revised_prompt"),
            )
            if image_request.response_format == "b64_json":
                item.b64_json = result.to_b64_json()
            data.append(item)

        elapsed = time.perf_counter() - request_start
        prompt_tokens = results[0].prompt_tokens if results else 0
        peak_memory = max((r.peak_memory for r in results), default=0.0)
        envelope = _build_metrics_envelope(
            endpoint="/v1/images/generations",
            model=image_request.model,
            stream=False,
            backend="image_generation",
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=0,
            generated_tokens=0,
            request_elapsed_s=elapsed,
            request_started_s=request_start,
            peak_memory_gb=peak_memory or None,
            finish_reason="stop",
            image_count=len(data),
        )
        runtime.metrics.record_success(envelope)
        return ImageGenerationResponse(
            created=created,
            data=data,
            output_format=image_request.output_format,
            size=f"{width}x{height}",
        )
    except HTTPException:
        runtime.metrics.record_failure(
            endpoint="/v1/images/generations",
            model=image_request.model,
            stream=False,
            error="http_exception",
        )
        raise
    except Exception as e:
        runtime.metrics.record_failure(
            endpoint="/v1/images/generations",
            model=image_request.model,
            stream=False,
            error=str(e),
        )
        logger.exception("Image generation failed: %s", e)
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")


async def images_edits_endpoint(request: Request):
    request_start = time.perf_counter()
    body = await request.json()
    image_request = ImageEditRequest(**body)
    if not image_request.prompt:
        raise HTTPException(status_code=400, detail="Missing prompt.")

    width, height = _resolve_optional_image_size(image_request)
    image_paths = _image_edit_paths(image_request)
    created = int(time.time())
    base_seed = (
        int(image_request.seed)
        if image_request.seed is not None
        else random.randrange(2**32)
    )

    runtime.metrics.begin_request(
        endpoint="/v1/images/edits",
        model=image_request.model,
        stream=False,
    )
    try:
        model, _, _ = get_cached_model(image_request.model, model_kind="image_edit")
        generation_lock = _runtime_cache_get("generation_lock", kind="image_edit")

        def _generate_all():
            results = []
            lock = generation_lock
            if lock is None:

                class _NullLock:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                lock = _NullLock()
            with lock:
                for index in range(image_request.n):
                    seed = base_seed + index
                    output_path = _image_edit_output_path(
                        image_request,
                        index=index,
                        count=image_request.n,
                        seed=seed,
                    )
                    core_request = CoreImageEditRequest(
                        prompt=image_request.prompt,
                        image_paths=image_paths,
                        seed=seed,
                        steps=image_request.steps,
                        width=width,
                        height=height,
                        guidance=image_request.guidance,
                        output_format=image_request.output_format,
                        extra={
                            key: value
                            for key in (
                                "negative_prompt",
                                "output_resolution",
                                "use_kv_cache",
                            )
                            if (value := getattr(image_request, key)) is not None
                        },
                    )
                    result = edit_image(
                        model,
                        core_request,
                        output_path=output_path,
                    )
                    results.append(result)
            return results

        results = _generate_all()
        data = []
        for result in results:
            item = ImageEditResponseData(
                width=result.width,
                height=result.height,
                seed=result.seed,
                path=str(result.path) if result.path is not None else None,
            )
            if image_request.response_format == "b64_json":
                item.b64_json = result.to_b64_json()
            data.append(item)

        elapsed = time.perf_counter() - request_start
        prompt_tokens = results[0].prompt_tokens if results else 0
        peak_memory = max((r.peak_memory for r in results), default=0.0)
        envelope = _build_metrics_envelope(
            endpoint="/v1/images/edits",
            model=image_request.model,
            stream=False,
            backend="image_edit",
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=0,
            generated_tokens=0,
            request_elapsed_s=elapsed,
            request_started_s=request_start,
            peak_memory_gb=peak_memory or None,
            finish_reason="stop",
            image_count=len(data),
        )
        runtime.metrics.record_success(envelope)
        response_width = results[0].width if results else width or 0
        response_height = results[0].height if results else height or 0
        return ImageEditResponse(
            created=created,
            data=data,
            output_format=image_request.output_format,
            size=f"{response_width}x{response_height}",
        )
    except HTTPException:
        runtime.metrics.record_failure(
            endpoint="/v1/images/edits",
            model=image_request.model,
            stream=False,
            error="http_exception",
        )
        raise
    except Exception as e:
        runtime.metrics.record_failure(
            endpoint="/v1/images/edits",
            model=image_request.model,
            stream=False,
            error=str(e),
        )
        logger.exception("Image edit failed: %s", e)
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Image edit failed: {e}")


def _parse_response_request(body):
    try:
        return OpenAIRequest(**body)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from exc


async def responses_input_tokens_endpoint(request: Request):
    body = await request.json()
    openai_request = _parse_response_request(body)
    try:
        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        prompt_items = compaction.resolve(
            prompt_items, model=openai_request.model, tenant=_read_tenant_id(request)
        )
        prompt_items, _ = compaction.split_trigger(prompt_items)
        chat_messages, images = _response_items_to_chat(prompt_items)
        _normalize_instruction_messages(
            chat_messages,
            openai_request.instructions,
        )
        _ensure_effective_input(chat_messages, images=images)

        model, processor, config = get_cached_model(
            openai_request.model, _adapter_path_or_inherit(openai_request)
        )
        del model
        chat_tools, _ = _response_tool_registry(openai_request.tools)
        gen_args = _build_gen_args(
            openai_request, processor, tenant_id=_read_tenant_id(request)
        )
        template_kwargs = gen_args.to_template_kwargs()
        if openai_request.tool_choice is not None:
            template_kwargs["tool_choice"] = openai_request.tool_choice
        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            tools=chat_tools or None,
            **template_kwargs,
        )
        if runtime.response_generator is not None:
            raw_inputs = await asyncio.to_thread(
                runtime.response_generator._cpu_preprocess,
                formatted_prompt,
                images if images else None,
                None,
            )
        else:
            image_token_index = getattr(config, "image_token_index", None)
            raw_inputs = prepare_inputs(
                processor,
                images=images if images else None,
                prompts=formatted_prompt,
                image_token_index=image_token_index,
            )
        return {"input_tokens": _count_prompt_tokens(raw_inputs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def responses_compact_endpoint(http_request: Request, request: CompactRequest):
    tenant = _read_tenant_id(http_request)
    context = compaction.resolve_context(
        _response_chain_items(request.previous_response_id)
        + _normalize_response_input(request.input),
        model=request.model,
        tenant=tenant,
    )
    model, processor, config = get_cached_model(
        request.model, _adapter_path_or_inherit(request)
    )
    result = await compaction.compact_response_context(
        request,
        context.items,
        model,
        processor,
        config,
        tenant,
        build_gen_args=_build_gen_args,
        apply_chat_template=apply_chat_template,
        generate=generate,
        covered=context.covered,
    )
    output = (
        [
            compaction.seal(
                result.items, model=request.model, tenant=tenant, covered=result.covered
            )
        ]
        if result.changed or context.covered
        else context.items
    )
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response.compaction",
        "created_at": int(time.time()),
        "output": output,
        "usage": (
            result.usage or OpenAIUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        ).model_dump(),
    }


def _response_request_metadata(request):
    tools = []
    for tool in request.tools or []:
        tool = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        if tool.get("type") == "function":
            tool = {
                "type": "function",
                **tool.get("function", {k: v for k, v in tool.items() if k != "type"}),
            }
            tool.setdefault("strict", None)
        tools.append(tool)
    choice = request.tool_choice or "auto"
    if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
        choice = {"type": "function", "name": choice["function"]["name"]}
    return {
        "tools": tools,
        "tool_choice": choice,
        "parallel_tool_calls": getattr(request, "parallel_tool_calls", True),
    }


def _response_usage(metrics, input_tokens, output_tokens, processor, reasoning):
    tokenizer = getattr(processor, "tokenizer", processor)
    encode = getattr(tokenizer, "encode", None)
    reasoning_tokens = 0
    if reasoning and callable(encode):
        reasoning_tokens = min(
            output_tokens, len(encode(reasoning, add_special_tokens=False))
        )
    return OpenAIUsage.from_metrics(
        metrics, input_tokens, output_tokens, reasoning_tokens=reasoning_tokens
    )


def _response_stream_error(exc, response):
    status = exc.status_code if isinstance(exc, HTTPException) else 500
    code = (
        "context_length_exceeded"
        if isinstance(exc, (compaction.ContextBudgetError, PromptTooLongError))
        else (
            "rate_limit_exceeded"
            if status == 429
            else ("invalid_prompt" if status < 500 else "server_error")
        )
    )
    return _response_sse_event(
        "response.failed",
        {
            "type": "response.failed",
            "response": {
                **response.model_dump(),
                "status": "failed",
                "error": {
                    "message": (
                        str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                    ),
                    "code": code,
                },
            },
        },
    )


async def _responses_compaction_trigger(request, items, tenant, covered=frozenset()):
    """Return a single opaque item, with progress while a summary is running."""
    model, processor, config = get_cached_model(
        request.model, _adapter_path_or_inherit(request)
    )
    compact_request = CompactRequest(
        **{**request.model_dump(), "stream": False, "max_output_tokens": 1024}
    )
    args = (compact_request, items, model, processor, config, tenant)
    options = dict(
        build_gen_args=_build_gen_args,
        apply_chat_template=apply_chat_template,
        generate=generate,
        covered=covered,
        allow_oversized=True,
    )
    pending = OpenAIResponse(
        **_response_request_metadata(request),
        id=f"resp_{uuid.uuid4().hex}",
        created_at=int(time.time()),
        object="response",
        status="in_progress",
        model=request.model,
        output=[],
        output_text="",
        usage=OpenAIUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        store=request.store,
        previous_response_id=request.previous_response_id,
    )

    def complete(result):
        capsule = compaction.seal(
            result.items, model=request.model, tenant=tenant, covered=result.covered
        )
        response = pending.model_copy(
            update={
                "status": "completed",
                "output": [capsule],
                "usage": result.usage or pending.usage,
            }
        )
        _store_response(response, items, [capsule], request.previous_response_id)
        return response

    if not request.stream:
        return complete(await compaction.compact_response_context(*args, **options))

    async def events():
        item_id = None
        try:
            for kind in ("response.created", "response.in_progress"):
                yield _response_sse_event(
                    kind, {"type": kind, "response": pending.model_dump()}
                )
            async with aclosing(
                compaction.stream_response_context(*args, **options)
            ) as progress:
                async for update in progress:
                    if isinstance(update, compaction.CompactedContext):
                        result = update
                    elif update["type"] == "mlx.compaction.started":
                        item_id = f"cmp_{uuid.uuid4().hex}"
                        yield _response_sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": {
                                    "type": "compaction",
                                    "id": item_id,
                                    "encrypted_content": "",
                                },
                            },
                        )
            response = complete(result)
            if item_id:
                response.output[0]["id"] = item_id
                _store_response(
                    response, items, response.output, request.previous_response_id
                )
            else:
                yield _response_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": response.output[0],
                    },
                )
            yield _response_sse_event(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": response.output[0],
                },
            )
            yield _response_sse_event(
                "response.completed",
                {"type": "response.completed", "response": response.model_dump()},
            )
        except Exception as exc:
            yield _response_stream_error(exc, pending)

    return StreamingResponse(
        _sequence_response_events(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def responses_retrieve_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    return stored.response


async def responses_delete_endpoint(response_id: str):
    with response_store_lock:
        existed = response_store.pop(response_id, None) is not None
    if not existed:
        raise HTTPException(status_code=404, detail="Response not found.")
    return {"id": response_id, "object": "response.deleted", "deleted": True}


async def responses_cancel_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    response = dict(stored.response)
    if response.get("status") == "in_progress":
        response["status"] = "cancelled"
    return response


async def responses_input_items_endpoint(response_id: str):
    with response_store_lock:
        stored = response_store.get(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    data = stored.input_items
    return {
        "object": "list",
        "data": data,
        "first_id": data[0].get("id") if data else None,
        "last_id": data[-1].get("id") if data else None,
        "has_more": False,
    }


async def _compacting_responses(request, items, current_input, tenant, covered):
    """Share one generation/compaction loop between streamed and JSON responses."""
    model, processor, config = get_cached_model(
        request.model, _adapter_path_or_inherit(request)
    )
    args = _build_gen_args(request, processor, tenant_id=tenant)
    chat_tools, registry = _response_tool_registry(request.tools)
    parser_type = _infer_tool_parser_from_processor(
        processor, override=request.tool_parser
    )
    tool_module = load_tool_module(parser_type) if parser_type and chat_tools else None
    if tool_module:
        args.skip_special_tokens = False
    options = args.to_template_kwargs()
    if request.tool_choice is not None:
        options["tool_choice"] = request.tool_choice

    def render(context):
        messages, images = _response_items_to_chat(context)
        _normalize_instruction_messages(messages, request.instructions)
        _ensure_effective_input(messages, images=images)
        return (
            apply_chat_template(
                processor,
                config,
                messages,
                num_images=len(images),
                tools=chat_tools or None,
                **options,
            ),
            images,
        )

    prompt, _ = render(items)
    starts_thinking = prompt_has_open_thinking(
        prompt,
        args.enable_thinking,
        args.thinking_start_token,
        args.thinking_end_token,
    )
    session = CompactingGeneration(
        request,
        items,
        model,
        processor,
        config,
        tenant,
        args,
        render=render,
        build_gen_args=_build_gen_args,
        apply_chat_template=apply_chat_template,
        generate=generate,
        stream_generate=stream_generate,
        prepare_inputs=prepare_inputs,
        covered=covered,
        tool_module=tool_module,
    )
    pending = OpenAIResponse(
        **_response_request_metadata(request),
        id=f"resp_{uuid.uuid4().hex}",
        object="response",
        created_at=int(time.time()),
        status="in_progress",
        model=request.model,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        output=[],
        output_text="",
        temperature=request.temperature,
        top_p=request.top_p,
        previous_response_id=request.previous_response_id,
        store=request.store,
        usage=None,
    )
    output = []
    active = {}
    pending_whitespace = {}
    shown = set()
    thinking = make_response_stream_state(
        processor,
        starts_thinking,
        args.thinking_start_token,
        args.thinking_end_token,
    )
    tools_state = ToolCallStreamState(
        getattr(tool_module, "tool_call_start", None),
        getattr(tool_module, "tool_call_end", None),
    )

    def emit(kind, **payload):
        return _response_sse_event(kind, {"type": kind, **payload})

    def delta(kind, text):
        if not text:
            return
        text = pending_whitespace.get(kind, "") + text
        if kind not in shown:
            text = text.lstrip()
        body = text.rstrip()
        pending_whitespace[kind] = text[len(body) :]
        if not body:
            return
        shown.add(kind)
        text = body
        if kind not in active:
            index = len(output)
            item = {
                "id": f"{'msg' if kind == 'message' else 'rs'}_{uuid.uuid4().hex}",
                "type": kind,
                "status": "in_progress",
                "content": [],
            }
            if kind == "message":
                item["role"] = "assistant"
                part = {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": [],
                }
            else:
                item["summary"] = []
                part = {"type": "reasoning_text", "text": ""}
            output.append(item)
            active[kind] = index
            yield emit(
                "response.output_item.added",
                output_index=index,
                item=copy.deepcopy(item),
            )
            item["content"].append(part)
            if kind == "message":
                yield emit(
                    "response.content_part.added",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    part=copy.deepcopy(part),
                )
        index = active[kind]
        item = output[index]
        item["content"][0]["text"] += text
        event_kind = "output_text" if kind == "message" else "reasoning_text"
        yield emit(
            f"response.{event_kind}.delta",
            item_id=item["id"],
            output_index=index,
            content_index=0,
            delta=text,
            **({"logprobs": []} if kind == "message" else {}),
        )

    def close_items(incomplete=False):
        for kind, index in sorted(active.items(), key=lambda pair: pair[1]):
            item = output[index]
            item["status"] = "incomplete" if incomplete else "completed"
            part = item["content"][0]
            event_kind = "output_text" if kind == "message" else "reasoning_text"
            yield emit(
                f"response.{event_kind}.done",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                text=part["text"],
                **({"logprobs": []} if kind == "message" else {}),
            )
            if kind == "message":
                yield emit(
                    "response.content_part.done",
                    item_id=item["id"],
                    output_index=index,
                    content_index=0,
                    part=part,
                )
            yield emit("response.output_item.done", output_index=index, item=item)
        active.clear()

    async def events():
        started = time.perf_counter()
        finalized = False
        runtime.metrics.begin_request(
            endpoint="/responses", model=request.model, stream=request.stream
        )
        try:
            yield emit("response.created", response=pending.model_dump())
            yield emit("response.in_progress", response=pending.model_dump())
            async with aclosing(session.tokens()) as tokens:
                async for token in tokens:
                    if isinstance(token, dict):
                        if token["kind"] == "compaction_start":
                            for event in close_items():
                                yield event
                            output.append(token["item"])
                            yield emit(
                                "response.output_item.added",
                                output_index=len(output) - 1,
                                item=token["item"],
                            )
                        else:
                            item = token["item"]
                            index = next(
                                (
                                    i
                                    for i, x in enumerate(output)
                                    if x["id"] == item["id"]
                                ),
                                None,
                            )
                            if index is None:
                                index = len(output)
                                output.append(item)
                                yield emit(
                                    "response.output_item.added",
                                    output_index=index,
                                    item=item,
                                )
                            output[index] = item
                            yield emit(
                                "response.output_item.done",
                                output_index=index,
                                item=item,
                            )
                        continue
                    parts = thinking.feed(token.text, last=bool(token.finish_reason))
                    for event in delta("reasoning", parts.reasoning):
                        yield event
                    content = tools_state.feed(
                        parts.content, last=bool(token.finish_reason)
                    )
                    for event in delta("message", content):
                        yield event
            reasoning, content = finish_content_streams(thinking, tools_state)
            for kind, text in (("reasoning", reasoning), ("message", content)):
                for event in delta(kind, text):
                    yield event
            for event in close_items(session.finish_reason == "length"):
                yield event
            parsed, _, _, _ = _response_output_items_from_text(
                session.full_text,
                "unused",
                tool_module,
                chat_tools,
                registry,
                args.thinking_start_token,
                args.thinking_end_token,
                processor=processor,
                starts_in_thinking=starts_thinking,
            )
            for item in parsed:
                if item["type"] in ("message", "reasoning"):
                    continue
                index = len(output)
                output.append(item)
                yield emit(
                    "response.output_item.added",
                    output_index=index,
                    item={**item, "status": "in_progress", "arguments": ""},
                )
                if item["type"] == "function_call":
                    yield emit(
                        "response.function_call_arguments.delta",
                        item_id=item["id"],
                        output_index=index,
                        delta=item.get("arguments", ""),
                    )
                    yield emit(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=index,
                        arguments=item.get("arguments", ""),
                    )
                yield emit("response.output_item.done", output_index=index, item=item)
            text = "".join(
                part["text"]
                for item in output
                if item["type"] == "message"
                for part in item["content"]
            )
            response = pending.model_copy(
                update={
                    "status": (
                        "incomplete"
                        if session.finish_reason == "length"
                        else "completed"
                    ),
                    "incomplete_details": (
                        {"reason": "max_output_tokens"}
                        if session.finish_reason == "length"
                        else None
                    ),
                    "output": output,
                    "output_text": text,
                    "usage": _response_usage(
                        session.metrics,
                        session.input_tokens,
                        session.output_tokens,
                        processor,
                        "".join(
                            part["text"]
                            for item in output
                            if item["type"] == "reasoning"
                            for part in item["content"]
                        ),
                    ),
                }
            )
            _store_response(
                response, current_input, output, request.previous_response_id
            )
            runtime.metrics.record_success(
                _build_metrics_envelope(
                    endpoint="/responses",
                    model=request.model,
                    stream=request.stream,
                    backend=(
                        "continuous_batching"
                        if runtime.response_generator
                        else "generate"
                    ),
                    prompt_tokens=session.input_tokens,
                    completion_tokens=session.output_tokens,
                    generated_tokens=session.output_tokens,
                    request_elapsed_s=time.perf_counter() - started,
                    request_started_s=started,
                    token_times=session.metrics.token_times,
                    prompt_tps=session.metrics.prompt_tps,
                    generation_tps=session.metrics.generation_tps,
                    peak_memory_gb=session.metrics.peak_memory,
                    finish_reason=session.finish_reason,
                )
            )
            finalized = True
            yield emit(f"response.{response.status}", response=response.model_dump())
        except Exception as exc:
            if not request.stream:
                raise
            yield _response_stream_error(exc, pending)
        finally:
            if not finalized:
                runtime.metrics.record_failure(
                    endpoint="/responses",
                    model=request.model,
                    stream=request.stream,
                    error="generation_failed_or_cancelled",
                )

    if request.stream:
        return StreamingResponse(
            _sequence_response_events(events()),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    async with aclosing(events()) as stream:
        async for event in stream:
            payload = json.loads(event.split("data: ", 1)[1])
            if payload["type"] in ("response.completed", "response.incomplete"):
                return OpenAIResponse.model_validate(payload["response"])


async def responses_endpoint(request: Request):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    request_start = time.perf_counter()
    body = await request.json()
    openai_request = _parse_response_request(body)

    try:
        kwargs = {}

        if openai_request.input is None:
            logger.warning("Responses request is missing input.")
            raise HTTPException(status_code=400, detail="Missing input.")

        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        tenant = _read_tenant_id(request)
        context = compaction.resolve_context(
            prompt_items, model=openai_request.model, tenant=tenant
        )
        prompt_items, triggered = compaction.split_trigger(context.items)
        if triggered:
            return await _responses_compaction_trigger(
                openai_request, prompt_items, tenant, context.covered
            )
        if openai_request.context_management:
            return await _compacting_responses(
                openai_request,
                prompt_items,
                current_input_items,
                tenant,
                context.covered,
            )
        chat_messages, images = _response_items_to_chat(prompt_items)
        instructions = _normalize_instruction_messages(
            chat_messages,
            openai_request.instructions,
        )
        _ensure_effective_input(chat_messages, images=images)

        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(
            openai_request.model, _adapter_path_or_inherit(openai_request)
        )

        chat_tools, tool_registry = _response_tool_registry(openai_request.tools)
        tool_parser_type = _infer_tool_parser_from_processor(
            processor, override=openai_request.tool_parser
        )
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                openai_request, processor, tenant_id=_read_tenant_id(request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if chat_tools and tool_module is not None:
            gen_args.skip_special_tokens = False

        template_kwargs = gen_args.to_template_kwargs()
        if openai_request.tool_choice is not None:
            template_kwargs["tool_choice"] = openai_request.tool_choice

        def render_prompt(items):
            messages, images = _response_items_to_chat(items)
            _normalize_instruction_messages(messages, openai_request.instructions)
            _ensure_effective_input(messages, images=images)
            prompt = apply_chat_template(
                processor,
                config,
                messages,
                num_images=len(images),
                tools=chat_tools or None,
                **template_kwargs,
            )
            thinking = prompt_has_open_thinking(
                prompt,
                gen_args.enable_thinking,
                gen_args.thinking_start_token,
                gen_args.thinking_end_token,
            )
            return prompt, images, thinking

        formatted_prompt, images, starts_in_thinking = render_prompt(prompt_items)

        logger.debug(
            "responses request: model=%s images=%d max_tokens=%s temp=%s stream=%s",
            openai_request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            openai_request.stream,
        )

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            runtime.metrics.begin_request(
                endpoint="/responses",
                model=openai_request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/responses",
                model=openai_request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                metrics_finalized = False
                metrics = GenerationMetrics()
                finish_reason = None
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        **_response_request_metadata(openai_request),
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        previous_response_id=openai_request.previous_response_id,
                        store=openai_request.store,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    output_indices = {}
                    pending_whitespace = {}

                    def start_output_item(item):
                        item_id = item["id"]
                        if item_id in output_indices:
                            return
                        index = len(output_indices)
                        output_indices[item_id] = index
                        pending = {**item, "status": "in_progress"}
                        if item["type"] == "message":
                            pending["content"] = []
                        elif item["type"] == "reasoning":
                            pending["summary"] = []
                        yield _response_sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": index,
                                "item": pending,
                            },
                        )
                        if item["type"] == "message":
                            part = ContentPartOutputText(
                                type="output_text", text="", annotations=[]
                            )
                            yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=item_id, output_index=index, content_index=0, part=part).model_dump_json()}\n\n"

                    def stream_delta(kind, delta, rate):
                        if kind == "reasoning":
                            item = {
                                "id": reasoning_item_id,
                                "type": "reasoning",
                                "summary": [],
                            }
                            event_type = "response.reasoning_text.delta"
                        else:
                            item = {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                            }
                            event_type = "response.output_text.delta"
                        text = pending_whitespace.get(kind, "") + delta
                        if item["id"] not in output_indices:
                            text = text.lstrip()
                        delta = text.rstrip()
                        pending_whitespace[kind] = text[len(delta) :]
                        if not delta:
                            return
                        yield from start_output_item(item)
                        yield _response_sse_event(
                            event_type,
                            {
                                "type": event_type,
                                "item_id": item["id"],
                                "output_index": output_indices[item["id"]],
                                "content_index": 0,
                                "delta": delta,
                                "timings": {"predicted_per_second": rate},
                                **(
                                    {"response_id": response_id}
                                    if kind == "reasoning"
                                    else {}
                                ),
                            },
                        )

                    # Stream text deltas using ResponseGenerator (continuous batching)
                    full_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}
                    tc_start = (
                        tool_module.tool_call_start
                        if tool_module is not None and chat_tools
                        else None
                    )
                    tc_end = (
                        tool_module.tool_call_end
                        if tool_module is not None and chat_tools
                        else None
                    )
                    tool_call_state = ToolCallStreamState(tc_start, tc_end)
                    thinking_state = make_response_stream_state(
                        processor,
                        starts_in_thinking,
                        gen_args.thinking_start_token,
                        gen_args.thinking_end_token,
                    )
                    reasoning_item_id = f"rs_{uuid.uuid4().hex}"
                    streamed_reasoning = ""

                    if runtime.response_generator is not None:
                        # generate() blocks on _cpu_preprocess + queue.get;
                        # offload so concurrent handlers preprocess in parallel.
                        ctx, token_iter = await asyncio.to_thread(
                            runtime.response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            None,  # audio
                            gen_args,
                        )
                        usage_stats["input_tokens"] = ctx.prompt_tokens

                        output_tokens = 0

                        def _next_token_resp_stream():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token_resp_stream)
                            if token is None:
                                break
                            output_tokens += getattr(token, "token_count", 1)
                            raw_delta = token.text
                            full_text += raw_delta
                            chunk_rate = metrics.record_chunk(token)
                            thinking_delta = thinking_state.feed(
                                raw_delta, last=bool(token.finish_reason)
                            )
                            if thinking_delta.reasoning:
                                streamed_reasoning += thinking_delta.reasoning
                                for event in stream_delta(
                                    "reasoning", thinking_delta.reasoning, chunk_rate
                                ):
                                    yield event
                            delta = thinking_delta.content
                            delta = tool_call_state.feed(
                                delta, last=bool(token.finish_reason)
                            )
                            usage_stats = {
                                "input_tokens": ctx.prompt_tokens,
                                "output_tokens": output_tokens,
                            }

                            if delta:
                                for event in stream_delta("message", delta, chunk_rate):
                                    yield event
                                await asyncio.sleep(0.01)

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            vision_cache=runtime.model_cache.get("vision_cache"),
                            apc_manager=runtime.apc_manager,
                            **gen_args.to_generate_kwargs(),
                            **kwargs,
                        )

                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            raw_delta = chunk.text
                            full_text += raw_delta
                            chunk_rate = metrics.record_chunk(chunk)
                            chunk_finish = getattr(chunk, "finish_reason", None)
                            thinking_delta = thinking_state.feed(
                                raw_delta, last=bool(chunk_finish)
                            )
                            if thinking_delta.reasoning:
                                streamed_reasoning += thinking_delta.reasoning
                                for event in stream_delta(
                                    "reasoning", thinking_delta.reasoning, chunk_rate
                                ):
                                    yield event
                            delta = thinking_delta.content
                            delta = tool_call_state.feed(delta, last=bool(chunk_finish))
                            if chunk_finish is not None:
                                finish_reason = chunk_finish
                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                            }

                            if delta:
                                for event in stream_delta("message", delta, chunk_rate):
                                    yield event
                                await asyncio.sleep(0.01)

                    tail_reasoning, tail = finish_content_streams(
                        thinking_state, tool_call_state
                    )
                    if tail_reasoning:
                        streamed_reasoning += tail_reasoning
                        for event in stream_delta(
                            "reasoning", tail_reasoning, metrics.rate
                        ):
                            yield event
                    if tail:
                        for event in stream_delta("message", tail, metrics.rate):
                            yield event

                    output_items, clean_text, _, output_finish_reason = (
                        _response_output_items_from_text(
                            full_text,
                            message_id,
                            tool_module,
                            chat_tools,
                            tool_registry,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                            reasoning_item_id,
                            processor=processor,
                            starts_in_thinking=starts_in_thinking,
                        )
                    )
                    if clean_text and not any(
                        item["type"] == "message" for item in output_items
                    ):
                        output_items.insert(
                            sum(item["type"] == "reasoning" for item in output_items),
                            {
                                "id": message_id,
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": clean_text,
                                        "annotations": [],
                                    }
                                ],
                            },
                        )
                    completed_output = []
                    for item in sorted(
                        output_items,
                        key=lambda item: output_indices.get(item["id"], float("inf")),
                    ):
                        for event in start_output_item(item):
                            yield event
                        output_index = output_indices[item["id"]]
                        completed_output.append(item)
                        if item["type"] == "reasoning" and streamed_reasoning:
                            yield _response_sse_event(
                                "response.reasoning_text.done",
                                {
                                    "type": "response.reasoning_text.done",
                                    "response_id": response_id,
                                    "item_id": item["id"],
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "text": streamed_reasoning.strip(),
                                },
                            )
                        elif item["type"] == "message":
                            yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=output_index, content_index=0, text=clean_text, timings=StreamingTimings(predicted_per_second=metrics.rate)).model_dump_json()}\n\n"
                            final_content_part = ContentPartOutputText(
                                type="output_text", text=clean_text, annotations=[]
                            )
                            yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=output_index, content_index=0, part=final_content_part).model_dump_json()}\n\n"
                        elif item["type"] == "function_call":
                            yield _response_sse_event(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "response_id": response_id,
                                    "item_id": item["id"],
                                    "output_index": output_index,
                                    "call_id": item.get("call_id"),
                                    "name": item.get("name"),
                                    "arguments": item.get("arguments") or "{}",
                                    "item": item,
                                },
                            )
                        yield _response_sse_event(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_index,
                                "item": item,
                            },
                        )

                    finish_reason = (
                        "length"
                        if finish_reason == "length"
                        else (
                            "tool_calls"
                            if output_finish_reason == "tool_calls"
                            else finish_reason or "stop"
                        )
                    )
                    envelope = _build_metrics_envelope(
                        endpoint="/responses",
                        model=openai_request.model,
                        stream=True,
                        backend=(
                            "continuous_batching"
                            if runtime.response_generator is not None
                            else "generate"
                        ),
                        prompt_tokens=usage_stats["input_tokens"],
                        completion_tokens=usage_stats["output_tokens"],
                        generated_tokens=usage_stats["output_tokens"],
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=metrics.token_times,
                        prompt_tps=metrics.prompt_tps,
                        generation_tps=metrics.generation_tps,
                        peak_memory_gb=metrics.peak_memory or None,
                        finish_reason=finish_reason,
                        image_count=len(images),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                    )
                    runtime.metrics.record_success(envelope)
                    metrics_finalized = True
                    completed_response = base_response.model_copy(
                        update={
                            "status": (
                                "incomplete"
                                if finish_reason == "length"
                                else "completed"
                            ),
                            "incomplete_details": (
                                {"reason": "max_output_tokens"}
                                if finish_reason == "length"
                                else None
                            ),
                            "output": completed_output,
                            "output_text": clean_text,
                            "usage": _response_usage(
                                metrics,
                                usage_stats["input_tokens"],
                                usage_stats["output_tokens"],
                                processor,
                                streamed_reasoning,
                            ),
                        }
                    )
                    _store_response(
                        completed_response,
                        current_input_items,
                        completed_output,
                        openai_request.previous_response_id,
                    )
                    terminal_event = f"response.{completed_response.status}"
                    yield _response_sse_event(
                        terminal_event,
                        {
                            "type": terminal_event,
                            "response": completed_response.model_dump(),
                        },
                    )

                except Exception as e:
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/responses",
                            model=openai_request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    logger.exception("Responses stream generation failed: %s", e)
                    yield _response_stream_error(e, base_response)

                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/responses",
                            model=openai_request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    logger.debug("Responses stream closed.")

            return StreamingResponse(
                _sequence_response_events(stream_generator()),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            runtime.metrics.begin_request(
                endpoint="/responses",
                model=openai_request.model,
                stream=False,
            )
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                metrics = GenerationMetrics()
                finish_reason = None

                if runtime.response_generator is not None:

                    def _blocking_resp():
                        metrics = GenerationMetrics()
                        ctx_, ti = runtime.response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            args=gen_args,
                        )
                        text = ""
                        ot = 0
                        fr = None
                        for tok in ti:
                            text += tok.text
                            ot += 1
                            metrics.record_chunk(tok)
                            if tok.finish_reason:
                                fr = tok.finish_reason
                                break
                        try:
                            ti.close()
                        except Exception:
                            pass
                        return ctx_.prompt_tokens, text, ot, fr, metrics

                    (
                        prompt_tokens,
                        full_text,
                        output_tokens,
                        finish_reason,
                        metrics,
                    ) = await asyncio.to_thread(_blocking_resp)
                else:
                    result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=runtime.model_cache.get("vision_cache"),
                        apc_manager=runtime.apc_manager,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = result.text
                    prompt_tokens = result.prompt_tokens
                    output_tokens = result.generation_tokens
                    metrics.record_result(result)
                    finish_reason = getattr(result, "finish_reason", None) or "stop"

                mx.clear_cache()
                gc.collect()

                output_items, content, reasoning, output_finish_reason = (
                    _response_output_items_from_text(
                        full_text,
                        message_id,
                        tool_module,
                        chat_tools,
                        tool_registry,
                        gen_args.thinking_start_token,
                        gen_args.thinking_end_token,
                        processor=processor,
                        starts_in_thinking=starts_in_thinking,
                    )
                )
                if finish_reason != "length" and output_finish_reason == "tool_calls":
                    finish_reason = "tool_calls"

                response = OpenAIResponse(
                    **_response_request_metadata(openai_request),
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="incomplete" if finish_reason == "length" else "completed",
                    incomplete_details=(
                        {"reason": "max_output_tokens"}
                        if finish_reason == "length"
                        else None
                    ),
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=output_items,
                    output_text=content,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    previous_response_id=openai_request.previous_response_id,
                    store=openai_request.store,
                    usage=_response_usage(
                        metrics, prompt_tokens, output_tokens, processor, reasoning
                    ),
                )
                _store_response(
                    response,
                    current_input_items,
                    output_items,
                    openai_request.previous_response_id,
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "responses done: prompt_tokens=%d output_tokens=%d "
                    "total_time=%.2fs",
                    prompt_tokens,
                    output_tokens,
                    elapsed,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                envelope = _build_metrics_envelope(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    backend=(
                        "continuous_batching"
                        if runtime.response_generator is not None
                        else "generate"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=output_tokens,
                    generated_tokens=output_tokens,
                    request_elapsed_s=elapsed,
                    request_started_s=request_start,
                    token_times=metrics.token_times,
                    prompt_tps=metrics.prompt_tps,
                    generation_tps=metrics.generation_tps,
                    peak_memory_gb=metrics.peak_memory or None,
                    finish_reason=finish_reason,
                    image_count=len(images),
                    structured_output=bool(gen_args.logits_processors),
                    thinking_enabled=bool(gen_args.enable_thinking),
                )
                runtime.metrics.record_success(envelope)

                return response

            except PromptTooLongError as e:
                runtime.metrics.record_failure(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    error=str(e),
                )
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                runtime.metrics.record_failure(
                    endpoint="/responses",
                    model=openai_request.model,
                    stream=False,
                    error=str(e),
                )
                logger.exception("Responses generation failed: %s", e)
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception("Unexpected error in /responses endpoint: %s", e)
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


async def _compacting_chat(request, http_request, model, processor, config):
    tenant = _read_tenant_id(http_request)
    args = _build_gen_args(request, processor, tenant_id=tenant)
    parser_type = _infer_tool_parser_from_processor(
        processor, override=request.tool_parser
    )
    tool_module = (
        load_tool_module(parser_type) if parser_type and request.tools else None
    )
    if tool_module:
        args.skip_special_tokens = False
    items = [
        {**message.model_dump(exclude_none=True), "type": "message"}
        for message in request.messages
    ]

    def render(context):
        messages = [_chat_message_to_prompt(item) for item in context]
        _, images = _response_items_to_chat(
            [item for item in context if item.get("role") == "user"]
        )
        _normalize_instruction_messages(messages)
        _ensure_effective_input(messages, images=images)
        messages, tools, choice = _prepare_chat_tool_choice(
            messages, request.tools, request.tool_choice
        )
        options = args.to_template_kwargs()
        if choice is not None:
            options["tool_choice"] = choice
        return (
            apply_chat_template(
                processor,
                config,
                messages,
                num_images=len(images),
                tools=tools,
                **options,
            ),
            images,
        )

    initial_prompt, _ = render(items)
    thinking = make_response_stream_state(
        processor,
        prompt_has_open_thinking(
            initial_prompt,
            args.enable_thinking,
            args.thinking_start_token,
            args.thinking_end_token,
        ),
        args.thinking_start_token,
        args.thinking_end_token,
    )
    tool_state = ToolCallStreamState(
        getattr(tool_module, "tool_call_start", None),
        getattr(tool_module, "tool_call_end", None),
    )
    generate_kwargs = {}
    if request.resize_shape is not None:
        if len(request.resize_shape) not in (1, 2):
            raise HTTPException(
                400, "resize_shape must contain exactly two integers (height, width)"
            )
        generate_kwargs["resize_shape"] = tuple(request.resize_shape) * (
            2 if len(request.resize_shape) == 1 else 1
        )
    session = CompactingGeneration(
        request,
        items,
        model,
        processor,
        config,
        tenant,
        args,
        render=render,
        build_gen_args=_build_gen_args,
        apply_chat_template=apply_chat_template,
        generate=generate,
        stream_generate=stream_generate,
        prepare_inputs=prepare_inputs,
        tool_module=tool_module,
        generate_kwargs=generate_kwargs,
    )
    async for _ in session.compact():
        pass
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    text, reasoning, logprobs = [], [], []

    def chunk(delta, finish=None, logprobs=None):
        return ChatStreamChunk(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatStreamChoice(delta=delta, finish_reason=finish, logprobs=logprobs)
            ],
        )

    async def events():
        started, finalized = time.perf_counter(), False
        runtime.metrics.begin_request(
            endpoint="/chat/completions", model=request.model, stream=request.stream
        )
        try:
            async with aclosing(session.tokens()) as tokens:
                async for token in tokens:
                    if isinstance(token, dict):
                        continue
                    parts = thinking.feed(token.text, last=bool(token.finish_reason))
                    content = tool_state.feed(
                        parts.content, last=bool(token.finish_reason)
                    )
                    text.append(content or "")
                    reasoning.append(parts.reasoning or "")
                    token_logprobs = None
                    if (
                        request.logprobs
                        and runtime.response_generator
                        and token.finish_reason != "stop"
                    ):
                        entry = _make_logprob_content(
                            runtime.response_generator.tokenizer,
                            token.token,
                            token.logprobs,
                            top_logprobs=getattr(token, "top_logprobs", None),
                            top_k=int(request.top_logprobs or 0),
                        )
                        logprobs.append(entry)
                        token_logprobs = ChatLogprobs(content=[entry])
                    if content or parts.reasoning or token_logprobs:
                        yield chunk(
                            ChatMessage(
                                role="assistant",
                                content=content,
                                reasoning=parts.reasoning,
                            ),
                            logprobs=token_logprobs,
                        )
            tail_reasoning, tail = finish_content_streams(thinking, tool_state)
            text.append(tail or "")
            reasoning.append(tail_reasoning or "")
            if tail or tail_reasoning:
                yield chunk(
                    ChatMessage(
                        role="assistant", content=tail, reasoning=tail_reasoning
                    )
                )
            calls = (
                process_tool_calls(session.full_text, tool_module, request.tools)
                if tool_module
                else None
            )
            finish = session.finish_reason or "stop"
            if calls and calls.calls:
                finish = "length" if finish == "length" else "tool_calls"
                yield chunk(
                    ChatMessage(role="assistant", tool_calls=calls.calls), finish
                )
            else:
                yield _final_chat_chunk(
                    request_id, request.model, finish, session.metrics.rate
                )
            session.finish_reason = finish
            session.tool_calls = calls.calls if calls else None
            runtime.metrics.record_success(
                _build_metrics_envelope(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=request.stream,
                    backend=(
                        "continuous_batching"
                        if runtime.response_generator
                        else "generate"
                    ),
                    prompt_tokens=session.input_tokens,
                    completion_tokens=session.output_tokens,
                    generated_tokens=session.output_tokens,
                    request_elapsed_s=time.perf_counter() - started,
                    request_started_s=started,
                    token_times=session.metrics.token_times,
                    prompt_tps=session.metrics.prompt_tps,
                    generation_tps=session.metrics.generation_tps,
                    peak_memory_gb=session.metrics.peak_memory,
                    finish_reason=finish,
                )
            )
            finalized = True
        finally:
            if not finalized:
                runtime.metrics.record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=request.stream,
                    error="generation_failed_or_cancelled",
                )

    if request.stream:

        async def stream():
            try:
                async with aclosing(events()) as updates:
                    async for item in updates:
                        yield f"data: {item.model_dump_json()}\n\n"
                if request.stream_options and request.stream_options.include_usage:
                    usage = _chat_usage_chunk(
                        request_id,
                        request.model,
                        session.metrics,
                        session.input_tokens,
                        session.output_tokens,
                    )
                    yield f"data: {usage.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
    async with aclosing(events()) as updates:
        async for _ in updates:
            pass
    return ChatResponse(
        id=request_id,
        created=created,
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content="".join(text).strip(),
                    reasoning="".join(reasoning).strip() or None,
                    tool_calls=session.tool_calls,
                ),
                finish_reason=session.finish_reason,
                logprobs=ChatLogprobs(content=logprobs) if logprobs else None,
            )
        ],
        usage=UsageStats(
            prompt_tokens=session.input_tokens,
            completion_tokens=session.output_tokens,
            total_tokens=session.input_tokens + session.output_tokens,
        ),
    )


async def chat_completions_endpoint(request: ChatRequest, http_request: Request):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    request_start = time.perf_counter()
    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )

        if request.context_management != []:
            _prepare_chat_tool_choice([], request.tools, request.tool_choice)
            model, processor, config = get_cached_model(request.model, adapter_path)
            if request.context_management or (
                compaction._context_limit(config, required=False) is not None
                and all(
                    compaction.supports_content(message.content)
                    for message in request.messages
                )
            ):
                return await _compacting_chat(
                    request, http_request, model, processor, config
                )

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        images = []
        audio = []
        videos = []
        processed_messages = []
        for message in request.messages:
            if isinstance(message.content, list):
                if message.role == "user":
                    for item in message.content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "input_image":
                            images.append(item["image_url"])
                        elif item_type == "image_url":
                            images.append(item["image_url"]["url"])
                        elif item_type == "input_audio":
                            audio.append(_decode_input_audio_data(item["input_audio"]))
                        elif item_type in ("input_video", "video_url", "video"):
                            video = _extract_video_reference(item)
                            if video:
                                videos.append(video)
            processed_messages.append(
                _chat_message_to_prompt(message.model_dump(exclude_none=True))
            )

        _normalize_instruction_messages(processed_messages)
        _ensure_effective_input(processed_messages, images=images, audio=audio)

        processed_messages, tools, tool_choice = _prepare_chat_tool_choice(
            processed_messages,
            request.tools,
            request.tool_choice,
        )

        model, processor, config = get_cached_model(request.model, adapter_path)

        video_resolution = resolve_video_inputs(
            processor,
            videos,
            images=images,
            fps=2.0,
            max_frames=16,
        )
        images, videos = video_resolution.images, video_resolution.videos
        if video_resolution.used_fallback:
            logger.info(
                "Processor %s has no native video support; sending %d of %d "
                "sampled frames as ordered images.",
                processor.__class__.__name__,
                video_resolution.selected_count,
                video_resolution.sampled_count,
            )

        # Detect tool parser from chat template
        tool_parser_type = _infer_tool_parser_from_processor(
            processor, override=request.tool_parser
        )
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None
        if not tools:
            tool_module = None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if tools and tool_module is not None:
            gen_args.skip_special_tokens = False

        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            video=videos or None,
            tools=tools,
            **template_kwargs,
        )

        logger.debug(
            "chat/completions request: model=%s images=%d audio=%d videos=%d "
            "max_tokens=%s temp=%s stream=%s",
            request.model,
            len(images),
            len(audio),
            len(videos),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
        )

        if request.stream:
            # Streaming response using ResponseGenerator for continuous batching
            runtime.metrics.begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/chat/completions",
                model=request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=audio if audio else None,
                videos=videos if videos else None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                metrics_finalized = False
                metrics = GenerationMetrics()
                finish_reason = None
                emit_usage = bool(
                    request.stream_options and request.stream_options.include_usage
                )
                try:
                    output_tokens = 0
                    full_output = ""
                    output_text = ""
                    stream_prompt_tokens = 0
                    tool_calls_made = False

                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if runtime.response_generator is not None:
                        # generate() does blocking Queue.get — run off event loop
                        generate_kwargs = {"args": gen_args}
                        if videos:
                            generate_kwargs["videos"] = videos
                        ctx, token_iter = await asyncio.to_thread(
                            runtime.response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            audio if audio else None,
                            **generate_kwargs,
                        )

                        output_tokens = 0
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        thinking_state = make_response_stream_state(
                            processor,
                            prompt_has_open_thinking(
                                formatted_prompt,
                                gen_args.enable_thinking,
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            ),
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        full_output = ""  # raw output for tool call parsing
                        # Track tool-call state to suppress markup from content
                        tc_start = tool_module.tool_call_start if tool_module else None
                        tc_end = tool_module.tool_call_end if tool_module else None
                        tool_call_state = ToolCallStreamState(tc_start, tc_end)

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += getattr(token, "token_count", 1)
                            full_output += token.text
                            chunk_rate = metrics.record_chunk(token)

                            # Detect thinking boundaries
                            thinking_delta = thinking_state.feed(
                                token.text, last=bool(token.finish_reason)
                            )
                            delta_reasoning = thinking_delta.reasoning
                            delta_content = thinking_delta.content

                            # Suppress tool-call markup from content
                            delta_content = tool_call_state.feed(
                                delta_content, last=bool(token.finish_reason)
                            )

                            chunk_logprobs = None
                            if request.logprobs and token.finish_reason != "stop":
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            runtime.response_generator.tokenizer,
                                            token.token,
                                            token.logprobs,
                                            top_logprobs=token.top_logprobs,
                                            top_k=req_top_k,
                                        )
                                    ]
                                )

                            # Skip empty deltas (e.g. suppressed tool-call tokens)
                            has_payload = (
                                bool(delta_content)
                                or bool(delta_reasoning)
                                or chunk_logprobs is not None
                            )
                            if has_payload:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=delta_reasoning,
                                        ),
                                        logprobs=chunk_logprobs,
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                    timings=StreamingTimings(
                                        predicted_per_second=chunk_rate
                                    ),
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break

                        tail_reasoning, tail = finish_content_streams(
                            thinking_state, tool_call_state
                        )
                        if tail or tail_reasoning:
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=tail,
                                            reasoning=tail_reasoning,
                                        )
                                    )
                                ],
                                timings=StreamingTimings(
                                    predicted_per_second=metrics.rate
                                ),
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                        # Parse tool calls from full output and emit final chunk
                        terminal_emitted = False
                        if tool_module is not None:
                            tc = process_tool_calls(full_output, tool_module, tools)
                            if tc.calls:
                                tool_calls_made = True
                                finish_reason = "tool_calls"
                                terminal_emitted = True
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc.calls,
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                    timings=StreamingTimings(
                                        predicted_per_second=metrics.rate
                                    ),
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if not terminal_emitted:
                            finish_reason = finish_reason or "stop"
                            chunk_data = _final_chat_chunk(
                                request_id,
                                request.model,
                                finish_reason,
                                metrics.rate,
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if emit_usage:
                            chunk_data = _chat_usage_chunk(
                                request_id,
                                request.model,
                                metrics,
                                ctx.prompt_tokens,
                                output_tokens,
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            audio=audio,
                            video=videos,
                            vision_cache=runtime.model_cache.get("vision_cache"),
                            apc_manager=runtime.apc_manager,
                            **gen_args.to_generate_kwargs(),
                            **kwargs,
                        )

                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        output_text = ""
                        thinking_state = make_response_stream_state(
                            processor,
                            prompt_has_open_thinking(
                                formatted_prompt,
                                gen_args.enable_thinking,
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            ),
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        tool_call_state = ToolCallStreamState(
                            tool_module.tool_call_start if tool_module else None,
                            tool_module.tool_call_end if tool_module else None,
                        )
                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            output_text += chunk.text
                            stream_prompt_tokens = chunk.prompt_tokens
                            output_tokens = chunk.generation_tokens
                            chunk_rate = metrics.record_chunk(chunk)
                            chunk_finish = getattr(chunk, "finish_reason", None)
                            if chunk_finish is not None:
                                finish_reason = chunk_finish

                            thinking_delta = thinking_state.feed(
                                chunk.text, last=bool(chunk_finish)
                            )
                            delta_content = tool_call_state.feed(
                                thinking_delta.content, last=bool(chunk_finish)
                            )
                            if delta_content or thinking_delta.reasoning:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=thinking_delta.reasoning,
                                        )
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                    timings=StreamingTimings(
                                        predicted_per_second=chunk_rate
                                    ),
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

                        tail_reasoning, tail = finish_content_streams(
                            thinking_state, tool_call_state
                        )
                        if tail or tail_reasoning:
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=tail,
                                            reasoning=tail_reasoning,
                                        )
                                    )
                                ],
                                timings=StreamingTimings(
                                    predicted_per_second=metrics.rate
                                ),
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                        tc = (
                            process_tool_calls(output_text, tool_module, tools)
                            if tool_module is not None
                            else None
                        )
                        if tc is not None and tc.calls:
                            tool_calls_made = True
                            finish_reason = "tool_calls"
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                choices=[
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc.calls,
                                        ),
                                    )
                                ],
                                timings=StreamingTimings(
                                    predicted_per_second=metrics.rate
                                ),
                            )
                        else:
                            finish_reason = finish_reason or "stop"
                            chunk_data = _final_chat_chunk(
                                request_id,
                                request.model,
                                finish_reason,
                                metrics.rate,
                            )
                        yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if emit_usage:
                            chunk_data = _chat_usage_chunk(
                                request_id,
                                request.model,
                                metrics,
                                stream_prompt_tokens,
                                output_tokens,
                            )
                            yield f"data: {chunk_data.model_dump_json()}\n\n"

                    metrics_text = full_output or output_text
                    completion_tokens = max(
                        0,
                        output_tokens
                        - _count_thinking_tag_tokens(
                            metrics_text,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        ),
                    )
                    envelope = _build_metrics_envelope(
                        endpoint="/chat/completions",
                        model=request.model,
                        stream=True,
                        backend=(
                            "continuous_batching"
                            if runtime.response_generator is not None
                            else "generate"
                        ),
                        prompt_tokens=(
                            ctx.prompt_tokens
                            if runtime.response_generator is not None
                            else stream_prompt_tokens
                        ),
                        completion_tokens=completion_tokens,
                        generated_tokens=output_tokens,
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=metrics.token_times,
                        prompt_tps=metrics.prompt_tps,
                        generation_tps=metrics.generation_tps,
                        peak_memory_gb=metrics.peak_memory or None,
                        finish_reason=finish_reason,
                        image_count=len(images),
                        audio_count=len(audio),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                        tool_parser=tool_parser_type,
                        tool_calls=tool_calls_made,
                    )
                    runtime.metrics.record_success(envelope)
                    metrics_finalized = True

                    # Signal stream end
                    yield "data: [DONE]\n\n"

                    elapsed = time.perf_counter() - request_start
                    logger.debug(
                        "chat/completions stream done: tokens=%d total_time=%.2fs",
                        output_tokens,
                        elapsed,
                    )

                except Exception as e:
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    logger.exception("Chat completion stream generation failed: %s", e)
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    # Close the token iterator to trigger cleanup (important for ResponseGenerator)
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/chat/completions",
                            model=request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    logger.debug("Chat completion stream closed.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            runtime.metrics.begin_request(
                endpoint="/chat/completions",
                model=request.model,
                stream=False,
            )
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                metrics = GenerationMetrics()
                finish_reason = None

                collected_logprobs: List[
                    Tuple[int, float, Optional[List[Tuple[int, float]]]]
                ] = []

                if runtime.response_generator is not None:

                    def _blocking_generate():
                        metrics = GenerationMetrics()
                        logprobs: List[
                            Tuple[int, float, Optional[List[Tuple[int, float]]]]
                        ] = []
                        text = ""
                        pt = gt = 0
                        fr = None
                        ctx, token_iter = runtime.response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            audio=audio if audio else None,
                            args=gen_args,
                            **({"videos": videos} if videos else {}),
                        )
                        pt = ctx.prompt_tokens
                        for token in token_iter:
                            text += token.text
                            gt += getattr(token, "token_count", 1)
                            metrics.record_chunk(token)
                            if request.logprobs and token.finish_reason != "stop":
                                logprobs.append(
                                    (token.token, token.logprobs, token.top_logprobs)
                                )
                            if token.finish_reason:
                                fr = token.finish_reason
                                break
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                        return pt, text, gt, fr, metrics, logprobs

                    (
                        prompt_tokens,
                        full_text,
                        output_tokens,
                        finish_reason,
                        metrics,
                        collected_logprobs,
                    ) = await asyncio.to_thread(_blocking_generate)
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        video=videos,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=runtime.model_cache.get("vision_cache"),
                        apc_manager=runtime.apc_manager,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = gen_result.text
                    prompt_tokens = gen_result.prompt_tokens
                    output_tokens = gen_result.generation_tokens
                    metrics.record_result(gen_result)
                    finish_reason = getattr(gen_result, "finish_reason", None) or "stop"

                mx.clear_cache()
                gc.collect()

                reasoning, content = _split_thinking(
                    full_text,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                    prompt_has_open_thinking(
                        formatted_prompt,
                        gen_args.enable_thinking,
                        gen_args.thinking_start_token,
                        gen_args.thinking_end_token,
                    ),
                    processor=processor,
                )

                # Count raw generated tokens minus thinking tag tokens
                completion_tokens = output_tokens - _count_thinking_tag_tokens(
                    full_text,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                )

                usage_stats = UsageStats.from_metrics(
                    metrics, prompt_tokens, completion_tokens
                )

                # Parse tool calls from generated output
                parsed_tool_calls = None
                if tool_module is not None:
                    tc = process_tool_calls(
                        model_output=full_text,
                        tool_module=tool_module,
                        tools=tools,
                    )
                    if tc.calls:
                        parsed_tool_calls = tc.calls
                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(
                            tc.remaining_text or "",
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        content = (
                            strip_protocol_markers(
                                clean_remaining,
                                tool_module,
                                gen_args.thinking_start_token,
                                gen_args.thinking_end_token,
                            )
                            or None
                        )

                response_logprobs = None
                if request.logprobs and collected_logprobs:
                    tokenizer = (
                        processor.tokenizer
                        if hasattr(processor, "tokenizer")
                        else processor
                    )
                    req_top_k = int(request.top_logprobs or 0)
                    response_logprobs = ChatLogprobs(
                        content=[
                            _make_logprob_content(
                                tokenizer,
                                tid,
                                lp,
                                top_logprobs=top_lps,
                                top_k=req_top_k,
                            )
                            for tid, lp, top_lps in collected_logprobs
                        ]
                    )

                choices = [
                    ChatChoice(
                        finish_reason=(
                            "tool_calls"
                            if parsed_tool_calls
                            else finish_reason or "stop"
                        ),
                        message=ChatMessage(
                            role="assistant",
                            content=content if content else None,
                            reasoning=reasoning,
                            tool_calls=parsed_tool_calls,
                        ),
                        logprobs=response_logprobs,
                    )
                ]
                result = ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    created=int(time.time()),
                    model=request.model,
                    usage=usage_stats,
                    choices=choices,
                    timings=GenerationTimings.from_metrics(
                        metrics, prompt_tokens, output_tokens
                    ),
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "chat/completions done: prompt_tokens=%d completion_tokens=%d "
                    "total_time=%.2fs peak_memory=%.2fGB",
                    prompt_tokens,
                    completion_tokens,
                    elapsed,
                    metrics.peak_memory,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                envelope = _build_metrics_envelope(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    backend=(
                        "continuous_batching"
                        if runtime.response_generator is not None
                        else "generate"
                    ),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    generated_tokens=output_tokens,
                    request_elapsed_s=elapsed,
                    request_started_s=request_start,
                    token_times=metrics.token_times,
                    prompt_tps=metrics.prompt_tps,
                    generation_tps=metrics.generation_tps,
                    peak_memory_gb=metrics.peak_memory or None,
                    finish_reason=(
                        "tool_calls" if parsed_tool_calls else finish_reason or "stop"
                    ),
                    image_count=len(images),
                    audio_count=len(audio),
                    structured_output=bool(gen_args.logits_processors),
                    thinking_enabled=bool(gen_args.enable_thinking),
                    tool_parser=tool_parser_type,
                    tool_calls=bool(parsed_tool_calls),
                )
                runtime.metrics.record_success(envelope)

                return result

            except PromptTooLongError as e:
                runtime.metrics.record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                runtime.metrics.record_failure(
                    endpoint="/chat/completions",
                    model=request.model,
                    stream=False,
                    error=str(e),
                )
                logger.exception("Chat completion generation failed: %s", e)
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        logger.exception("Unexpected error in /chat/completions endpoint: %s", e)
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
