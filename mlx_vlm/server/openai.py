import asyncio
import base64
import binascii
import gc
import json
import logging
import random
import re
import time
import traceback
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from ..generate import generate, stream_generate
from ..generate.edit_image import ImageEditRequest as CoreImageEditRequest
from ..generate.edit_image import edit_image
from ..generate.image import ImageGenerationRequest as CoreImageGenerationRequest
from ..generate.image import generate_image, parse_size
from ..prompt_utils import apply_chat_template, extract_text_from_content
from ..tool_parsers import _infer_tool_parser_from_processor, load_tool_module
from ..utils import prepare_inputs
from .generation import (
    GenerationMetrics,
    PromptTooLongError,
    _build_metrics_envelope,
    _count_prompt_tokens,
)
from .responses_state import (
    ThinkingStreamState,
    _normalize_response_input,
    _response_chain_items,
    _response_items_to_chat,
    _response_output_items_from_text,
    _response_tool_registry,
)
from .responses_state import _sse_event as _response_sse_event
from .responses_state import (
    _store_response,
    process_tool_calls,
    response_store,
    response_store_lock,
    suppress_tool_call_content,
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
    ContentPartOutputText,
    GenerationTimings,
    ImageEditRequest,
    ImageEditResponse,
    ImageEditResponseData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageGenerationResponseData,
    InputAudio,
    MessageItem,
    OpenAIRequest,
    OpenAIResponse,
    OpenAIUsage,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
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
) -> ChatStreamChunk:
    return ChatStreamChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
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
        traceback.print_exc()
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
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Image edit failed: {e}")


async def responses_input_tokens_endpoint(request: Request):
    body = await request.json()
    openai_request = OpenAIRequest(**body)
    try:
        model, processor, config = get_cached_model(
            openai_request.model, _adapter_path_or_inherit(openai_request)
        )
        del model
        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        chat_messages, images = _response_items_to_chat(prompt_items)
        if openai_request.instructions:
            chat_messages.insert(
                0, {"role": "system", "content": openai_request.instructions}
            )
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
    openai_request = OpenAIRequest(**body)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(
            openai_request.model, _adapter_path_or_inherit(openai_request)
        )

        kwargs = {}

        if openai_request.input is None:
            print("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        current_input_items = _normalize_response_input(openai_request.input)
        prompt_items = (
            _response_chain_items(openai_request.previous_response_id)
            + current_input_items
        )
        chat_messages, images = _response_items_to_chat(prompt_items)
        instructions = openai_request.instructions
        if instructions:
            chat_messages.insert(0, {"role": "system", "content": instructions})
        elif chat_messages and chat_messages[0].get("role") in ("system", "developer"):
            instructions = chat_messages[0].get("content")

        chat_tools, tool_registry = _response_tool_registry(openai_request.tools)
        tool_parser_type = _infer_tool_parser_from_processor(processor)
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

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            tools=chat_tools or None,
            **template_kwargs,
        )

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

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas using ResponseGenerator (continuous batching)
                    full_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}
                    in_tool_call = False
                    tc_start = (
                        tool_module.tool_call_start
                        if tool_module is not None and chat_tools
                        else None
                    )
                    thinking_state = ThinkingStreamState(
                        gen_args.enable_thinking,
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
                            thinking_delta = thinking_state.feed(raw_delta)
                            if thinking_delta.reasoning:
                                streamed_reasoning += thinking_delta.reasoning
                                yield _response_sse_event(
                                    "response.reasoning_text.delta",
                                    {
                                        "type": "response.reasoning_text.delta",
                                        "response_id": response_id,
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "delta": thinking_delta.reasoning,
                                    },
                                )
                            delta = thinking_delta.content
                            in_tool_call, delta = suppress_tool_call_content(
                                full_text, in_tool_call, tc_start, delta
                            )
                            metrics.record_chunk(token)
                            usage_stats = {
                                "input_tokens": ctx.prompt_tokens,
                                "output_tokens": output_tokens,
                            }

                            if delta:
                                yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
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
                            thinking_delta = thinking_state.feed(raw_delta)
                            if thinking_delta.reasoning:
                                streamed_reasoning += thinking_delta.reasoning
                                yield _response_sse_event(
                                    "response.reasoning_text.delta",
                                    {
                                        "type": "response.reasoning_text.delta",
                                        "response_id": response_id,
                                        "item_id": reasoning_item_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "delta": thinking_delta.reasoning,
                                    },
                                )
                            delta = thinking_delta.content
                            in_tool_call, delta = suppress_tool_call_content(
                                full_text, in_tool_call, tc_start, delta
                            )
                            metrics.record_chunk(chunk)
                            chunk_finish = getattr(chunk, "finish_reason", None)
                            if chunk_finish is not None:
                                finish_reason = chunk_finish
                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                            }

                            if delta:
                                yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

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
                        )
                    )
                    tool_output_items = [
                        item
                        for item in output_items
                        if item.get("type") not in ("message", "reasoning")
                    ]
                    reasoning_output_items = [
                        item for item in output_items if item.get("type") == "reasoning"
                    ]
                    if streamed_reasoning:
                        yield _response_sse_event(
                            "response.reasoning_text.done",
                            {
                                "type": "response.reasoning_text.done",
                                "response_id": response_id,
                                "item_id": reasoning_item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "text": streamed_reasoning,
                            },
                        )

                    # Send response.output_text.done event (to match the openai pipeline)
                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=clean_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=clean_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part] if clean_text else [],
                    )
                    message_output_items = [
                        item for item in output_items if item.get("type") == "message"
                    ]
                    final_message_payload = (
                        message_output_items[0]
                        if message_output_items
                        else final_message_item.model_dump()
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_payload).model_dump_json()}\n\n"

                    completed_output = []
                    completed_output.extend(reasoning_output_items)
                    if message_output_items:
                        completed_output.extend(message_output_items)
                    elif clean_text:
                        completed_output.append(final_message_item.model_dump())
                    tool_start_index = len(completed_output)
                    completed_output.extend(tool_output_items)
                    for output_index, tool_item in enumerate(
                        tool_output_items, start=tool_start_index
                    ):
                        yield _response_sse_event(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "output_index": output_index,
                                "item": tool_item,
                            },
                        )
                        if tool_item.get("type") == "function_call":
                            yield _response_sse_event(
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "response_id": response_id,
                                    "item_id": tool_item.get("id")
                                    or tool_item.get("call_id"),
                                    "output_index": output_index,
                                    "call_id": tool_item.get("call_id"),
                                    "name": tool_item.get("name"),
                                    "arguments": tool_item.get("arguments") or "{}",
                                    "item": tool_item,
                                },
                            )
                        yield _response_sse_event(
                            "response.output_item.done",
                            {
                                "type": "response.output_item.done",
                                "output_index": output_index,
                                "item": tool_item,
                            },
                        )

                    # Send response.completed event (to match the openai pipeline)
                    finish_reason = (
                        "tool_calls"
                        if output_finish_reason == "tool_calls"
                        else finish_reason or "stop"
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
                            "status": "completed",
                            "output": completed_output,
                            "output_text": clean_text,
                            "usage": OpenAIUsage.from_metrics(
                                metrics,
                                usage_stats["input_tokens"],
                                usage_stats["output_tokens"],
                            ),
                        }
                    )
                    _store_response(
                        completed_response,
                        current_input_items,
                        completed_output,
                        openai_request.previous_response_id,
                    )
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

                except Exception as e:
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/responses",
                            model=openai_request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

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
                    print("Stream finished.")

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
                    )
                )
                if output_finish_reason == "tool_calls":
                    finish_reason = "tool_calls"

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=output_items,
                    output_text=content,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    previous_response_id=openai_request.previous_response_id,
                    store=openai_request.store,
                    usage=OpenAIUsage.from_metrics(
                        metrics, prompt_tokens, output_tokens
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
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
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
        model, processor, config = get_cached_model(request.model, adapter_path)

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
            msg = {"role": message.role}

            if isinstance(message.content, str):
                msg["content"] = message.content
            elif isinstance(message.content, list):
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
                msg["content"] = extract_text_from_content(message.content)
            else:
                msg["content"] = message.content

            # Preserve tool-calling metadata.
            # Ensure arguments are dicts (not JSON strings) for Jinja templates
            # that iterate them with |items (e.g. Qwen3.5).
            if message.tool_calls is not None:
                normalized_calls = []
                for tc in message.tool_calls:
                    tc = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc, dict) and "function" in tc:
                        fn = dict(tc["function"])
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                fn["arguments"] = {}
                        tc["function"] = fn
                    normalized_calls.append(tc)
                msg["tool_calls"] = normalized_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                msg["name"] = message.name
            if message.reasoning_content is not None:
                msg["reasoning_content"] = message.reasoning_content
                msg["reasoning"] = message.reasoning_content

            processed_messages.append(msg)

        # Detect tool parser from chat template
        tools = getattr(request, "tools", None)
        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if tools and tool_module is not None:
            gen_args.skip_special_tokens = False

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            video=videos or None,
            tools=tools,
            **gen_args.to_template_kwargs(),
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
                        thinking_state = ThinkingStreamState(
                            gen_args.enable_thinking,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        full_output = ""  # raw output for tool call parsing
                        # Track tool-call state to suppress markup from content
                        in_tool_call = False
                        tc_start = tool_module.tool_call_start if tool_module else None
                        tc_end = tool_module.tool_call_end if tool_module else None

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
                            metrics.record_chunk(token)

                            # Detect thinking boundaries
                            thinking_delta = thinking_state.feed(token.text)
                            delta_reasoning = thinking_delta.reasoning
                            delta_content = thinking_delta.content

                            # Suppress tool-call markup from content
                            in_tool_call, delta_content = suppress_tool_call_content(
                                full_output, in_tool_call, tc_start, delta_content
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
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                finish_reason = token.finish_reason
                                break

                        # Parse tool calls from full output and emit final chunk
                        terminal_emitted = False
                        if tool_module is not None:
                            tc = process_tool_calls(full_output, tool_module, tools)
                            if tc["calls"]:
                                tool_calls_made = True
                                finish_reason = "tool_calls"
                                terminal_emitted = True
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc["calls"],
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                        if not terminal_emitted:
                            finish_reason = finish_reason or "stop"
                            chunk_data = _final_chat_chunk(
                                request_id,
                                request.model,
                                finish_reason,
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
                        thinking_state = ThinkingStreamState(
                            gen_args.enable_thinking,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            output_text += chunk.text
                            stream_prompt_tokens = chunk.prompt_tokens
                            output_tokens = chunk.generation_tokens
                            metrics.record_chunk(chunk)
                            chunk_finish = getattr(chunk, "finish_reason", None)
                            if chunk_finish is not None:
                                finish_reason = chunk_finish

                            thinking_delta = thinking_state.feed(chunk.text)
                            if thinking_delta.content or thinking_delta.reasoning:
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=thinking_delta.content,
                                            reasoning=thinking_delta.reasoning,
                                        )
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                                await asyncio.sleep(0.01)

                        finish_reason = finish_reason or "stop"
                        chunk_data = _final_chat_chunk(
                            request_id,
                            request.model,
                            finish_reason,
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
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
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
                    print("Stream finished.")

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
                    if tc["calls"]:
                        parsed_tool_calls = tc["calls"]
                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(
                            tc["remaining_text"] or "",
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        )
                        if clean_remaining:
                            # Strip model control tokens
                            clean_remaining = re.sub(
                                r"<\|[^>]+\|>|<[^>]+>", "", clean_remaining
                            ).strip()
                        content = clean_remaining or None

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
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
