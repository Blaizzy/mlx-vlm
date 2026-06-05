import asyncio
import gc
import json
import logging
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..generate import generate, stream_generate
from ..prompt_utils import apply_chat_template
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
    process_tool_calls,
    suppress_tool_call_content,
)
from .runtime import runtime
from .schemas import AnthropicMessageResponse, AnthropicRequest, AnthropicUsage

logger = logging.getLogger("mlx_vlm.server")

_INHERIT_ADAPTER = None
get_cached_model = None
_build_gen_args = None
_read_tenant_id = None
_preflight_stream_context_budget = None
_as_plain_dict = None
_split_thinking = None
_count_thinking_tag_tokens = None


def register_routes(app, deps):
    global _INHERIT_ADAPTER
    global get_cached_model, _build_gen_args, _read_tenant_id
    global _preflight_stream_context_budget, _as_plain_dict
    global _split_thinking, _count_thinking_tag_tokens
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
    _as_plain_dict = deps.as_plain_dict
    _split_thinking = deps.split_thinking
    _count_thinking_tag_tokens = deps.count_thinking_tag_tokens

    app.post("/messages")(anthropic_messages_endpoint)
    app.post("/v1/messages", include_in_schema=False)(anthropic_messages_endpoint)
    app.post("/messages/count_tokens")(anthropic_count_tokens_endpoint)
    app.post("/v1/messages/count_tokens", include_in_schema=False)(
        anthropic_count_tokens_endpoint
    )


def _anthropic_error_response(
    status_code: int, message: str, error_type: str = "invalid_request_error"
):
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
        headers={"request-id": f"req_{uuid.uuid4().hex}"},
    )


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _anthropic_system_text(system: Optional[Union[str, List[Any]]]) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = []
    for item in system:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if text:
                parts.append(str(text))
        elif item is not None:
            parts.append(str(item))
    return "\n".join(parts).strip() or None


def _normalize_anthropic_system_messages(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    normalized_messages = []
    system_parts = []
    saw_system_message = False
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            saw_system_message = True
            text = _anthropic_system_text(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        normalized_messages.append(message)

    if not saw_system_message:
        return body

    normalized_body = dict(body)
    normalized_body["messages"] = normalized_messages
    existing_system = _anthropic_system_text(normalized_body.get("system"))
    if existing_system:
        system_parts.insert(0, existing_system)
    if system_parts:
        normalized_body["system"] = "\n".join(system_parts)
    return normalized_body


def _anthropic_image_source_to_ref(source: Any) -> Optional[str]:
    source = _as_plain_dict(source)
    if not isinstance(source, dict):
        return None
    source_type = source.get("type")
    if source_type == "url":
        return source.get("url")
    if source_type == "base64":
        media_type = source.get("media_type") or "image/png"
        data = source.get("data")
        if data:
            return f"data:{media_type};base64,{data}"
    return None


def _anthropic_tool_result_content_to_openai(
    content: Any, images: Optional[List[str]] = None
) -> Union[str, List[Dict[str, Any]]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        content_parts: List[Dict[str, Any]] = []
        saw_image = False

        def append_text(text: Any) -> None:
            if text:
                text_parts.append(str(text))
                content_parts.append({"type": "text", "text": str(text)})

        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    append_text(item.get("text"))
                elif item_type == "document":
                    source = _as_plain_dict(item.get("source"))
                    if isinstance(source, dict) and source.get("type") == "text":
                        append_text(source.get("data"))
                elif item_type == "image":
                    image_ref = _anthropic_image_source_to_ref(item.get("source"))
                    if image_ref:
                        saw_image = True
                        if images is not None:
                            images.append(image_ref)
                        content_parts.append({"type": "image"})
                elif item.get("content"):
                    append_text(item["content"])
            elif item is not None:
                append_text(item)
        if saw_image:
            return content_parts
        return "\n".join(text_parts).strip()
    return str(content)


def _anthropic_tool_result_content_to_text(content: Any) -> str:
    converted = _anthropic_tool_result_content_to_openai(content)
    if isinstance(converted, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in converted
            if item.get("type") == "text"
        ).strip()
    return converted


def _anthropic_tool_to_openai(tool: Any) -> Optional[Dict[str, Any]]:
    tool = _as_plain_dict(tool)
    if not isinstance(tool, dict):
        return None
    name = tool.get("name")
    input_schema = tool.get("input_schema")
    if not name or input_schema is None:
        # Anthropic server tools (web_search, code_execution, etc.) cannot be
        # executed by this local server, so they are accepted but not surfaced
        # to model chat templates.
        return None
    function = {
        "name": name,
        "description": tool.get("description", ""),
        "parameters": input_schema,
    }
    if tool.get("strict") is not None:
        function["strict"] = tool.get("strict")
    return {"type": "function", "function": function}


def _anthropic_tools_to_openai(
    tools: Optional[List[Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    converted = []
    for tool in tools:
        converted_tool = _anthropic_tool_to_openai(tool)
        if converted_tool is not None:
            converted.append(converted_tool)
    return converted or None


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Optional[Any]:
    tool_choice = _as_plain_dict(tool_choice)
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "none"):
        return choice_type
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return None


def _anthropic_tool_use_to_openai(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
        },
    }


def _openai_tool_call_to_anthropic(call: Any) -> Dict[str, Any]:
    call = _as_plain_dict(call) or {}
    function = _as_plain_dict(call.get("function")) or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}
    return {
        "type": "tool_use",
        "id": call.get("id") or f"toolu_{uuid.uuid4().hex}",
        "name": function.get("name", ""),
        "input": arguments if isinstance(arguments, dict) else {},
    }


def _anthropic_content_blocks_to_text_and_tools(
    role: str,
    content: Union[str, List[Any]],
    images: List[str],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(content, str):
        return content, [], []

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for raw_item in content or []:
        item = _as_plain_dict(raw_item)
        if not isinstance(item, dict):
            if item is not None:
                text_parts.append(str(item))
            continue

        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if text:
                text_parts.append(str(text))
        elif item_type == "image" and role == "user":
            image_ref = _anthropic_image_source_to_ref(item.get("source"))
            if image_ref:
                images.append(image_ref)
        elif item_type == "tool_use" and role == "assistant":
            tool_calls.append(_anthropic_tool_use_to_openai(item))
        elif item_type == "tool_result" and role == "user":
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("tool_use_id"),
                    "content": _anthropic_tool_result_content_to_openai(
                        item.get("content"), images
                    ),
                    "name": item.get("name"),
                }
            )
        elif item_type in ("thinking", "redacted_thinking"):
            continue
        elif item.get("text"):
            text_parts.append(str(item["text"]))

    return "\n".join(text_parts).strip(), tool_calls, tool_results


def _anthropic_messages_to_internal(
    request: AnthropicRequest,
) -> Tuple[
    List[Dict[str, Any]], List[str], Optional[List[Dict[str, Any]]], Optional[Any]
]:
    images: List[str] = []
    processed_messages: List[Dict[str, Any]] = []

    system_text = _anthropic_system_text(request.system)
    if system_text:
        processed_messages.append({"role": "system", "content": system_text})

    for message in request.messages:
        content_text, tool_calls, tool_results = (
            _anthropic_content_blocks_to_text_and_tools(
                message.role, message.content, images
            )
        )
        if content_text or tool_calls or not tool_results:
            msg: Dict[str, Any] = {"role": message.role, "content": content_text}
            if tool_calls:
                msg["tool_calls"] = tool_calls
                if not content_text:
                    msg["content"] = None
            processed_messages.append(msg)
        processed_messages.extend(tool_results)

    tools = _anthropic_tools_to_openai(request.tools)
    tool_choice = _anthropic_tool_choice_to_openai(request.tool_choice)
    return processed_messages, images, tools, tool_choice


def _anthropic_request_with_derived_fields(
    request: AnthropicRequest,
) -> AnthropicRequest:
    thinking = _as_plain_dict(request.thinking)
    if request.enable_thinking is None and isinstance(thinking, dict):
        thinking_type = thinking.get("type")
        if thinking_type in ("enabled", "adaptive"):
            request.enable_thinking = True
        elif thinking_type == "disabled":
            request.enable_thinking = False
    if request.thinking_budget is None and isinstance(thinking, dict):
        budget = thinking.get("budget_tokens")
        if budget is not None:
            request.thinking_budget = int(budget)

    output_config = _as_plain_dict(request.output_config)
    if request.response_format is None and isinstance(output_config, dict):
        fmt = _as_plain_dict(output_config.get("format"))
        if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
            request.response_format = {
                "type": "json_schema",
                "json_schema": {"schema": fmt.get("schema", {})},
            }
    return request


def _anthropic_stop_reason(
    finish_reason: Optional[str],
    tool_calls: bool = False,
    stop_sequence: Optional[str] = None,
) -> str:
    if tool_calls:
        return "tool_use"
    if stop_sequence is not None:
        return "stop_sequence"
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    return "end_turn"


def _apply_stop_sequences(
    text: str, stop_sequences: Optional[List[str]]
) -> Tuple[str, Optional[str]]:
    if not text or not stop_sequences:
        return text, None
    best_index = None
    best_sequence = None
    for sequence in stop_sequences:
        if not sequence:
            continue
        index = text.find(sequence)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
            best_sequence = sequence
    if best_index is None:
        return text, None
    return text[:best_index], best_sequence


def _anthropic_content_from_generation(
    full_text: str,
    parsed_tool_calls: Optional[List[Any]] = None,
    include_thinking: bool = False,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    reasoning, content = _split_thinking(
        full_text, thinking_start_token, thinking_end_token
    )
    blocks: List[Dict[str, Any]] = []
    if include_thinking and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    if parsed_tool_calls:
        blocks.extend(
            _openai_tool_call_to_anthropic(call) for call in parsed_tool_calls
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


# Anthropic-compatible endpoints


async def anthropic_messages_endpoint(http_request: Request):
    request_start = time.perf_counter()
    try:
        body = _normalize_anthropic_system_messages(await http_request.json())
        request = _anthropic_request_with_derived_fields(AnthropicRequest(**body))
    except Exception as e:
        return _anthropic_error_response(400, f"Invalid request body: {e}")

    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )
        model, processor, config = get_cached_model(request.model, adapter_path)

        processed_messages, images, tools, tool_choice = (
            _anthropic_messages_to_internal(request)
        )
        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            return _anthropic_error_response(400, str(e))

        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )

        logger.debug(
            "anthropic messages request: model=%s images=%d max_tokens=%s "
            "temp=%s stream=%s tools=%d",
            request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
            len(tools or []),
        )

        if request.stream:
            runtime.metrics.begin_request(
                endpoint="/v1/messages",
                model=request.model,
                stream=True,
            )
            await _preflight_stream_context_budget(
                endpoint="/v1/messages",
                model=request.model,
                prompt=formatted_prompt,
                images=images if images else None,
                audio=None,
                args=gen_args,
            )

            async def stream_generator():
                token_iterator = None
                token_iter = None
                metrics_finalized = False
                metrics = GenerationMetrics()
                prompt_tokens = 0
                output_tokens = 0
                finish_reason = None
                message_id = f"msg_{uuid.uuid4().hex}"
                block_index = 0
                open_block_type = None
                full_output = ""
                text_output = ""
                thinking_state = ThinkingStreamState(
                    gen_args.enable_thinking,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                )
                in_tool_call = False
                tc_start = tool_module.tool_call_start if tool_module else None
                message_started = False

                def close_open_block():
                    nonlocal open_block_type, block_index
                    if open_block_type is None:
                        return ""
                    event = _sse_event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": block_index},
                    )
                    open_block_type = None
                    block_index += 1
                    return event

                def open_block(block_type: str):
                    nonlocal open_block_type
                    if open_block_type == block_type:
                        return ""
                    event = close_open_block()
                    content_block: Dict[str, Any]
                    if block_type == "thinking":
                        content_block = {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "",
                        }
                    else:
                        content_block = {"type": "text", "text": ""}
                    open_block_type = block_type
                    return event + _sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": content_block,
                        },
                    )

                def start_message_event():
                    nonlocal message_started
                    if message_started:
                        return
                    message_started = True
                    start_message = AnthropicMessageResponse(
                        id=message_id,
                        content=[],
                        model=request.model,
                        stop_reason=None,
                        usage=AnthropicUsage.from_metrics(
                            metrics, prompt_tokens, output_tokens
                        ),
                    )
                    yield _sse_event(
                        "message_start",
                        {
                            "type": "message_start",
                            "message": start_message.model_dump(),
                        },
                    )

                try:
                    if runtime.response_generator is not None:
                        ctx, token_iter = await asyncio.to_thread(
                            runtime.response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            None,
                            gen_args,
                        )
                        prompt_tokens = ctx.prompt_tokens

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        token_source = "continuous_batching"
                    else:
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            vision_cache=runtime.model_cache.get("vision_cache"),
                            apc_manager=runtime.apc_manager,
                            **gen_args.to_generate_kwargs(),
                        )

                        def _next_token():
                            try:
                                return next(token_iterator)
                            except StopIteration:
                                return None

                        token_source = "generate"

                    while True:
                        token = await asyncio.to_thread(_next_token)
                        if token is None:
                            break
                        if not hasattr(token, "text"):
                            continue

                        # GenerationResult.generation_tokens is cumulative;
                        # StreamingToken lacks the field and is counted one-at-a-time.
                        token_count = getattr(token, "generation_tokens", None)
                        if token_count is not None:
                            output_tokens = int(token_count)
                        else:
                            output_tokens += 1
                        delta = token.text
                        full_output += delta
                        metrics.record_chunk(token)
                        if prompt_tokens == 0:
                            prompt_tokens = int(getattr(token, "prompt_tokens", 0) or 0)
                        for event in start_message_event():
                            yield event

                        thinking_delta = thinking_state.feed(delta)
                        delta_reasoning = thinking_delta.reasoning
                        delta_content = thinking_delta.content

                        in_tool_call, delta_content = suppress_tool_call_content(
                            full_output, in_tool_call, tc_start, delta_content
                        )

                        if delta_reasoning is not None and gen_args.enable_thinking:
                            yield open_block("thinking")
                            yield _sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "thinking_delta",
                                        "thinking": delta_reasoning,
                                    },
                                },
                            )
                        if (
                            thinking_delta.thinking_closed
                            and open_block_type == "thinking"
                        ):
                            yield _sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "signature_delta",
                                        "signature": "",
                                    },
                                },
                            )
                            yield close_open_block()
                        if delta_content:
                            text_output += delta_content
                            yield open_block("text")
                            yield _sse_event(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": delta_content,
                                    },
                                },
                            )

                        if getattr(token, "finish_reason", None):
                            finish_reason = token.finish_reason
                            break

                    for event in start_message_event():
                        yield event
                    yield close_open_block()

                    parsed_tool_calls = None
                    if tool_module is not None and tools:
                        tc = process_tool_calls(full_output, tool_module, tools)
                        if tc["calls"]:
                            parsed_tool_calls = tc["calls"]

                    if parsed_tool_calls:
                        for call in parsed_tool_calls:
                            tool_block = _openai_tool_call_to_anthropic(call)
                            input_json = json.dumps(
                                tool_block.get("input") or {}, ensure_ascii=False
                            )
                            yield _sse_event(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tool_block["id"],
                                        "name": tool_block["name"],
                                        "input": {},
                                    },
                                },
                            )
                            if input_json:
                                yield _sse_event(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": input_json,
                                        },
                                    },
                                )
                            yield _sse_event(
                                "content_block_stop",
                                {
                                    "type": "content_block_stop",
                                    "index": block_index,
                                },
                            )
                            block_index += 1

                    stop_sequence = None
                    if not parsed_tool_calls:
                        _, stop_sequence = _apply_stop_sequences(
                            text_output, request.stop_sequences
                        )

                    anth_stop_reason = _anthropic_stop_reason(
                        finish_reason,
                        tool_calls=bool(parsed_tool_calls),
                        stop_sequence=stop_sequence,
                    )
                    yield _sse_event(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": anth_stop_reason,
                                "stop_sequence": stop_sequence,
                            },
                            "usage": {"output_tokens": output_tokens},
                        },
                    )
                    yield _sse_event("message_stop", {"type": "message_stop"})

                    completion_tokens = max(
                        0,
                        output_tokens
                        - _count_thinking_tag_tokens(
                            full_output,
                            gen_args.thinking_start_token,
                            gen_args.thinking_end_token,
                        ),
                    )
                    envelope = _build_metrics_envelope(
                        endpoint="/v1/messages",
                        model=request.model,
                        stream=True,
                        backend=token_source,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        generated_tokens=output_tokens,
                        request_elapsed_s=time.perf_counter() - request_start,
                        request_started_s=request_start,
                        token_times=metrics.token_times,
                        prompt_tps=metrics.prompt_tps,
                        generation_tps=metrics.generation_tps,
                        peak_memory_gb=metrics.peak_memory or None,
                        finish_reason=anth_stop_reason,
                        image_count=len(images),
                        structured_output=bool(gen_args.logits_processors),
                        thinking_enabled=bool(gen_args.enable_thinking),
                        tool_parser=tool_parser_type,
                        tool_calls=bool(parsed_tool_calls),
                    )
                    runtime.metrics.record_success(envelope)
                    metrics_finalized = True

                except Exception as e:
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/v1/messages",
                            model=request.model,
                            stream=True,
                            error=str(e),
                        )
                        metrics_finalized = True
                    yield _sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e),
                            },
                        },
                    )
                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    if not metrics_finalized:
                        runtime.metrics.record_failure(
                            endpoint="/v1/messages",
                            model=request.model,
                            stream=True,
                            error="stream_closed_before_completion",
                        )
                    mx.clear_cache()
                    gc.collect()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "request-id": f"req_{uuid.uuid4().hex}",
                },
            )

        runtime.metrics.begin_request(
            endpoint="/v1/messages",
            model=request.model,
            stream=False,
        )
        try:
            full_text = ""
            prompt_tokens = 0
            output_tokens = 0
            metrics = GenerationMetrics()
            finish_reason = None

            if runtime.response_generator is not None:

                def _blocking_generate():
                    metrics = GenerationMetrics()
                    text = ""
                    ot = 0
                    fr = None
                    ctx, token_iter = runtime.response_generator.generate(
                        prompt=formatted_prompt,
                        images=images if images else None,
                        audio=None,
                        args=gen_args,
                    )
                    for tok in token_iter:
                        text += tok.text
                        ot += 1
                        metrics.record_chunk(tok)
                        if tok.finish_reason:
                            fr = tok.finish_reason
                            break
                    try:
                        token_iter.close()
                    except Exception:
                        pass
                    return ctx.prompt_tokens, text, ot, fr, metrics

                (
                    prompt_tokens,
                    full_text,
                    output_tokens,
                    finish_reason,
                    metrics,
                ) = await asyncio.to_thread(_blocking_generate)
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
                )
                full_text = result.text
                prompt_tokens = result.prompt_tokens
                output_tokens = result.generation_tokens
                metrics.record_result(result)
                finish_reason = getattr(result, "finish_reason", None) or "stop"

            parsed_tool_calls = None
            response_text = full_text
            if tool_module is not None and tools:
                tc = process_tool_calls(full_text, tool_module, tools)
                if tc["calls"]:
                    parsed_tool_calls = tc["calls"]
                    response_text = tc["remaining_text"] or ""

            response_text, stop_sequence = _apply_stop_sequences(
                response_text, request.stop_sequences
            )
            content_blocks = _anthropic_content_from_generation(
                response_text,
                parsed_tool_calls=parsed_tool_calls,
                include_thinking=bool(gen_args.enable_thinking),
                thinking_start_token=gen_args.thinking_start_token,
                thinking_end_token=gen_args.thinking_end_token,
            )
            stop_reason = _anthropic_stop_reason(
                finish_reason,
                tool_calls=bool(parsed_tool_calls),
                stop_sequence=stop_sequence,
            )
            response = AnthropicMessageResponse(
                id=f"msg_{uuid.uuid4().hex}",
                content=content_blocks,
                model=request.model,
                stop_reason=stop_reason,
                stop_sequence=stop_sequence,
                usage=AnthropicUsage.from_metrics(
                    metrics, prompt_tokens, output_tokens
                ),
            )

            completion_tokens = max(
                0,
                output_tokens
                - _count_thinking_tag_tokens(
                    full_text,
                    gen_args.thinking_start_token,
                    gen_args.thinking_end_token,
                ),
            )
            envelope = _build_metrics_envelope(
                endpoint="/v1/messages",
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
                request_elapsed_s=time.perf_counter() - request_start,
                request_started_s=request_start,
                token_times=metrics.token_times,
                prompt_tps=metrics.prompt_tps,
                generation_tps=metrics.generation_tps,
                peak_memory_gb=metrics.peak_memory or None,
                finish_reason=stop_reason,
                image_count=len(images),
                structured_output=bool(gen_args.logits_processors),
                thinking_enabled=bool(gen_args.enable_thinking),
                tool_parser=tool_parser_type,
                tool_calls=bool(parsed_tool_calls),
            )
            runtime.metrics.record_success(envelope)
            mx.clear_cache()
            gc.collect()
            return response

        except PromptTooLongError as e:
            runtime.metrics.record_failure(
                endpoint="/v1/messages",
                model=request.model,
                stream=False,
                error=str(e),
            )
            mx.clear_cache()
            gc.collect()
            return _anthropic_error_response(400, str(e))
        except Exception as e:
            runtime.metrics.record_failure(
                endpoint="/v1/messages",
                model=request.model,
                stream=False,
                error=str(e),
            )
            traceback.print_exc()
            mx.clear_cache()
            gc.collect()
            return _anthropic_error_response(
                500, f"Generation failed: {e}", "api_error"
            )

    except Exception as e:
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        return _anthropic_error_response(500, str(e), "api_error")


async def anthropic_count_tokens_endpoint(http_request: Request):
    try:
        body = _normalize_anthropic_system_messages(await http_request.json())
        request = _anthropic_request_with_derived_fields(AnthropicRequest(**body))
        model, processor, config = get_cached_model(request.model)
        processed_messages, images, tools, tool_choice = (
            _anthropic_messages_to_internal(request)
        )
        gen_args = _build_gen_args(
            request, processor, tenant_id=_read_tenant_id(http_request)
        )
        template_kwargs = gen_args.to_template_kwargs()
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice
        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            tools=tools,
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
    except Exception as e:
        return _anthropic_error_response(400, str(e))
