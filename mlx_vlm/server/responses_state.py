import json
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

RESPONSE_STORE_LIMIT = int(os.environ.get("MLX_VLM_RESPONSE_STORE_LIMIT", "1024"))


@dataclass
class StoredResponse:
    response: Dict[str, Any]
    input_items: List[Dict[str, Any]]
    output_items: List[Dict[str, Any]]
    previous_response_id: Optional[str] = None


response_store: Dict[str, StoredResponse] = {}
response_store_order: deque = deque()
response_store_lock = Lock()


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content."""
    if not tc_start:
        return in_tool_call, delta_content
    if not in_tool_call:
        if tc_start in full_output:
            return True, None

        if any(full_output.endswith(tc_start[:j]) for j in range(2, len(tc_start))):
            return False, None
    else:
        return True, None
    return in_tool_call, delta_content


def process_tool_calls(model_output: str, tool_module, tools):
    """Parse tool calls from model output using the appropriate tool parser."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for i, match in enumerate(matches):
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    args = tool_call["arguments"]
                    called_tools.append(
                        {
                            "type": "function",
                            "index": i,
                            "id": str(uuid.uuid4()),
                            "function": {
                                "name": tool_call["name"].strip(),
                                "arguments": (
                                    args
                                    if isinstance(args, str)
                                    else json.dumps(args, ensure_ascii=False)
                                ),
                            },
                        }
                    )
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _sse_event(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, default=_jsonable)}\n\n"


def _split_thinking(text: str) -> Tuple[Optional[str], str]:
    if not text:
        return None, text

    if "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>")
        if start < end:
            reasoning = text[start + len("<think>") : end].strip()
            content = (text[:start] + text[end + len("</think>") :]).strip()
            return reasoning or None, content

    if "<|channel>thought" in text and "<channel|>" in text:
        start_marker = "<|channel>thought"
        end_marker = "<channel|>"
        start = text.find(start_marker)
        end = text.find(end_marker, start)
        if start < end:
            reasoning = text[start + len(start_marker) : end].strip()
            content = (text[:start] + text[end + len(end_marker) :]).strip()
            return reasoning or None, content

    return None, text.strip()


def _normalize_response_input(input_value: Any) -> List[Dict[str, Any]]:
    if isinstance(input_value, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    if not isinstance(input_value, list):
        raise HTTPException(status_code=400, detail="Invalid input format.")

    items = []
    for item in input_value:
        item = _as_plain_dict(item)
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Invalid input format.")
        item_type = item.get("type")
        if item_type is None and item.get("role") is not None:
            item = {**item, "type": "message"}
        items.append(item)
    return items


def _response_call_to_chat_tool_call(item: Dict[str, Any]) -> Dict[str, Any]:
    call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
    name = item.get("name")
    arguments = item.get("arguments")
    if item.get("type") == "shell_call":
        name = name or "shell"
        action = item.get("action") or {}
        arguments = arguments or json.dumps(action, ensure_ascii=False)
    elif item.get("type") == "apply_patch_call":
        name = name or "apply_patch"
        arguments = arguments or item.get("patch") or item.get("input") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "type": "function",
        "id": call_id,
        "function": {"name": name or "tool", "arguments": arguments},
    }


def _append_response_item_to_prompt(
    item: Dict[str, Any],
    chat_messages: List[Dict[str, Any]],
    images: List[Any],
):
    item_type = item.get("type")
    if item_type == "message":
        role = item.get("role") or "user"
        content = item.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                part = _as_plain_dict(part)
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in ("input_text", "output_text", "text"):
                    text_parts.append(str(part.get("text", "")))
                elif part_type == "input_image":
                    image = part.get("image_url") or part.get("file_id")
                    if image:
                        images.append(image)
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    images.append(
                        image_url.get("url")
                        if isinstance(image_url, dict)
                        else image_url
                    )
            content = "\n".join(p for p in text_parts if p)
        chat_messages.append({"role": role, "content": content or ""})
        return

    if item_type in ("function_call", "shell_call", "apply_patch_call"):
        chat_messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_response_call_to_chat_tool_call(item)],
            }
        )
        return

    if item_type in (
        "function_call_output",
        "shell_call_output",
        "apply_patch_call_output",
        "tool_result",
    ):
        output = item.get("output", item.get("content", ""))
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)
        chat_messages.append(
            {
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("tool_call_id"),
                "content": output,
            }
        )


def _response_chain_items(previous_response_id: Optional[str]) -> List[Dict[str, Any]]:
    if not previous_response_id:
        return []
    chain: List[StoredResponse] = []
    seen = set()
    current_id = previous_response_id
    with response_store_lock:
        while current_id:
            if current_id in seen:
                break
            seen.add(current_id)
            stored = response_store.get(current_id)
            if stored is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Previous response not found: {current_id}",
                )
            chain.append(stored)
            current_id = stored.previous_response_id

    items: List[Dict[str, Any]] = []
    for stored in reversed(chain):
        items.extend(stored.input_items)
        items.extend(stored.output_items)
    return items


def _response_items_to_chat(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    chat_messages: List[Dict[str, Any]] = []
    images: List[Any] = []
    for item in items:
        _append_response_item_to_prompt(item, chat_messages, images)
    return chat_messages, images


def _store_response(
    response: Any,
    input_items: List[Dict[str, Any]],
    output_items: List[Dict[str, Any]],
    previous_response_id: Optional[str],
):
    if getattr(response, "store", True) is False:
        return
    payload = response.model_dump(exclude_none=True)
    with response_store_lock:
        response_store[response.id] = StoredResponse(
            response=payload,
            input_items=input_items,
            output_items=output_items,
            previous_response_id=previous_response_id,
        )
        response_store_order.append(response.id)
        while len(response_store_order) > RESPONSE_STORE_LIMIT:
            old_id = response_store_order.popleft()
            response_store.pop(old_id, None)


def _response_tool_to_chat_tool(tool: Any) -> Optional[Dict[str, Any]]:
    tool = _as_plain_dict(tool)
    if not isinstance(tool, dict):
        return None
    tool_type = tool.get("type")
    if tool_type == "function" and isinstance(tool.get("function"), dict):
        return tool
    if tool_type == "function":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("parameters") or {},
            },
        }
    if tool_type == "shell":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "shell",
                "description": tool.get("description") or "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    if tool_type == "apply_patch":
        return {
            "type": "function",
            "function": {
                "name": tool.get("name") or "apply_patch",
                "description": tool.get("description") or "Apply a patch to files.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                },
            },
        }
    return None


def _response_tool_registry(
    tools: Optional[List[Any]],
) -> Tuple[List[Any], Dict[str, str]]:
    chat_tools = []
    registry: Dict[str, str] = {}
    for tool in tools or []:
        plain = _as_plain_dict(tool)
        chat_tool = _response_tool_to_chat_tool(plain)
        if chat_tool is None:
            continue
        chat_tools.append(chat_tool)
        function = chat_tool.get("function", {})
        name = function.get("name")
        if name:
            registry[name] = (plain or {}).get("type", "function")
    return chat_tools, registry


def _tool_call_to_response_item(
    call: Dict[str, Any],
    registry: Dict[str, str],
) -> Dict[str, Any]:
    function = call.get("function", {})
    name = function.get("name") or "tool"
    arguments = function.get("arguments") or "{}"
    call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
    tool_type = registry.get(name, "function")
    if tool_type == "shell":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"command": arguments}
        command = parsed.get("command", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"sh_{uuid.uuid4().hex}",
            "type": "shell_call",
            "call_id": call_id,
            "status": "completed",
            "action": {"type": "exec", "command": command},
        }
    if tool_type == "apply_patch":
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            parsed = {"patch": arguments}
        patch = parsed.get("patch", parsed) if isinstance(parsed, dict) else parsed
        return {
            "id": f"apc_{uuid.uuid4().hex}",
            "type": "apply_patch_call",
            "call_id": call_id,
            "status": "completed",
            "patch": patch,
        }
    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def _response_output_items_from_text(
    full_text: str,
    message_id: str,
    tool_module: Any,
    chat_tools: List[Any],
    tool_registry: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], str, Optional[str], str]:
    reasoning, content = _split_thinking(full_text)
    if tool_module is not None and chat_tools:
        tc = process_tool_calls(full_text, tool_module, chat_tools)
        if tc["calls"]:
            items = [
                _tool_call_to_response_item(call, tool_registry) for call in tc["calls"]
            ]
            _, remaining = _split_thinking(tc.get("remaining_text") or "")
            remaining = re.sub(r"<\|[^>]+\|>|<[^>]+>", "", remaining).strip()
            return items, remaining, reasoning, "tool_calls"
    item = {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    if reasoning:
        item["reasoning"] = reasoning
    return [item], content, reasoning, "stop"
