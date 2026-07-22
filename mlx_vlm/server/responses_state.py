import json
import logging
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

logger = logging.getLogger("mlx_vlm.server")

RESPONSE_STORE_LIMIT = int(os.environ.get("MLX_VLM_RESPONSE_STORE_LIMIT", "1024"))
_CONTENT_MARKERS = ("<|START_TEXT|>", "<|END_TEXT|>")


def _strip_content_markers(text: str) -> str:
    for marker in _CONTENT_MARKERS:
        text = text.replace(marker, "")
    return text


@dataclass
class StoredResponse:
    response: Dict[str, Any]
    input_items: List[Dict[str, Any]]
    output_items: List[Dict[str, Any]]
    previous_response_id: Optional[str] = None


@dataclass
class ThinkingStreamDelta:
    reasoning: Optional[str] = None
    content: Optional[str] = None
    thinking_closed: bool = False


class ThinkingStreamState:
    """Split streamed thinking delimiters from user-visible content."""

    _DEFAULT_OPEN_CLOSE_MARKERS = (
        ("<|channel>thought", "<channel|>"),
        ("<think>", "</think>"),
        ("<|START_THINKING|>", "<|END_THINKING|>"),
    )

    def __init__(
        self,
        enable_thinking: bool = False,
        thinking_start_token: Optional[str] = None,
        thinking_end_token: Optional[str] = None,
    ):
        self.open_close_markers = self._build_open_close_markers(
            thinking_start_token, thinking_end_token
        )
        self.open_markers = tuple(marker for marker, _ in self.open_close_markers)
        self.close_markers = tuple(marker for _, marker in self.open_close_markers)
        self.in_thinking = bool(enable_thinking)
        self.thinking_done = False
        self.buffer = ""

    def feed(self, text: str) -> ThinkingStreamDelta:
        self.buffer += text or ""
        reasoning = []
        content = []
        thinking_closed = False

        while self.buffer:
            if self.in_thinking:
                idx, marker = self._find_first(self.buffer, self.close_markers)
                if idx < 0:
                    emit, self.buffer = self._split_partial(
                        self.buffer, self.close_markers
                    )
                    emit = self._strip_open_marker(emit)
                    if emit:
                        reasoning.append(emit)
                    break

                before = self._strip_open_marker(self.buffer[:idx])
                if before:
                    reasoning.append(before)

                self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
                self.in_thinking = False
                self.thinking_done = True
                thinking_closed = True
                continue

            if self.thinking_done:
                emit, self.buffer = self._split_partial(self.buffer, _CONTENT_MARKERS)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            idx, marker = self._find_first(self.buffer, self.open_markers)
            if idx < 0:
                emit, self.buffer = self._split_partial(self.buffer, self.open_markers)
                emit = _strip_content_markers(emit)
                if emit:
                    content.append(emit)
                break

            if idx:
                emit = _strip_content_markers(self.buffer[:idx])
                if emit:
                    content.append(emit)

            self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
            self.in_thinking = True

        return ThinkingStreamDelta(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
            thinking_closed=thinking_closed,
        )

    @classmethod
    def _build_open_close_markers(
        cls,
        thinking_start_token: Optional[str],
        thinking_end_token: Optional[str],
    ) -> Tuple[Tuple[str, str], ...]:
        markers = []
        if thinking_start_token and thinking_end_token:
            markers.append((thinking_start_token, thinking_end_token))
        for marker_pair in cls._DEFAULT_OPEN_CLOSE_MARKERS:
            if marker_pair not in markers:
                markers.append(marker_pair)
        return tuple(markers)

    @staticmethod
    def _find_first(text: str, markers: Tuple[str, ...]) -> Tuple[int, str]:
        found_idx = -1
        found_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0 and (found_idx < 0 or idx < found_idx):
                found_idx = idx
                found_marker = marker
        return found_idx, found_marker

    @staticmethod
    def _split_partial(text: str, markers: Tuple[str, ...]) -> Tuple[str, str]:
        hold = 0
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for length in range(max_len, 0, -1):
                if text.endswith(marker[:length]):
                    hold = max(hold, length)
                    break
        if hold:
            return text[:-hold], text[-hold:]
        return text, ""

    def _strip_open_marker(self, text: str) -> str:
        for marker in self.open_markers:
            if marker in text:
                before, after = text.split(marker, 1)
                return before + after.lstrip("\n")
        return text


def prompt_has_open_thinking(
    prompt: Any,
    enable_thinking: bool = False,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> bool:
    """Return whether generation starts inside a prompt-opened thinking block."""
    if not isinstance(prompt, str):
        return False

    stripped_prompt = prompt.rstrip()
    for start_marker, _ in ThinkingStreamState._build_open_close_markers(
        thinking_start_token, thinking_end_token
    ):
        if stripped_prompt.endswith(start_marker):
            return True
    return False


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
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    parsed = tool_module.parse_tool_call(call, tools)
                    parsed_calls = parsed if isinstance(parsed, list) else [parsed]
                    for tool_call in parsed_calls:
                        args = tool_call["arguments"]
                        called_tools.append(
                            {
                                "type": "function",
                                "index": len(called_tools),
                                "id": str(uuid.uuid4()),
                                "function": {
                                    "name": tool_call["name"].strip(),
                                    "arguments": (
                                        args
                                        if isinstance(args, str)
                                        else json.dumps(args, ensure_ascii=False)
                                    ),
                                },
                            },
                        )
                except Exception:
                    logger.warning("Invalid tool call: %s", call)
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


def _clean_reasoning(reasoning: str, start_marker: str) -> str:
    reasoning = reasoning.replace(start_marker, "")
    if start_marker == "<|channel>thought":
        reasoning = reasoning.lstrip("thought")
    return reasoning.strip()


def _split_thinking(
    text: str,
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    if not text:
        return None, text

    for start_marker, end_marker in ThinkingStreamState._build_open_close_markers(
        thinking_start_token, thinking_end_token
    ):
        start = text.find(start_marker)
        end = text.find(end_marker, start if start >= 0 else 0)
        if start >= 0 and start < end:
            reasoning = text[start + len(start_marker) : end].strip()
            content = _strip_content_markers(
                text[:start] + text[end + len(end_marker) :]
            ).strip()
            return reasoning or None, content

        if end_marker in text:
            reasoning, content = text.split(end_marker, 1)
            reasoning = _clean_reasoning(reasoning, start_marker)
            return reasoning or None, _strip_content_markers(content).strip()

        if start_marker in text:
            reasoning = _clean_reasoning(text, start_marker)
            return reasoning or None, ""

    return None, _strip_content_markers(text).strip()


def _response_output_items_from_text(
    full_text: str,
    message_id: str,
    tool_module: Any,
    chat_tools: List[Any],
    tool_registry: Dict[str, str],
    thinking_start_token: Optional[str] = None,
    thinking_end_token: Optional[str] = None,
    reasoning_item_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str], str]:
    reasoning, content = _split_thinking(
        full_text, thinking_start_token, thinking_end_token
    )
    reasoning_items = _reasoning_output_items(reasoning, reasoning_item_id)
    if tool_module is not None and chat_tools:
        tc = process_tool_calls(full_text, tool_module, chat_tools)
        if tc["calls"]:
            items = [
                _tool_call_to_response_item(call, tool_registry) for call in tc["calls"]
            ]
            _, remaining = _split_thinking(
                tc.get("remaining_text") or "",
                thinking_start_token,
                thinking_end_token,
            )
            remaining = re.sub(r"<\|[^>]+\|>|<[^>]+>", "", remaining).strip()
            return reasoning_items + items, remaining, reasoning, "tool_calls"
    item = {
        "id": message_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }
    if reasoning:
        item["reasoning"] = reasoning
    return reasoning_items + [item], content, reasoning, "stop"


def _reasoning_output_items(
    reasoning: Optional[str],
    item_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not reasoning:
        return []
    return [
        {
            "id": item_id or f"rs_{uuid.uuid4().hex}",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning}],
        }
    ]


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
