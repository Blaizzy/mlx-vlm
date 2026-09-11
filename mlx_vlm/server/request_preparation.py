"""Shared prompt preparation after protocol-specific input normalization."""

import base64
import binascii
import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, List, Optional

from fastapi import HTTPException

from ..generate.video import resolve_video_inputs
from ..prompt_utils import apply_chat_template, extract_text_from_content
from ..tools import _prepare_chat_tool_choice
from .generation import GenerationArguments
from .responses_state import _response_items_to_chat, _response_tool_registry
from .schemas import InputAudio

logger = logging.getLogger("mlx_vlm.server")


@dataclass
class PromptInput:
    """Normalized messages; absent modalities retain each API's template defaults."""

    messages: List[dict]
    images: List[Any] = field(default_factory=list)
    audio: Optional[List[Any]] = None
    videos: Optional[List[Any]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Any = None
    generation_kwargs: dict = field(default_factory=dict)


@dataclass
class PreparedPrompt:
    prompt: Any
    images: List[Any]
    audio: List[Any]
    videos: List[Any]
    generation_kwargs: dict


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


def _looks_like_audio_reference(value: str) -> bool:
    return value.startswith(_AUDIO_REFERENCE_PREFIXES) or value.lower().endswith(
        _AUDIO_REFERENCE_SUFFIXES
    )


def _normalize_response_instruction_messages(
    chat_messages: List[dict],
    instructions: Optional[str],
) -> Optional[str]:
    instruction_parts = [instructions] if instructions else []
    conversation = []

    for message in chat_messages:
        if message.get("role") in ("system", "developer"):
            content = message.get("content")
            if content:
                instruction_parts.append(str(content))
        else:
            conversation.append(message)

    normalized_instructions = "\n\n".join(instruction_parts) or None
    if normalized_instructions:
        conversation.insert(
            0,
            {"role": "system", "content": normalized_instructions},
        )
    chat_messages[:] = conversation
    return normalized_instructions


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


def normalize_chat_input(request) -> PromptInput:
    generation_kwargs = {}

    if request.resize_shape is not None:
        if len(request.resize_shape) not in [1, 2]:
            raise HTTPException(
                status_code=400,
                detail="resize_shape must contain exactly two integers (height, width)",
            )
        generation_kwargs["resize_shape"] = (
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

    _ensure_effective_input(processed_messages, images=images, audio=audio)

    processed_messages, tools, tool_choice = _prepare_chat_tool_choice(
        processed_messages,
        request.tools,
        request.tool_choice,
    )

    return PromptInput(
        messages=processed_messages,
        images=images,
        audio=audio,
        videos=videos,
        tools=tools,
        tool_choice=tool_choice,
        generation_kwargs=generation_kwargs,
    )


def normalize_responses_input(request, prompt_items):
    """Retain Responses instruction merging and response-item conversion semantics."""
    messages, images = _response_items_to_chat(prompt_items)
    instructions = _normalize_response_instruction_messages(
        messages, request.instructions
    )
    _ensure_effective_input(messages, images=images)
    tools, registry = _response_tool_registry(request.tools)
    return (
        PromptInput(
            messages=messages,
            images=images,
            tools=tools or None,
            tool_choice=request.tool_choice,
        ),
        instructions,
        registry,
    )


def prepare_prompt(
    source: PromptInput,
    processor,
    config,
    args: GenerationArguments,
    *,
    render: Callable = apply_chat_template,
) -> PreparedPrompt:
    """Render normalized history without running a protocol adapter a second time."""
    images, videos = list(source.images), list(source.videos or [])
    media_kwargs = {}
    if source.audio is not None:
        media_kwargs["num_audios"] = len(source.audio)
    if source.videos is not None:
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

        media_kwargs["video"] = videos or None
    template_kwargs = args.to_template_kwargs()
    if source.tool_choice is not None:
        template_kwargs["tool_choice"] = source.tool_choice
    prompt = render(
        processor,
        config,
        source.messages,
        num_images=len(images),
        tools=source.tools,
        **media_kwargs,
        **template_kwargs,
    )
    return PreparedPrompt(
        prompt=prompt,
        images=images,
        audio=list(source.audio or []),
        videos=videos,
        generation_kwargs=dict(source.generation_kwargs),
    )
