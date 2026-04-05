import argparse
import asyncio
import gc
import json
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Literal, Optional, Union

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypeAlias, TypedDict

from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
    PromptCacheState,
    generate,
    normalize_resize_shape,
    stream_generate,
)
from .prompt_utils import apply_chat_template
from .tool_parsers import _infer_tool_parser, load_tool_module
from .utils import load
from .version import __version__
from .vision_cache import VisionFeatureCache
from .responses_models import (
    ResponsesRequest,
    ResponseObject,
    ResponseUsage,
    ResponseErrorObject,
    ResponseIncompleteDetails,
    ResponseMessageItem,
    ResponseFunctionCallItem,
    ContentPartOutputText as ResponseContentPartOutputText,
    BaseStreamEvent as ResponseBaseStreamEvent,
    ResponseCreatedEvent as ResponsesCreatedEvent,
    ResponseInProgressEvent as ResponsesInProgressEvent,
    ResponseOutputItemAddedEvent as ResponsesOutputItemAddedEvent,
    ResponseContentPartAddedEvent as ResponsesContentPartAddedEvent,
    ResponseOutputTextDeltaEvent as ResponsesOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent as ResponsesOutputTextDoneEvent,
    ResponseContentPartDoneEvent as ResponsesContentPartDoneEvent,
    ResponseOutputItemDoneEvent as ResponsesOutputItemDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent as ResponsesFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent as ResponsesFunctionCallArgumentsDoneEvent,
    ResponseCompletedEvent as ResponsesCompletedEvent,
)
from .responses_store import ResponseStore

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080

_responses_store = ResponseStore()


def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        print(
            f"Model {model} is quantization aware, (Rotating)KVCache cache will not be quantized to {kv_bits} bits, use --max-kv-size [tokens] instead."
        )
        return None
    return kv_bits


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        print(
            f"Model {model} uses QuantizedKVCache cache, can't set max KV size, use --kv-bits [bits] instead."
        )
        return None
    return max_kv_tokens


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


@asynccontextmanager
async def lifespan(app):
    # Startup
    model_path = os.environ.get("PRELOAD_MODEL")
    adapter_path = os.environ.get("PRELOAD_ADAPTER") or None
    if model_path:
        try:
            print(f"Preloading model: {model_path}")
            get_cached_model(model_path, adapter_path)
        except Exception as e:
            print(f"Failed to preload model: {e}")
            print("Server will continue without a preloaded model.")
    yield
    unload_model_sync()


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

# Loading/unloading utilities

model_cache = {}

# Prompt cache: reuse KV state across requests with the same prompt prefix.
# Keyed by model name — one PromptCacheState per loaded model.
_prompt_cache_states: dict[str, PromptCacheState] = {}

# Concurrency guard: MLX generation is single-threaded on Metal.
# Concurrent requests would corrupt shared GPU state. The semaphore
# serializes access to the generation pipeline.
_generation_semaphore: Optional[asyncio.Semaphore] = None


def get_max_concurrent_requests() -> int:
    return int(os.environ.get("MAX_CONCURRENT_REQUESTS", 1))


def get_generation_semaphore() -> asyncio.Semaphore:
    """Get or create the generation semaphore."""
    global _generation_semaphore
    if _generation_semaphore is None:
        _generation_semaphore = asyncio.Semaphore(get_max_concurrent_requests())
    return _generation_semaphore


def get_prompt_cache_state(model_name: str) -> PromptCacheState:
    """Get or create a PromptCacheState for the given model."""
    if model_name not in _prompt_cache_states:
        _prompt_cache_states[model_name] = PromptCacheState()
    return _prompt_cache_states[model_name]


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")

    def dump_kwargs(self, *fields: str) -> dict[str, Any]:
        return self.model_dump(include=set(fields), exclude_none=True)


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        print("Model and processor loaded successfully.")
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def get_cached_model(model_path: str, adapter_path: Optional[str] = None):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    """
    global model_cache

    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        print(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    # Load the model resources
    model, processor, config = load_model_resources(model_path, adapter_path)

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": VisionFeatureCache(),
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )
    # Clear vision cache before dropping references
    if "vision_cache" in model_cache:
        model_cache["vision_cache"].clear()
    model_cache = {}
    # Clear prompt cache states
    _prompt_cache_states.clear()
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint


class ResponseInputTextParam(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["input_text", "text"]
    ]  # The type of the input item. Always `input_text`.


class ResponseInputImageParam(TypedDict, total=False):
    detail: Literal["high", "low", "auto"] = Field(
        "auto", description="The detail level of the image to be sent to the model."
    )
    """The detail level of the image to be sent to the model.

    One of `high`, `low`, or `auto`.
    """
    type: Required[
        Literal["input_image"]
    ]  # The type of the input item. Always `input_image`.
    image_url: Required[str]
    file_id: Optional[str]
    """The ID of the file to be sent to the model.
     NOTE : wouldn't this help the model if we passed the file_id as well to the vlm models
    """


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    type: Required[
        Literal["input_audio"]
    ]  # The type of the input item. Always `input_audio`.
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    type: Required[
        Literal["image_url"]
    ]  # The type of the input item. Always`image_url`.
    image_url: Required[ImageUrl]


ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResizeShapeInput: TypeAlias = tuple[int] | tuple[int, int]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]


class ResponseOutputText(TypedDict, total=False):
    text: Required[str]
    type: Required[
        Literal["output_text"]
    ]  # The type of the output item. Always `output_text`


ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


class ChatMessage(FlexibleBaseModel):
    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ...,
        description="Role of the message sender (e.g., 'system', 'user', 'assistant').",
    )
    content: Optional[
        Union[
            str,
            ResponseInputMessageContentListParam,
            ResponseOutputMessageContentList,
        ]
    ] = Field(None, description="Content of the message.")
    tool_calls: List = []


class GenerationParams(FlexibleBaseModel):
    temperature: float = Field(
        DEFAULT_TEMPERATURE,
        description="Temperature for sampling.",
    )
    top_p: float = Field(
        DEFAULT_TOP_P,
        description="Top-p sampling.",
    )
    top_k: Optional[int] = Field(
        None,
        description="Top-k sampling cutoff.",
    )
    min_p: Optional[float] = Field(
        None,
        description="Min-p sampling threshold.",
    )
    repetition_penalty: Optional[float] = Field(
        None, description="Penalty applied to repeated tokens."
    )
    logit_bias: Optional[dict[int, float]] = Field(
        None, description="Additive logit bias keyed by token id."
    )

    def shared_generation_kwargs(self) -> dict[str, Any]:
        return self.dump_kwargs(
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "logit_bias",
        )


class TemplateParams(FlexibleBaseModel):
    enable_thinking: Optional[bool] = Field(
        None,
        description="Enable thinking mode in the chat template.",
    )
    thinking_budget: Optional[int] = Field(
        None,
        description="Maximum number of thinking tokens before forcing the end token.",
    )
    thinking_start_token: Optional[str] = Field(
        DEFAULT_THINKING_START_TOKEN,
        description="Token that marks the start of a thinking block.",
    )
    thinking_end_token: Optional[str] = Field(
        DEFAULT_THINKING_END_TOKEN,
        description="Token that marks the end of a thinking block.",
    )

    def template_kwargs(self) -> dict[str, Any]:
        kwargs = self.dump_kwargs(
            "enable_thinking",
            "thinking_budget",
            "thinking_start_token",
            "thinking_end_token",
        )
        kwargs.setdefault("enable_thinking", False)
        return kwargs


class OpenAIRequest(GenerationParams, TemplateParams):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[ChatMessage]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        description="Maximum number of tokens to generate.",
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs = self.dump_kwargs("max_output_tokens")
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
        return {**kwargs, **self.shared_generation_kwargs()}


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class OpenAIErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


class OpenAIResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for this Response")
    object: Literal["response"] = Field(
        ..., description="The object type of this resource - always set to response"
    )
    created_at: int = Field(
        ..., description="Unix timestamp (in seconds) of when this Response was created"
    )
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        ..., description="The status of the response generation"
    )
    error: Optional[OpenAIErrorObject] = Field(
        None,
        description="An error object returned when the model fails to generate a Response",
    )
    instructions: Optional[str] = Field(
        None,
        description="Inserts a system (or developer) message as the first item in the model's context",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a response",
    )
    model: str = Field(..., description="Model ID used to generate the response")
    output: List[Union[ChatMessage, Any]] = Field(
        ..., description="An array of content items generated by the model"
    )
    output_text: Optional[str] = Field(
        None,
        description="SDK-only convenience property containing aggregated text output",
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature between 0 and 2"
    )
    top_p: Optional[float] = Field(
        None, ge=0, le=1, description="Nucleus sampling probability mass"
    )
    truncation: Union[Literal["auto", "disabled"], str] = Field(
        "disabled", description="The truncation strategy to use"
    )
    usage: OpenAIUsage = Field(
        ..., description="Token usage details"
    )  # we need the model to return stats
    user: Optional[str] = Field(
        None, description="A unique identifier representing your end-user"
    )


class BaseStreamEvent(BaseModel):
    type: str


class ContentPartOutputText(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[str] = []


class MessageItem(BaseModel):
    id: str
    type: Literal["message"]
    status: Literal["in_progress", "completed"]
    role: str
    content: List[ContentPartOutputText] = []


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"]
    response: OpenAIResponse


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"]
    response: OpenAIResponse


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"]
    output_index: int
    item: MessageItem


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"]
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"]
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"]
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"]
    output_index: int
    item: MessageItem


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"]
    response: OpenAIResponse


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
]

# Models for /chat/completion endpoint


class VLMRequest(GenerationParams, TemplateParams):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS,
        description="Maximum number of tokens to generate.",
    )
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for a square resize or two integers for (height, width).",
    )

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value):
        return normalize_resize_shape(value)

    def generation_kwargs(self) -> dict[str, Any]:
        return {
            **self.dump_kwargs("max_tokens", "resize_shape"),
            **self.shared_generation_kwargs(),
        }


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class UsageStats(OpenAIUsage):
    """
    Inherits from OpenAIUsage and adds additional fields for usage statistics.
    """

    prompt_tps: float = Field(..., description="Tokens per second for the prompt.")
    generation_tps: float = Field(
        ..., description="Tokens per second for the generation."
    )
    peak_memory: float = Field(
        ..., description="Peak memory usage during the generation."
    )


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]


class ChatChoice(BaseModel):
    finish_reason: str
    message: ChatMessage


class ChatResponse(BaseModel):
    model: str
    choices: List[ChatChoice]
    usage: Optional[UsageStats]


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage


class ChatStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChoice]
    usage: Optional[UsageStats]


def resolve_stop_sequences(
    stop: Optional[Union[str, list]],
) -> Optional[list]:
    """Normalize stop sequences for the generation stopping criteria.

    The generation pipeline's ``add_eos_token_ids`` accepts strings
    and handles tokenization internally.

    Args:
        stop: A single stop string or list of stop strings, or None.

    Returns:
        A list of stop strings (max 4), or None.
    """
    if not stop:
        return None
    if isinstance(stop, str):
        stop = [stop]
    sequences = [s for s in stop[:4] if isinstance(s, str) and s]
    return sequences if sequences else None


def resolve_tool_choice(
    tools: Optional[list],
    tool_choice: Optional[Any],
) -> tuple[Optional[list], Optional[str]]:
    """Apply tool_choice policy to the tools list.

    Args:
        tools: The original tools list from the request.
        tool_choice: ``"none"``, ``"auto"``, ``"required"``, or a dict
            specifying a particular tool.

    Returns:
        Tuple of ``(filtered_tools, system_instruction)``.
    """
    if not tools or tool_choice is None or tool_choice == "auto":
        return tools, None

    if tool_choice == "none":
        return None, None

    if tool_choice == "required":
        return tools, "You must call one of the available tools to answer this request."

    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        name = func.get("name") if isinstance(func, dict) else None
        if name:
            filtered = [
                t for t in tools
                if (t.get("function", {}) or {}).get("name") == name
                or t.get("name") == name
            ]
            return (
                filtered or tools,
                f'You must call the "{name}" tool to answer this request.',
            )

    return tools, None


def build_generation_kwargs(
    request: Any,
    template_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prefill_step_size": get_prefill_step_size(),
        "kv_bits": get_quantized_kv_bits(request.model),
        "kv_group_size": get_kv_group_size(),
        "kv_quant_scheme": get_kv_quant_scheme(),
        "max_kv_size": get_max_kv_size(request.model),
        "quantized_kv_start": get_quantized_kv_start(),
        **request.generation_kwargs(),
        **template_kwargs,
    }


def process_tool_calls(model_output: str, tool_module, tools):
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
            tool_call_index = 0
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    called_tool = {}
                    called_tool["type"] = "function"
                    called_tool["index"] = tool_call_index
                    called_tool["id"] = str(uuid.uuid4())
                    called_tool["function"] = {}
                    called_tool["function"]["name"] = tool_call["name"].strip()
                    called_tool["function"]["arguments"] = json.dumps(
                        tool_call["arguments"], ensure_ascii=False
                    )
                    called_tools.append(called_tool)
                    tool_call_index += 1
                except Exception:
                    print(f"Invalid tool call: {call}")
    return dict(calls=called_tools, remaining_text=remaining)


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# Responses API helpers
# ---------------------------------------------------------------------------


def responses_input_to_messages(
    input_items: Union[str, list],
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
) -> tuple[list[dict], list[str]]:
    """Convert Responses API input items to chat messages and images.

    Args:
        input_items: String input or list of input items.
        instructions: Optional system instructions to prepend.
        previous_response_id: Optional previous response ID for context replay.

    Returns:
        Tuple of (chat_messages, image_urls).
    """
    chat_messages: list[dict] = []
    images: list[str] = []

    # Replay previous response context
    if previous_response_id:
        replayed = _responses_store.replay_input(previous_response_id)
        if replayed is None:
            raise HTTPException(
                status_code=404,
                detail=f"Previous response not found: {previous_response_id}",
            )
        # Recursively process replayed items
        prev_messages, prev_images = responses_input_to_messages(replayed)
        chat_messages.extend(prev_messages)
        images.extend(prev_images)

    # Prepend instructions as system message
    if instructions:
        chat_messages.insert(0, {"role": "system", "content": instructions})

    # Handle string input
    if isinstance(input_items, str):
        chat_messages.append({"role": "user", "content": input_items})
        return chat_messages, images

    # Handle list of input items
    for item in input_items:
        if isinstance(item, dict):
            item_type = item.get("type", "")
            role = item.get("role", "")

            # Function call output item
            if item_type == "function_call_output":
                call_id = item.get("call_id", "unknown")
                output = item.get("output", "")
                chat_messages.append({
                    "role": "tool",
                    "content": output,
                    "tool_call_id": call_id,
                })
                continue

            # Function call item (from previous assistant turn)
            if item_type == "function_call":
                chat_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", ""),
                        },
                    }],
                })
                continue

            # Regular message with role and content
            if role:
                content = item.get("content", "")

                # Normalize developer role to system
                msg_role = "system" if role == "developer" else role

                if isinstance(content, str):
                    chat_messages.append({"role": msg_role, "content": content})
                elif isinstance(content, list):
                    # Process content items
                    text_parts = []
                    for ci in content:
                        if isinstance(ci, dict):
                            ci_type = ci.get("type", "")
                            if ci_type in ("input_text", "text"):
                                text_parts.append(ci.get("text", ""))
                            elif ci_type == "input_image":
                                images.append(ci.get("image_url", ""))
                            elif ci_type == "image_url":
                                img = ci.get("image_url", {})
                                if isinstance(img, dict):
                                    images.append(img.get("url", ""))
                                elif isinstance(img, str):
                                    images.append(img)
                            elif ci_type == "output_text":
                                # Multi-turn: previous assistant output
                                chat_messages.append({
                                    "role": "assistant",
                                    "content": ci.get("text", ""),
                                })
                            elif ci_type == "input_audio":
                                pass  # Audio not yet supported in responses
                            else:
                                pass  # Skip unsupported content types gracefully

                    if text_parts:
                        chat_messages.append({
                            "role": msg_role,
                            "content": "\n".join(text_parts),
                        })
                else:
                    chat_messages.append({
                        "role": msg_role,
                        "content": str(content) if content else "",
                    })
                continue

        # Handle Pydantic ChatMessage objects
        elif hasattr(item, "role"):
            role = item.role
            msg_role = "system" if role == "developer" else role
            content = item.content

            if content is None:
                chat_messages.append({"role": msg_role, "content": ""})
            elif isinstance(content, str):
                chat_messages.append({"role": msg_role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for ci in content:
                    if isinstance(ci, dict):
                        ci_type = ci.get("type", "")
                        if ci_type in ("input_text", "text"):
                            text_parts.append(ci.get("text", ""))
                        elif ci_type == "input_image":
                            images.append(ci.get("image_url", ""))
                        elif ci_type == "image_url":
                            img = ci.get("image_url", {})
                            if isinstance(img, dict):
                                images.append(img.get("url", ""))
                            elif isinstance(img, str):
                                images.append(img)
                        elif ci_type == "output_text":
                            chat_messages.append({
                                "role": "assistant",
                                "content": ci.get("text", ""),
                            })

                if text_parts:
                    chat_messages.append({
                        "role": msg_role,
                        "content": "\n".join(text_parts),
                    })

    return chat_messages, images


def build_responses_output(
    raw_text: str,
    tool_parser_type: Optional[str],
    tool_module: Optional[Any],
    tools: Optional[list],
) -> list[Union[ResponseMessageItem, ResponseFunctionCallItem]]:
    """Build structured Responses API output items from raw model text.

    Parses tool calls from the raw text if a tool parser is available,
    creating ResponseFunctionCallItem for each detected call and a
    ResponseMessageItem for any remaining text.

    Args:
        raw_text: The raw text output from the model.
        tool_parser_type: The detected tool parser type (e.g., "gemma4"), or None.
        tool_module: The loaded tool parser module, or None.
        tools: The tool definitions from the request, or None.

    Returns:
        List of output items (message items and/or function call items).
    """
    output_items: list[Union[ResponseMessageItem, ResponseFunctionCallItem]] = []
    remaining_text = raw_text

    # Try to parse tool calls
    if tool_parser_type and tool_module and tools:
        try:
            result = process_tool_calls(raw_text, tool_module, tools)
            if result["calls"]:
                for call in result["calls"]:
                    func_info = call.get("function", {})
                    output_items.append(
                        ResponseFunctionCallItem(
                            name=func_info.get("name", ""),
                            arguments=func_info.get("arguments", "{}"),
                            call_id=call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        )
                    )
                remaining_text = result.get("remaining_text", "").strip()
        except Exception:
            # If tool parsing fails, fall through to plain text
            remaining_text = raw_text

    # Create message item for any remaining text
    if remaining_text or not output_items:
        msg_item = ResponseMessageItem(
            content=[ResponseContentPartOutputText(text=remaining_text)] if remaining_text else [],
        )
        # Insert message before function calls (matching OpenAI ordering)
        output_items.insert(0, msg_item)

    return output_items


# OpenAI compatible endpoints


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(request: ResponsesRequest):
    """OpenAI-compatible Responses API endpoint.

    Supports tool calling, multi-turn via previous_response_id, and streaming
    with proper SSE event sequences including function_call argument events.
    """

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model)

        # Convert input to chat messages
        chat_messages, images = responses_input_to_messages(
            request.input,
            instructions=request.instructions,
            previous_response_id=request.previous_response_id,
        )

        # Set up tool parser (apply tool_choice policy)
        tools = request.tools
        tool_choice_val = getattr(request, "tool_choice", "auto")
        tools, tool_instruction = resolve_tool_choice(tools, tool_choice_val)
        if tool_instruction:
            chat_messages.insert(0, {"role": "system", "content": tool_instruction})

        tool_parser_type = None
        tool_module = None
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        if hasattr(tokenizer, "chat_template") and tools:
            tool_parser_type = _infer_tool_parser(tokenizer.chat_template)
            if tool_parser_type is not None:
                tool_module = load_tool_module(tool_parser_type)

        # Build template kwargs
        template_kwargs = request.template_kwargs()

        # Apply chat template (pass tools so the template can include tool defs)
        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            tools=tools,
            **template_kwargs,
        )
        generation_kwargs = build_generation_kwargs(request, template_kwargs)

        # Resolve stop sequences to token IDs
        stop_seqs = resolve_stop_sequences(getattr(request, "stop", None))
        if stop_seqs:
            generation_kwargs["eos_tokens"] = stop_seqs

        generated_at = int(time.time())
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        if request.stream:
            # ----------------------------------------------------------
            # Streaming response
            # ----------------------------------------------------------
            async def stream_responses_generator():
                seq = 0  # sequence_number counter

                def _evt(event_type: str, event_obj) -> str:
                    nonlocal seq
                    event_obj.sequence_number = seq
                    seq += 1
                    return f"event: {event_type}\ndata: {event_obj.model_dump_json()}\n\n"

                try:
                    # Build base ResponseObject (in_progress, empty output)
                    base_response = ResponseObject(
                        id=response_id,
                        created_at=generated_at,
                        status="in_progress",
                        model=request.model,
                        output=[],
                        instructions=request.instructions,
                        max_output_tokens=request.max_output_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        tools=tools or [],
                        tool_choice=request.tool_choice,
                        parallel_tool_calls=request.parallel_tool_calls,
                        previous_response_id=request.previous_response_id,
                        metadata=request.metadata,
                        usage=ResponseUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    )

                    # response.created
                    yield _evt("response.created", ResponsesCreatedEvent(response=base_response))
                    # response.in_progress
                    yield _evt("response.in_progress", ResponsesInProgressEvent(response=base_response))

                    # output_item.added (message)
                    msg_item = ResponseMessageItem(id=message_id, status="in_progress", content=[])
                    yield _evt(
                        "response.output_item.added",
                        ResponsesOutputItemAddedEvent(output_index=0, item=msg_item),
                    )

                    # content_part.added
                    empty_part = ResponseContentPartOutputText(text="")
                    yield _evt(
                        "response.content_part.added",
                        ResponsesContentPartAddedEvent(
                            item_id=message_id, output_index=0, content_index=0, part=empty_part,
                        ),
                    )

                    # Stream text deltas (with prompt cache + concurrency guard)
                    sem = get_generation_semaphore()
                    await sem.acquire()
                    cache_state = get_prompt_cache_state(request.model)
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        vision_cache=model_cache.get("vision_cache"),
                        prompt_cache_state=cache_state,
                        **generation_kwargs,
                    )

                    full_text = ""
                    visible_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}
                    in_tool_call = False
                    tool_call_start_tag = tool_module.tool_call_start if tool_module else "<tool_call>"

                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            continue

                        delta = chunk.text
                        full_text += delta
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                        }

                        # Suppress tool call tokens from being streamed as text
                        if not in_tool_call and tool_call_start_tag in full_text:
                            in_tool_call = True
                        if in_tool_call:
                            continue

                        # Check if this delta starts a tool call tag
                        # (partial match: buffer might end with "<tool" before "_call>")
                        if tools and tool_call_start_tag[:1] in delta:
                            pending = full_text[-(len(delta) + len(tool_call_start_tag)):]
                            if any(
                                tool_call_start_tag[:i] == pending[-i:]
                                for i in range(2, len(tool_call_start_tag) + 1)
                            ):
                                continue

                        visible_text += delta
                        yield _evt(
                            "response.output_text.delta",
                            ResponsesOutputTextDeltaEvent(
                                item_id=message_id, output_index=0, content_index=0, delta=delta,
                            ),
                        )

                    # Determine finish reason
                    max_tok = request.max_output_tokens
                    is_length = usage_stats["output_tokens"] >= max_tok
                    status = "incomplete" if is_length else "completed"

                    # Use visible_text (sans tool call markup) for text events
                    display_text = visible_text.strip()

                    # output_text.done
                    yield _evt(
                        "response.output_text.done",
                        ResponsesOutputTextDoneEvent(
                            item_id=message_id, output_index=0, content_index=0, text=display_text,
                        ),
                    )

                    # content_part.done
                    final_part = ResponseContentPartOutputText(text=display_text)
                    yield _evt(
                        "response.content_part.done",
                        ResponsesContentPartDoneEvent(
                            item_id=message_id, output_index=0, content_index=0, part=final_part,
                        ),
                    )

                    # output_item.done (message)
                    final_msg = ResponseMessageItem(
                        id=message_id, status="completed", content=[final_part],
                    )
                    yield _evt(
                        "response.output_item.done",
                        ResponsesOutputItemDoneEvent(output_index=0, item=final_msg),
                    )

                    # Collect all output items for final response
                    all_output_items: list = [final_msg]

                    # Parse tool calls from accumulated text
                    if tool_parser_type and tool_module and tools:
                        try:
                            tc_result = process_tool_calls(full_text, tool_module, tools)
                            if tc_result["calls"]:
                                for idx, call in enumerate(tc_result["calls"]):
                                    func_info = call.get("function", {})
                                    fc_item = ResponseFunctionCallItem(
                                        name=func_info.get("name", ""),
                                        arguments=func_info.get("arguments", "{}"),
                                        call_id=call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                    )
                                    out_idx = len(all_output_items)

                                    # output_item.added (function_call)
                                    yield _evt(
                                        "response.output_item.added",
                                        ResponsesOutputItemAddedEvent(output_index=out_idx, item=fc_item),
                                    )

                                    # function_call_arguments.delta (full arguments in one shot)
                                    yield _evt(
                                        "response.function_call_arguments.delta",
                                        ResponsesFunctionCallArgumentsDeltaEvent(
                                            item_id=fc_item.id,
                                            output_index=out_idx,
                                            delta=fc_item.arguments,
                                        ),
                                    )

                                    # function_call_arguments.done
                                    yield _evt(
                                        "response.function_call_arguments.done",
                                        ResponsesFunctionCallArgumentsDoneEvent(
                                            item_id=fc_item.id,
                                            output_index=out_idx,
                                            arguments=fc_item.arguments,
                                        ),
                                    )

                                    # output_item.done (function_call)
                                    yield _evt(
                                        "response.output_item.done",
                                        ResponsesOutputItemDoneEvent(output_index=out_idx, item=fc_item),
                                    )

                                    all_output_items.append(fc_item)
                        except Exception:
                            pass  # Tool parsing failure is non-fatal in streaming

                    # response.completed
                    total_tokens = usage_stats["input_tokens"] + usage_stats["output_tokens"]
                    completed_response = base_response.model_copy(
                        update={
                            "status": status,
                            "output": all_output_items,
                            "incomplete_details": (
                                ResponseIncompleteDetails(reason="max_output_tokens")
                                if status == "incomplete"
                                else None
                            ),
                            "usage": ResponseUsage(
                                input_tokens=usage_stats["input_tokens"],
                                output_tokens=usage_stats["output_tokens"],
                                total_tokens=total_tokens,
                            ),
                        }
                    )
                    yield _evt("response.completed", ResponsesCompletedEvent(response=completed_response))

                    # Save to store for previous_response_id
                    _responses_store.save(
                        response_id,
                        request.input if isinstance(request.input, str) else [
                            item.model_dump() if hasattr(item, "model_dump") else item
                            for item in request.input
                        ],
                        [item.model_dump() for item in all_output_items],
                    )

                    # Final sentinel
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    mx.clear_cache()
                    gc.collect()
                    sem.release()
                    print("Stream finished, cleared cache.")

            return StreamingResponse(
                stream_responses_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # ----------------------------------------------------------
            # Non-streaming response
            # ----------------------------------------------------------
            sem = get_generation_semaphore()
            await sem.acquire()
            try:
                cache_state = get_prompt_cache_state(request.model)
                result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    verbose=False,
                    vision_cache=model_cache.get("vision_cache"),
                    prompt_cache_state=cache_state,
                    **generation_kwargs,
                )
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                # Build output items (with tool call parsing)
                output_items = build_responses_output(
                    result.text, tool_parser_type, tool_module, tools,
                )

                # Determine status
                is_length = result.generation_tokens >= request.max_output_tokens
                status = "incomplete" if is_length else "completed"
                incomplete_details = (
                    ResponseIncompleteDetails(reason="max_output_tokens")
                    if status == "incomplete"
                    else None
                )

                response_obj = ResponseObject(
                    id=response_id,
                    created_at=generated_at,
                    model=request.model,
                    output=output_items,
                    status=status,
                    incomplete_details=incomplete_details,
                    instructions=request.instructions,
                    max_output_tokens=request.max_output_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    tools=tools or [],
                    tool_choice=request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    previous_response_id=request.previous_response_id,
                    metadata=request.metadata,
                    usage=ResponseUsage(
                        input_tokens=result.prompt_tokens,
                        output_tokens=result.generation_tokens,
                        total_tokens=result.total_tokens,
                    ),
                )

                # Save to store for previous_response_id support
                _responses_store.save(
                    response_obj.id,
                    request.input if isinstance(request.input, str) else [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in request.input
                    ],
                    [item.model_dump() for item in output_items],
                )

                return response_obj.model_dump()

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
            finally:
                sem.release()

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        mx.clear_cache()
        gc.collect()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post(
    "/chat/completions", response_model=None
)  # Response model handled dynamically based on stream flag
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(request: ChatRequest):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(request.model, request.adapter_path)

        images = []
        audio = []
        processed_messages = []
        for message in request.messages:
            if message.content is None:
                processed_messages.append({"role": message.role, "content": ""})
            elif isinstance(message.content, str):
                processed_messages.append(
                    {"role": message.role, "content": message.content}
                )
            elif isinstance(message.content, list):
                text_content = ""
                for item in message.content:
                    if isinstance(item, dict):
                        # Only extract images/audio from user messages
                        if message.role == "user":
                            if item["type"] == "input_image":
                                images = [item["image_url"]]
                            elif item["type"] == "image_url":
                                images = [item["image_url"]["url"]]
                            elif item["type"] == "input_audio":
                                audio.append(item["input_audio"]["data"])
                        if item["type"] in ("text", "input_text"):
                            text_content = item.get("text", "")
                processed_messages.append(
                    {"role": message.role, "content": text_content}
                )

        tools = None
        if hasattr(request, "tools"):
            tools = request.tools

        # Apply tool_choice policy
        tool_choice = getattr(request, "tool_choice", None)
        tools, tool_instruction = resolve_tool_choice(tools, tool_choice)
        if tool_instruction:
            processed_messages.insert(0, {"role": "system", "content": tool_instruction})

        tool_parser_type = None
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        if hasattr(tokenizer, "chat_template"):
            tool_parser_type = _infer_tool_parser(tokenizer.chat_template)
            if tool_parser_type is not None:
                tool_module = load_tool_module(tool_parser_type)
        template_kwargs = request.template_kwargs()
        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            tools=tools,
            **template_kwargs,
        )
        generation_kwargs = build_generation_kwargs(request, template_kwargs)

        # Resolve stop sequences to token IDs
        stop_seqs = resolve_stop_sequences(getattr(request, "stop", None))
        if stop_seqs:
            generation_kwargs["eos_tokens"] = stop_seqs

        if request.stream:
            # Streaming response
            async def stream_generator():
                sem = get_generation_semaphore()
                await sem.acquire()
                token_iterator = None
                try:
                    # Use stream_generate with prompt cache reuse
                    cache_state = get_prompt_cache_state(request.model)
                    token_iterator = stream_generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        vision_cache=model_cache.get("vision_cache"),
                        prompt_cache_state=cache_state,
                        **generation_kwargs,
                    )

                    output_text = ""
                    request_id = f"chatcmpl-{uuid.uuid4()}"
                    for chunk in token_iterator:
                        if chunk is None or not hasattr(chunk, "text"):
                            print("Warning: Received unexpected chunk format:", chunk)
                            continue

                        output_text += chunk.text

                        # Yield chunks in Server-Sent Events (SSE) format
                        usage_stats = {
                            "input_tokens": chunk.prompt_tokens,
                            "output_tokens": chunk.generation_tokens,
                            "total_tokens": chunk.prompt_tokens
                            + chunk.generation_tokens,
                            "prompt_tps": chunk.prompt_tps,
                            "generation_tps": chunk.generation_tps,
                            "peak_memory": chunk.peak_memory,
                        }

                        choices = [
                            ChatStreamChoice(
                                delta=ChatMessage(role="assistant", content=chunk.text)
                            )
                        ]
                        chunk_data = ChatStreamChunk(
                            id=request_id,
                            created=int(time.time()),
                            model=request.model,
                            usage=usage_stats,
                            choices=choices,
                        )

                        yield f"data: {chunk_data.model_dump_json()}\n\n"

                    if tool_parser_type is not None:
                        tool_calls = process_tool_calls(
                            model_output=output_text,
                            tool_module=tool_module,
                            tools=tools,
                        )
                    else:
                        tool_calls = {}
                        tool_calls["calls"] = []

                    # Signal stream end with correct finish_reason
                    stream_finish = "tool_calls" if tool_calls.get("calls") else "stop"
                    choices = [
                        ChatStreamChoice(
                            finish_reason=stream_finish,
                            delta=ChatMessage(
                                role="assistant",
                                content="",
                                tool_calls=tool_calls["calls"],
                            ),
                        )
                    ]

                    chunk_data = ChatStreamChunk(
                        id=request_id,
                        created=int(time.time()),
                        model=request.model,
                        usage=usage_stats,
                        choices=choices,
                    )
                    yield f"data: {chunk_data.model_dump_json()}\n\n"

                    yield "data: [DONE]\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    mx.clear_cache()
                    gc.collect()
                    sem.release()
                    print("Stream finished, cleared cache.")

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
            sem = get_generation_semaphore()
            await sem.acquire()
            try:
                # Use generate from generate.py
                cache_state = get_prompt_cache_state(request.model)
                gen_result = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=images,
                    audio=audio,
                    verbose=False,  # Keep API output clean
                    vision_cache=model_cache.get("vision_cache"),
                    prompt_cache_state=cache_state,
                    **generation_kwargs,
                )
                # Clean up resources
                mx.clear_cache()
                gc.collect()
                print("Generation finished, cleared cache.")

                usage_stats = UsageStats(
                    input_tokens=gen_result.prompt_tokens,
                    output_tokens=gen_result.generation_tokens,
                    total_tokens=gen_result.total_tokens,
                    prompt_tps=gen_result.prompt_tps,
                    generation_tps=gen_result.generation_tps,
                    peak_memory=gen_result.peak_memory,
                )

                if tool_parser_type is not None:
                    tool_calls = process_tool_calls(
                        model_output=gen_result.text,
                        tool_module=tool_module,
                        tools=tools,
                    )
                else:
                    tool_calls = {}
                    tool_calls["calls"] = []
                    tool_calls["remaining_text"] = gen_result.text

                finish = "tool_calls" if tool_calls.get("calls") else "stop"
                choices = [
                    ChatChoice(
                        finish_reason=finish,
                        message=ChatMessage(
                            role="assistant",
                            content=tool_calls["remaining_text"],
                            tool_calls=tool_calls["calls"],
                        ),
                    )
                ]

                result = ChatResponse(
                    model=request.model, usage=usage_stats, choices=choices
                )

                return result

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                mx.clear_cache()
                gc.collect()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
            finally:
                sem.release()

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


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
    }


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": "Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="Optional path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_SERVER_HOST,
        help="Host for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port for the HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
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
        "--kv-bits",
        type=float,
        default=0,
        help="Number of bits for KV cache quantization.",
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
        "--max-kv-size",
        type=int,
        default=0,
        help="Maximum KV size for the prompt cache (tokens).",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index (of token) for the quantized KV cache.",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=1,
        help="Maximum number of concurrent generation requests. "
        "MLX runs single-threaded on Metal; values > 1 may cause GPU errors. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on file changes (development only). "
        "WARNING: watches the entire working directory — can cause excessive memory "
        "usage with large models in repos with frequent file changes.",
    )
    args = parser.parse_args()
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["PRELOAD_MODEL"] = args.model
    if args.adapter_path:
        os.environ["PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    os.environ["MAX_CONCURRENT_REQUESTS"] = str(args.max_concurrent_requests)

    uvicorn.run(
        "mlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
