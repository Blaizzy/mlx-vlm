"""Pydantic models for the OpenAI Responses API (/v1/responses).

This module defines all request, response, and streaming event models
for the OpenAI-compatible Responses endpoint. Models are self-contained
to avoid circular imports with server.py.

Reference: https://developers.openai.com/api/reference/resources/responses
"""

import uuid
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Required, TypeAlias, TypedDict

# ---------------------------------------------------------------------------
# Constants (mirrored from mlx_vlm.generate to avoid heavy imports)
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"


# ---------------------------------------------------------------------------
# Base classes (duplicated from server.py for import independence)
# ---------------------------------------------------------------------------


class FlexibleBaseModel(BaseModel):
    """Base model that silently accepts unknown fields for forward compatibility."""

    model_config = ConfigDict(extra="allow")

    def dump_kwargs(self, *fields: str) -> dict[str, Any]:
        """Return a dict of the requested fields, omitting ``None`` values."""
        return {
            key: getattr(self, key)
            for key in fields
            if hasattr(self, key) and getattr(self, key) is not None
        }


class GenerationParams(FlexibleBaseModel):
    """Sampling parameters shared across endpoints."""

    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: Optional[int] = Field(None, description="Top-k sampling cutoff.")
    min_p: Optional[float] = Field(None, description="Min-p sampling threshold.")
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
    """Chat template parameters (thinking mode, etc.)."""

    enable_thinking: Optional[bool] = Field(
        None, description="Enable thinking mode in the chat template."
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


# ---------------------------------------------------------------------------
# Input content types (TypedDicts matching OpenAI SDK)
# ---------------------------------------------------------------------------


class ResponseInputTextParam(TypedDict, total=False):
    """Text content item — accepts both ``input_text`` and ``text`` types."""

    text: Required[str]
    type: Required[Literal["input_text", "text"]]


class ResponseInputImageParam(TypedDict, total=False):
    """Image content item with a direct image URL."""

    detail: Literal["high", "low", "auto"]
    type: Required[Literal["input_image"]]
    image_url: Required[str]
    file_id: Optional[str]


class InputAudio(TypedDict, total=False):
    data: Required[str]
    format: Required[str]


class ResponseInputAudioParam(TypedDict, total=False):
    """Audio content item."""

    type: Required[Literal["input_audio"]]
    input_audio: Required[InputAudio]


class ImageUrl(TypedDict, total=False):
    url: Required[str]


class ResponseImageUrlParam(TypedDict, total=False):
    """Image content item with nested ``image_url.url`` (chat/completions format)."""

    type: Required[Literal["image_url"]]
    image_url: Required[ImageUrl]


class ResponseOutputText(TypedDict, total=False):
    """Output text item used in multi-turn assistant messages."""

    text: Required[str]
    type: Required[Literal["output_text"]]


ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

ResponseInputMessageContentListParam: TypeAlias = List[ResponseInputContentParam]
ResponseOutputMessageContentList: TypeAlias = List[ResponseOutputText]


# ---------------------------------------------------------------------------
# Chat message model
# ---------------------------------------------------------------------------


class ChatMessage(FlexibleBaseModel):
    """A single message in the conversation input."""

    role: Literal["user", "assistant", "system", "developer", "tool"] = Field(
        ..., description="Role of the message sender."
    )
    content: Optional[
        Union[
            str,
            ResponseInputMessageContentListParam,
            ResponseOutputMessageContentList,
        ]
    ] = Field(None, description="Content of the message.")
    tool_calls: List = []


# ---------------------------------------------------------------------------
# Function tool definition
# ---------------------------------------------------------------------------


class ResponseFunctionTool(BaseModel):
    """A function tool the model may call."""

    type: Literal["function"] = "function"
    name: str = Field(..., description="The name of the function.")
    description: Optional[str] = Field(
        None, description="A description of what the function does."
    )
    parameters: Optional[dict] = Field(
        None, description="JSON Schema object describing the function parameters."
    )
    strict: Optional[bool] = Field(
        None, description="Whether to enforce strict schema adherence."
    )


# ---------------------------------------------------------------------------
# Function call input items (for multi-turn tool use)
# ---------------------------------------------------------------------------


class ResponseFunctionCallInputItem(BaseModel):
    """A function call from a previous assistant turn, included in input."""

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(..., description="Unique ID for this tool call.")
    name: str = Field(..., description="The function name that was called.")
    arguments: str = Field(..., description="JSON string of the function arguments.")
    status: Optional[str] = "completed"


class ResponseFunctionCallOutputInputItem(BaseModel):
    """The output/result of a function call, sent back by the client."""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(
        ..., description="The call_id of the function call this is a result for."
    )
    output: str = Field(..., description="The function output as a string.")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class ResponsesRequest(GenerationParams, TemplateParams):
    """OpenAI Responses API request body.

    Reference: https://developers.openai.com/api/reference/resources/responses/create
    """

    input: Union[str, List[Any]] = Field(
        ..., description="Input text or list of input items (messages, tool outputs)."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate."
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )
    tools: Optional[List[dict]] = Field(
        None, description="Tool definitions the model may call."
    )
    tool_choice: Optional[Any] = Field(
        "auto", description='Tool choice: "none", "auto", "required", or specific tool.'
    )
    parallel_tool_calls: bool = Field(
        True, description="Allow parallel tool calls."
    )
    previous_response_id: Optional[str] = Field(
        None,
        description="ID of a previous response for multi-turn context replay.",
    )
    instructions: Optional[str] = Field(
        None,
        description="System/developer message inserted into context.",
    )
    metadata: Optional[dict] = Field(
        None, description="Up to 16 key-value pairs of metadata."
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Up to 4 sequences where the API will stop generating further tokens.",
    )
    response_format: Optional[dict] = Field(
        None,
        description='Output format: {"type": "text"} or {"type": "json_object"}.',
    )
    prompt_cache_key: Optional[str] = Field(
        None,
        description="Stable key for prompt cache routing across turns.",
    )

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs = self.dump_kwargs("max_output_tokens")
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
        return {**kwargs, **self.shared_generation_kwargs()}


# ---------------------------------------------------------------------------
# Output item models
# ---------------------------------------------------------------------------


class ContentPartOutputText(BaseModel):
    """A text content part in an output message."""

    type: Literal["output_text"] = "output_text"
    text: str = ""
    annotations: List[str] = []


class ResponseMessageItem(BaseModel):
    """An assistant message output item."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed"] = "completed"
    content: List[ContentPartOutputText] = []


class ResponseFunctionCallItem(BaseModel):
    """A function call output item."""

    type: Literal["function_call"] = "function_call"
    id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex[:24]}")
    call_id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    name: str = Field(..., description="The function name being called.")
    arguments: str = Field(..., description="JSON string of the function arguments.")
    status: Literal["completed"] = "completed"


class ResponseIncompleteDetails(BaseModel):
    """Details about why a response is incomplete."""

    reason: Literal["max_output_tokens", "content_filter"]


# ---------------------------------------------------------------------------
# Usage and error models
# ---------------------------------------------------------------------------


class ResponseUsage(BaseModel):
    """Token usage details."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseErrorObject(BaseModel):
    """Error object returned when the model fails to generate a Response."""

    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None
    type: Optional[str] = None


# ---------------------------------------------------------------------------
# Response object
# ---------------------------------------------------------------------------


class ResponseObject(BaseModel):
    """The top-level Response object returned by /v1/responses.

    Reference: https://developers.openai.com/api/reference/resources/responses/object
    """

    id: str = Field(
        default_factory=lambda: f"resp_{uuid.uuid4().hex[:24]}",
        description="Unique identifier for this Response.",
    )
    object: Literal["response"] = Field(
        "response", description="The object type — always ``response``."
    )
    created_at: int = Field(..., description="Unix timestamp of creation.")
    status: Literal["completed", "failed", "in_progress", "incomplete"] = Field(
        "completed", description="The status of the response generation."
    )
    error: Optional[ResponseErrorObject] = Field(None)
    incomplete_details: Optional[ResponseIncompleteDetails] = Field(None)
    instructions: Optional[str] = Field(None)
    max_output_tokens: Optional[int] = Field(None)
    model: str = Field(..., description="Model ID used to generate the response.")
    output: List[Union[ResponseMessageItem, ResponseFunctionCallItem]] = Field(
        default_factory=list,
        description="An array of content items generated by the model.",
    )
    parallel_tool_calls: bool = Field(True)
    previous_response_id: Optional[str] = Field(None)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    tools: List = Field(default_factory=list)
    tool_choice: Optional[Any] = Field("auto")
    truncation: Literal["auto", "disabled"] = Field("disabled")
    metadata: Optional[dict] = Field(None)
    usage: ResponseUsage = Field(..., description="Token usage details.")
    user: Optional[str] = Field(None)

    @property
    def output_text(self) -> str:
        """Aggregate text from all output_text content parts."""
        parts = []
        for item in self.output:
            if isinstance(item, ResponseMessageItem):
                for part in item.content:
                    if part.type == "output_text" and part.text:
                        parts.append(part.text)
        return "".join(parts) or ""


# ---------------------------------------------------------------------------
# Streaming event models
# ---------------------------------------------------------------------------


class BaseStreamEvent(BaseModel):
    """Base class for all SSE streaming events."""

    type: str
    sequence_number: int = 0


class ResponseCreatedEvent(BaseStreamEvent):
    type: Literal["response.created"] = "response.created"
    response: ResponseObject


class ResponseInProgressEvent(BaseStreamEvent):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: ResponseObject


class ResponseOutputItemAddedEvent(BaseStreamEvent):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: Union[ResponseMessageItem, ResponseFunctionCallItem]


class ResponseContentPartAddedEvent(BaseStreamEvent):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputTextDeltaEvent(BaseStreamEvent):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputTextDoneEvent(BaseStreamEvent):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str
    output_index: int
    content_index: int
    text: str


class ResponseContentPartDoneEvent(BaseStreamEvent):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int
    content_index: int
    part: ContentPartOutputText


class ResponseOutputItemDoneEvent(BaseStreamEvent):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: Union[ResponseMessageItem, ResponseFunctionCallItem]


class ResponseFunctionCallArgumentsDeltaEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.delta"] = (
        "response.function_call_arguments.delta"
    )
    item_id: str
    output_index: int
    delta: str


class ResponseFunctionCallArgumentsDoneEvent(BaseStreamEvent):
    type: Literal["response.function_call_arguments.done"] = (
        "response.function_call_arguments.done"
    )
    item_id: str
    output_index: int
    arguments: str


class ResponseCompletedEvent(BaseStreamEvent):
    type: Literal["response.completed"] = "response.completed"
    response: ResponseObject


StreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseContentPartAddedEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseContentPartDoneEvent,
    ResponseOutputItemDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseCompletedEvent,
]
