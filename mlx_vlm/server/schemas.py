import os
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypeAlias, TypedDict

from ..generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    normalize_resize_shape,
)


def get_server_max_tokens():
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")


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

    One of `high`, `low`, or `auto`. Defaults to `auto`.
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


ResizeShapeInput: TypeAlias = Union[Tuple[int], Tuple[int, int]]

ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
]

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
        description="Role of the message sender.",
    )
    content: Union[
        str,
        None,
        ResponseInputMessageContentListParam,
        ResponseOutputMessageContentList,
    ] = Field(None, description="Content of the message.")
    reasoning: Optional[str] = Field(
        None, description="Thinking/reasoning content (when thinking is enabled)."
    )
    tool_calls: Optional[List[Any]] = Field(
        None, description="Tool calls made by the assistant."
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is a response to."
    )
    name: Optional[str] = Field(None, description="Name of the tool/function.")


class OpenAIRequest(FlexibleBaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[Any]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    max_output_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    repetition_context_size: Optional[int] = Field(
        None, description="Repetition penalty context size."
    )
    presence_penalty: Optional[float] = Field(None, description="Presence penalty.")
    presence_context_size: Optional[int] = Field(
        None, description="Presence penalty context size."
    )
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty.")
    frequency_context_size: Optional[int] = Field(
        None, description="Frequency penalty context size."
    )
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = Field(
        None,
        description=(
            "Override server thinking mode for this request. If omitted, the "
            "server default set by --enable-thinking is used."
        ),
    )
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )
    response_format: Optional[Any] = Field(
        None, description="OpenAI-compatible response_format for structured outputs."
    )
    text: Optional[Any] = Field(
        None, description="Responses API text format configuration."
    )
    instructions: Optional[str] = Field(
        None, description="System/developer instructions for this response."
    )
    previous_response_id: Optional[str] = Field(
        None,
        description="ID of a previous response whose input/output items should be included.",
    )
    tools: Optional[List[Any]] = Field(
        None, description="Responses API tool definitions."
    )
    tool_choice: Optional[Any] = Field(None, description="Tool choice policy.")
    store: Optional[bool] = Field(
        True, description="Whether to store this response for later retrieval."
    )


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
    previous_response_id: Optional[str] = Field(
        None, description="ID of the previous response used for this response."
    )
    store: Optional[bool] = Field(
        True, description="Whether this response is stored for later retrieval."
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
    item: Any


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
    item: Any


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


class VLMRequest(FlexibleBaseModel):
    model: str = Field(
        DEFAULT_MODEL_PATH,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    min_p: float = Field(0.0, description="Min-p sampling.")
    seed: int = Field(DEFAULT_SEED, description="Seed for random generation.")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    repetition_context_size: Optional[int] = Field(
        None, description="Repetition penalty context size."
    )
    presence_penalty: Optional[float] = Field(None, description="Presence penalty.")
    presence_context_size: Optional[int] = Field(
        None, description="Presence penalty context size."
    )
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty.")
    frequency_context_size: Optional[int] = Field(
        None, description="Frequency penalty context size."
    )
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = Field(
        None,
        description=(
            "Override server thinking mode for this request. If omitted, the "
            "server default set by --enable-thinking is used."
        ),
    )
    thinking_budget: Optional[int] = Field(None, description="Max thinking tokens.")
    thinking_start_token: Optional[str] = Field(
        None, description="Thinking start token."
    )
    logprobs: Optional[bool] = Field(
        None,
        description="Return log-probabilities for each output token.",
    )
    top_logprobs: Optional[int] = Field(
        None,
        description=(
            "Number of most-likely tokens to return at each position "
            "(0-20). Requires logprobs=true. The server-side cap is set by "
            "the TOP_LOGPROBS_K env var; values above the cap are clamped."
        ),
    )
    resize_shape: Optional[ResizeShapeInput] = Field(
        None,
        description="Resize shape for the image. Provide one integer for square or two for (height, width).",
    )
    response_format: Optional[Any] = Field(
        None, description="OpenAI-compatible response_format for structured outputs."
    )

    @field_validator("resize_shape", mode="before")
    @classmethod
    def normalize_resize_shape_field(cls, value):
        return normalize_resize_shape(value)


class GenerationRequest(VLMRequest):
    """
    Inherits from VLMRequest and adds additional fields for the generation request.
    """

    stream: bool = Field(
        False, description="Whether to stream the response chunk by chunk."
    )


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class UsageStats(BaseModel):
    """OpenAI-compatible usage statistics for chat completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = PromptTokensDetails()
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]


class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None


class ChatLogprobContent(BaseModel):
    token: str
    logprob: float
    bytes: Optional[List[int]] = None
    top_logprobs: List[TopLogprob] = []


class ChatLogprobs(BaseModel):
    content: List[ChatLogprobContent] = []


class ChatChoice(BaseModel):
    index: int = 0
    finish_reason: str = "stop"
    message: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = []
    usage: Optional[UsageStats] = None


class ChatStreamChoice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    delta: ChatMessage
    logprobs: Optional[ChatLogprobs] = None


class ChatStreamChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[ChatStreamChoice] = []
    usage: Optional[UsageStats] = None


# Models for Anthropic-compatible /v1/messages endpoint


class AnthropicMessageParam(FlexibleBaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Any]]


class AnthropicRequest(FlexibleBaseModel):
    model: str = Field(..., description="The model to use for generation.")
    messages: List[AnthropicMessageParam]
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    system: Optional[Union[str, List[Any]]] = None
    stream: bool = False
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for sampling."
    )
    top_p: float = Field(DEFAULT_TOP_P, description="Top-p sampling.")
    top_k: int = Field(0, description="Top-k sampling.")
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    metadata: Optional[Any] = None
    thinking: Optional[Any] = None
    output_config: Optional[Any] = None
    adapter_path: Optional[str] = None
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty.")
    repetition_context_size: Optional[int] = Field(
        None, description="Repetition penalty context size."
    )
    presence_penalty: Optional[float] = Field(None, description="Presence penalty.")
    presence_context_size: Optional[int] = Field(
        None, description="Presence penalty context size."
    )
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty.")
    frequency_context_size: Optional[int] = Field(
        None, description="Frequency penalty context size."
    )
    logit_bias: Optional[Any] = Field(None, description="Logit bias dict.")
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    thinking_start_token: Optional[str] = None
    response_format: Optional[Any] = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Any]
    model: str
    stop_reason: Optional[
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
    ] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# Models for /models endpoint


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int


class ModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[ModelInfo]
