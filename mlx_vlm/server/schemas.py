import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Required, TypeAlias, TypedDict

if TYPE_CHECKING:
    from .generation import GenerationMetrics

from ..generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    normalize_resize_shape,
)
from ..generate.image import (
    DEFAULT_IMAGE_GUIDANCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGE_STEPS,
)


def get_server_max_tokens():
    return int(os.environ.get("MLX_VLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


class FlexibleBaseModel(BaseModel):
    """Base model that ignores/accepts any unknown OpenAI SDK fields."""

    model_config = ConfigDict(extra="allow")


# OpenAI API Models


class ImageGenerationRequest(FlexibleBaseModel):
    prompt: str = Field(..., description="Text prompt for image generation.")
    model: str = Field(
        "",
        description="Image generation model name or local snapshot path.",
    )
    n: int = Field(1, ge=1, le=10, description="Number of images to generate.")
    size: Optional[str] = Field(
        DEFAULT_IMAGE_SIZE,
        description="Image size as WIDTHxHEIGHT. Width/height fields override this.",
    )
    width: Optional[int] = Field(None, description="Generated image width.")
    height: Optional[int] = Field(None, description="Generated image height.")
    steps: int = Field(
        DEFAULT_IMAGE_STEPS,
        ge=1,
        description="Number of image generation inference steps.",
    )
    seed: Optional[int] = Field(
        None,
        description="Base seed. Multiple outputs use (seed + i) values.",
    )
    guidance: float = Field(
        DEFAULT_IMAGE_GUIDANCE,
        description="Classifier-free guidance scale.",
    )
    auto_json_caption: Optional[bool] = Field(
        None,
        description="For Ideogram 4, wrap plain prompts into JSON captions.",
    )
    prompt_expansion_model: Optional[str] = Field(
        None,
        description=(
            "Text model path or Hugging Face repo used to expand plain "
            "Ideogram 4 prompts into structured JSON captions."
        ),
    )
    response_format: Literal["b64_json", "path"] = Field(
        "b64_json",
        description="Return base64 PNG data or write files and return local paths.",
    )
    output_format: Literal["png"] = Field(
        "png", description="Output image format. Only PNG is currently supported."
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path. For n>1, an index is added to the stem.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for path responses.",
    )
    user: Optional[str] = Field(
        None, description="OpenAI-compatible user identifier; currently ignored."
    )


class ImageGenerationResponseData(BaseModel):
    b64_json: Optional[str] = None
    path: Optional[str] = None
    revised_prompt: Optional[str] = None
    mime_type: str = "image/png"
    width: int
    height: int
    seed: int


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageGenerationResponseData]
    output_format: Literal["png"] = "png"
    size: str


class ImageEditRequest(FlexibleBaseModel):
    prompt: str = Field(..., description="Text prompt for image editing.")
    image: Union[str, List[str]] = Field(
        ..., description="Local path or paths of reference images."
    )
    model: str = Field(..., description="Image edit model name or local snapshot path.")
    n: int = Field(1, ge=1, le=10, description="Number of images to generate.")
    size: Optional[str] = Field(
        None,
        description="Edited image size as WIDTHxHEIGHT. Width/height fields override this.",
    )
    width: Optional[int] = Field(None, description="Edited image width.")
    height: Optional[int] = Field(None, description="Edited image height.")
    steps: int = Field(
        DEFAULT_IMAGE_STEPS,
        ge=1,
        description="Number of image edit inference steps.",
    )
    seed: Optional[int] = Field(
        None,
        description="Base seed. Multiple outputs use (seed + i) values.",
    )
    guidance: float = Field(
        DEFAULT_IMAGE_GUIDANCE,
        description="Classifier-free guidance scale.",
    )
    response_format: Literal["b64_json", "path"] = Field(
        "b64_json",
        description="Return base64 PNG data or write files and return local paths.",
    )
    output_format: Literal["png"] = Field(
        "png", description="Output image format. Only PNG is currently supported."
    )
    output_path: Optional[str] = Field(
        None,
        description="Output file path. For n>1, an index is added to the stem.",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for path responses.",
    )
    user: Optional[str] = Field(
        None, description="OpenAI-compatible user identifier; currently ignored."
    )

    @field_validator("image")
    @classmethod
    def validate_image(cls, value):
        images = [value] if isinstance(value, str) else list(value)
        if not images:
            raise ValueError("At least one image is required.")
        if not all(isinstance(item, str) and item for item in images):
            raise ValueError("Image paths must be non-empty strings.")
        return value


class ImageEditResponseData(BaseModel):
    b64_json: Optional[str] = None
    path: Optional[str] = None
    revised_prompt: Optional[str] = None
    mime_type: str = "image/png"
    width: int
    height: int
    seed: int


class ImageEditResponse(BaseModel):
    created: int
    data: List[ImageEditResponseData]
    output_format: Literal["png"] = "png"
    size: str


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


class VideoUrl(TypedDict, total=False):
    url: Required[str]


class ResponseInputVideoParam(TypedDict, total=False):
    type: Required[Literal["input_video"]]
    video_url: Required[Union[str, VideoUrl]]
    video: Optional[str]


class ResponseVideoUrlParam(TypedDict, total=False):
    type: Required[Literal["video_url"]]
    video_url: Required[Union[str, VideoUrl]]


class ResponseVideoParam(TypedDict, total=False):
    type: Required[Literal["video"]]
    video: Required[str]


ResizeShapeInput: TypeAlias = Union[Tuple[int], Tuple[int, int]]

ResponseInputContentParam: TypeAlias = Union[
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseImageUrlParam,
    ResponseInputAudioParam,
    ResponseInputVideoParam,
    ResponseVideoUrlParam,
    ResponseVideoParam,
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
    reasoning_content: Optional[str] = Field(
        None,
        description="Thinking/reasoning content (when thinking is enabled).",
    )
    reasoning: Optional[str] = Field(
        None,
        description=(
            "Deprecated alias for reasoning_content, kept for backward compatibility."
        ),
    )
    tool_calls: Optional[List[Any]] = Field(
        None, description="Tool calls made by the assistant."
    )
    tool_call_id: Optional[str] = Field(
        None, description="ID of the tool call this message is a response to."
    )
    name: Optional[str] = Field(None, description="Name of the tool/function.")

    @model_validator(mode="after")
    def sync_reasoning_aliases(self):
        if self.reasoning_content is None and self.reasoning is not None:
            self.reasoning_content = self.reasoning
        elif self.reasoning is None and self.reasoning_content is not None:
            self.reasoning = self.reasoning_content
        return self


class OpenAIRequest(FlexibleBaseModel):
    """
    OpenAI-compatible request structure.
    Using this structure : https://github.com/openai/openai-python/blob/main/src/openai/resources/responses/responses.py
    """

    input: Union[str, List[Any]] = Field(
        ..., description="Input text or list of chat messages."
    )
    model: str = Field(..., description="The model to use for generation.")
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
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
    thinking_end_token: Optional[str] = Field(None, description="Thinking end token.")
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


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class OpenAIUsage(BaseModel):
    """Token usage details including input tokens, output tokens, breakdown, and total tokens used."""

    input_tokens: int
    input_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails
    )
    output_tokens: int
    total_tokens: int

    @classmethod
    def from_metrics(
        cls, metrics: "GenerationMetrics", input_tokens: int, output_tokens: int
    ) -> "OpenAIUsage":
        # Per spec, `input_tokens` is the total prompt count; cached portion is
        # reported separately in `input_tokens_details.cached_tokens`.
        return cls(
            input_tokens=input_tokens,
            input_tokens_details=PromptTokensDetails(
                cached_tokens=metrics.cached_tokens
            ),
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )


class GenerationTimings(BaseModel):
    """Per-request timing breakdown."""

    prompt_n: int
    cache_n: int
    predicted_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float
    peak_memory: float = 0.0

    @staticmethod
    def _derive_gen_tps(token_times: List[float]) -> Optional[float]:
        if len(token_times) < 2:
            return None
        elapsed = token_times[-1] - token_times[0]
        return (len(token_times) - 1) / elapsed if elapsed > 0 else None

    @classmethod
    def from_metrics(
        cls,
        metrics: "GenerationMetrics",
        prompt_tokens: int,
        output_tokens: int,
    ) -> "GenerationTimings":
        generation_tps = metrics.generation_tps or cls._derive_gen_tps(
            metrics.token_times
        )
        cached_tokens = metrics.cached_tokens
        prompt_n = max(0, int(prompt_tokens) - int(cached_tokens))
        prompt_s = prompt_tokens / metrics.prompt_tps if metrics.prompt_tps else 0.0
        prompt_ms = prompt_s * 1000.0
        predicted_ms = (
            output_tokens / generation_tps * 1000.0 if generation_tps else 0.0
        )
        return cls(
            prompt_n=prompt_n,
            cache_n=int(cached_tokens),
            predicted_n=int(output_tokens),
            prompt_ms=prompt_ms,
            prompt_per_token_ms=(prompt_ms / prompt_n) if prompt_n else 0.0,
            prompt_per_second=(prompt_n / prompt_s) if prompt_s else 0.0,
            predicted_ms=predicted_ms,
            predicted_per_token_ms=(
                predicted_ms / output_tokens if output_tokens else 0.0
            ),
            predicted_per_second=float(generation_tps or 0.0),
            peak_memory=float(metrics.peak_memory or 0.0),
        )


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
        ...,
        description="The path to the local model directory or Hugging Face repo.",
    )
    adapter_path: Optional[str] = Field(
        None, description="The path to the adapter weights."
    )
    max_tokens: int = Field(
        default_factory=get_server_max_tokens,
        description="Maximum number of tokens to generate.",
    )
    max_denoising_steps: Optional[int] = Field(
        None,
        description="Maximum denoising steps for diffusion generation.",
    )
    block_length: Optional[int] = Field(
        None,
        description="Block length for masked diffusion text generation.",
    )
    num_to_transfer: Optional[int] = Field(
        None,
        description="Target number of masked tokens to transfer per diffusion step.",
    )
    max_transfer_per_step: Optional[int] = Field(
        None,
        description="Maximum masked tokens to transfer per diffusion step.",
    )
    editing_threshold: Optional[float] = Field(
        None,
        description="Confidence threshold for masked diffusion post-fill edits.",
    )
    max_post_steps: Optional[int] = Field(
        None,
        description="Maximum masked diffusion post-fill editing steps per block.",
    )
    stability_steps: Optional[int] = Field(
        None,
        description="Stop masked diffusion post-fill refinement after stable steps.",
    )
    diffusion_full_canvas: Optional[bool] = Field(
        None,
        description="Use the checkpoint canvas length for diffusion generation.",
    )
    diffusion_min_canvas_length: Optional[int] = Field(
        None,
        description="Minimum active canvas length for diffusion generation.",
    )
    diffusion_max_canvas_length: Optional[int] = Field(
        None,
        description="Maximum active canvas length for diffusion generation.",
    )
    diffusion_sampler: Optional[Literal["entropy-bound", "confidence-threshold"]] = (
        Field(
            None,
            description="Canvas update sampler for diffusion generation.",
        )
    )
    threshold: Optional[float] = Field(
        None,
        description="Token probability threshold for diffusion confidence transfer.",
    )
    min_threshold: Optional[float] = Field(
        None,
        description="Lowest token probability threshold for masked diffusion transfer.",
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
    thinking_end_token: Optional[str] = Field(None, description="Thinking end token.")
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


class UsageStats(BaseModel):
    """OpenAI-compatible usage statistics for chat completions. Throughput and
    memory metrics live in `GenerationTimings` (sibling `timings` field on the
    response) to keep this object spec-clean."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails
    )

    @classmethod
    def from_metrics(
        cls, metrics: "GenerationMetrics", prompt_tokens: int, completion_tokens: int
    ) -> "UsageStats":
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=metrics.cached_tokens
            ),
        )


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(GenerationRequest):
    messages: List[ChatMessage]
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[Any]] = Field(None, description="Tools the model may call.")
    tool_choice: Optional[Any] = Field(
        None,
        description=(
            "Controls tool use: none, auto, required, or a specific function."
        ),
    )


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
    timings: Optional[GenerationTimings] = None


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
    timings: Optional[GenerationTimings] = None


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
    thinking_end_token: Optional[str] = None
    response_format: Optional[Any] = None


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_metrics(
        cls, metrics: "GenerationMetrics", prompt_tokens: int, output_tokens: int
    ) -> "AnthropicUsage":
        # Per spec, `input_tokens` excludes the cached portion, which is
        # reported via `cache_read_input_tokens`. We don't currently distinguish
        # cache creation from reads, so `cache_creation_input_tokens` stays 0.
        cached_tokens = max(0, int(metrics.cached_tokens))
        return cls(
            input_tokens=max(0, int(prompt_tokens) - cached_tokens),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            output_tokens=int(output_tokens),
        )


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
