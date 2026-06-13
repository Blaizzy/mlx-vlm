import time

from huggingface_hub import scan_cache_dir

from ..generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generate,
    stream_generate,
)
from ..prompt_utils import apply_chat_template, extract_text_from_content
from ..structured import build_json_schema_logits_processor
from ..tool_parsers import _infer_tool_parser_from_processor, load_tool_module
from ..version import __version__
from ..vision_cache import VisionFeatureCache
from . import app as _app_module
from .anthropic import (
    _anthropic_content_blocks_to_text_and_tools,
    _anthropic_content_from_generation,
    _anthropic_error_response,
    _anthropic_image_source_to_ref,
    _anthropic_messages_to_internal,
    _anthropic_request_with_derived_fields,
    _anthropic_stop_reason,
    _anthropic_system_text,
    _anthropic_tool_choice_to_openai,
    _anthropic_tool_result_content_to_openai,
    _anthropic_tool_result_content_to_text,
    _anthropic_tool_to_openai,
    _anthropic_tool_use_to_openai,
    _anthropic_tools_to_openai,
    _apply_stop_sequences,
    _openai_tool_call_to_anthropic,
    _sse_event,
    anthropic_count_tokens_endpoint,
    anthropic_messages_endpoint,
)
from .audio import (
    AudioInferenceHandle,
    AudioRequestQueue,
    AudioResultChunk,
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    audio_speech_endpoint,
    audio_transcriptions_endpoint,
    audio_translations_endpoint,
)
from .generation import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_SPECULATIVE_BATCH_COALESCE_MS,
    DEFAULT_TOKEN_QUEUE_TIMEOUT,
    METRICS_HISTORY_LIMIT,
    METRICS_RECENT_LIMIT,
    BatchGenerator,
    GenerationArguments,
    GenerationContext,
    PromptTooLongError,
    ResponseGenerator,
    ServerMetricsStore,
    StreamingToken,
    _build_metrics_envelope,
    _check_configured_context_budget,
    _count_prompt_tokens,
    _get_draft_block_size_from_env,
    _make_cache,
    get_configured_context_limit,
    get_kv_group_size,
    get_kv_quant_scheme,
    get_max_kv_size,
    get_prefill_step_size,
    get_quantized_kv_bits,
    get_quantized_kv_start,
    get_server_enable_thinking,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
    get_speculative_batch_coalesce_s,
    get_token_queue_timeout,
    get_top_logprobs_k,
    load_model_resources,
    make_streaming_detokenizer,
    run_speculative_server_rounds,
)
from .openai import (
    chat_completions_endpoint,
    responses_cancel_endpoint,
    responses_delete_endpoint,
    responses_endpoint,
    responses_input_items_endpoint,
    responses_input_tokens_endpoint,
    responses_retrieve_endpoint,
)
from .responses_state import (
    RESPONSE_STORE_LIMIT,
    StoredResponse,
    ThinkingStreamDelta,
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
    response_store_order,
    suppress_tool_call_content,
)
from .runtime import ModelCacheRegistry, runtime
from .schemas import (
    AnthropicMessageParam,
    AnthropicMessageResponse,
    AnthropicRequest,
    AnthropicUsage,
    BaseStreamEvent,
    ChatChoice,
    ChatLogprobContent,
    ChatLogprobs,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
    ContentPartOutputText,
    FlexibleBaseModel,
    GenerationRequest,
    GenerationTimings,
    ImageUrl,
    InputAudio,
    MessageItem,
    ModelInfo,
    ModelsResponse,
    OpenAIErrorObject,
    OpenAIRequest,
    OpenAIResponse,
    OpenAIUsage,
    PromptTokensDetails,
    ResizeShapeInput,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseImageUrlParam,
    ResponseInProgressEvent,
    ResponseInputAudioParam,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
    ResponseInputVideoParam,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessageContentList,
    ResponseOutputText,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    ResponseVideoUrlParam,
    StreamEvent,
    TopLogprob,
    UsageStats,
    VLMRequest,
)


def __getattr__(name):
    return getattr(_app_module, name)


for _name in dir(_app_module):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_app_module, _name)


__all__ = [
    _name
    for _name in globals()
    if _name != "_app_module" and not _name.startswith("__")
]
