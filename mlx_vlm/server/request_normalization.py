"""Normalize compatible API requests into server generation arguments."""

from typing import Optional, Tuple, Union

from ..generate import (
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from ..structured import build_json_schema_logits_processor
from .generation import (
    GenerationArguments,
    get_server_enable_thinking,
    get_server_max_tokens,
    get_server_thinking_budget,
    get_server_thinking_end_token,
    get_server_thinking_start_token,
)
from .runtime import runtime

_DISABLED_REASONING_EFFORTS = {"none", "off", "disabled", "false", "0"}


def _request_field_is_set(request, field_name: str) -> bool:
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None:
        return field_name in fields_set
    return getattr(request, field_name, None) is not None


def _request_field_or_default(request, field_name: str, default):
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None and field_name not in fields_set:
        return default
    value = getattr(request, field_name, default)
    return default if value is None else value


def _reasoning_effort_enabled(effort) -> Tuple[Optional[bool], Optional[str]]:
    if effort is None:
        return None, None
    normalized = str(effort).strip().lower()
    if not normalized:
        return None, None
    return normalized not in _DISABLED_REASONING_EFFORTS, normalized


def _standard_reasoning_control(
    request,
) -> Tuple[Optional[bool], Optional[str], bool]:
    """Normalize OpenAI reasoning fields into a thinking-mode decision."""
    if _request_field_is_set(request, "reasoning"):
        reasoning = getattr(request, "reasoning", None)
        if hasattr(reasoning, "model_dump"):
            reasoning = reasoning.model_dump(exclude_none=True)
        if isinstance(reasoning, dict):
            enabled, effort = _reasoning_effort_enabled(reasoning.get("effort"))
            return (True if enabled is None else enabled), effort, True
        if isinstance(reasoning, bool):
            return reasoning, None, True
        enabled, effort = _reasoning_effort_enabled(reasoning)
        if enabled is not None:
            return enabled, effort, True

    if _request_field_is_set(request, "reasoning_effort"):
        enabled, effort = _reasoning_effort_enabled(
            getattr(request, "reasoning_effort", None)
        )
        if enabled is not None:
            return enabled, effort, True

    return None, None, False


def _model_config_field_or_default(processor, field_name: str, default):
    config = runtime.model_cache.get("config")
    if config is None and processor is not None:
        config = getattr(processor, "config", None)
    return getattr(config, field_name, default)


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _extract_response_format_schema(request) -> Optional[Union[str, dict]]:
    response_format = _as_plain_dict(getattr(request, "response_format", None))

    text_config = _as_plain_dict(getattr(request, "text", None))
    if response_format is None and isinstance(text_config, dict):
        response_format = _as_plain_dict(text_config.get("format"))

    if response_format is None:
        return None

    format_type = response_format.get("type")
    if format_type in (None, "text"):
        return None
    if format_type in ("json_object", "object"):
        return {"type": "object"}
    if format_type != "json_schema":
        raise ValueError(f"Unsupported response_format type: {format_type!r}")

    json_schema = _as_plain_dict(response_format.get("json_schema"))
    if json_schema is None:
        # Responses API text.format places schema directly on the format object.
        json_schema = response_format

    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    if schema is None:
        raise ValueError("response_format json_schema must include a schema field")
    return schema


def _build_structured_logits_processors(
    request,
    processor,
    logits_processor_factory=build_json_schema_logits_processor,
):
    schema = _extract_response_format_schema(request)
    if schema is None:
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    return [logits_processor_factory(tokenizer, schema)]


def _build_gen_args(
    request,
    processor=None,
    tenant_id: Optional[str] = None,
    structured_logits_processor_builder=_build_structured_logits_processors,
) -> GenerationArguments:
    """Build generation arguments from a compatible API request."""
    max_tokens = getattr(request, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(request, "max_output_tokens", None)
    if max_tokens is None:
        max_tokens = get_server_max_tokens()
    logit_bias = getattr(request, "logit_bias", None)
    if logit_bias is not None and isinstance(logit_bias, dict):
        logit_bias = {int(k): v for k, v in logit_bias.items()}
    standard_reasoning, reasoning_effort, has_standard_reasoning = (
        _standard_reasoning_control(request)
    )
    server_enable_thinking = get_server_enable_thinking()
    if _request_field_is_set(request, "enable_thinking"):
        enable_thinking = bool(getattr(request, "enable_thinking", False))
        template_reasoning = enable_thinking
    elif has_standard_reasoning:
        enable_thinking = bool(standard_reasoning)
        template_reasoning = enable_thinking
    else:
        enable_thinking = server_enable_thinking
        # Preserve a model template's native default when the server default is
        # off and the request did not express a reasoning preference.
        template_reasoning = True if server_enable_thinking else None
    default_temperature = _model_config_field_or_default(
        processor, "temperature", DEFAULT_TEMPERATURE
    )
    default_top_p = _model_config_field_or_default(processor, "top_p", DEFAULT_TOP_P)
    default_top_k = _model_config_field_or_default(processor, "top_k", 0)
    if _model_config_field_or_default(processor, "do_sample", None) is False:
        default_temperature = 0.0
    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=_request_field_or_default(
            request, "temperature", default_temperature
        ),
        top_p=_request_field_or_default(request, "top_p", default_top_p),
        top_k=_request_field_or_default(request, "top_k", default_top_k),
        min_p=getattr(request, "min_p", 0.0),
        top_n_sigma=getattr(request, "top_n_sigma", 0.0),
        p_less=getattr(request, "p_less", False),
        typical_p=getattr(request, "typical_p", 1.0),
        seed=getattr(request, "seed", None),
        logprobs=bool(getattr(request, "logprobs", False)),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        repetition_context_size=_request_field_or_default(
            request,
            "repetition_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        presence_penalty=getattr(request, "presence_penalty", None),
        presence_context_size=_request_field_or_default(
            request,
            "presence_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        frequency_penalty=getattr(request, "frequency_penalty", None),
        frequency_context_size=_request_field_or_default(
            request,
            "frequency_context_size",
            DEFAULT_REPETITION_CONTEXT_SIZE,
        ),
        max_denoising_steps=_request_field_or_default(
            request, "max_denoising_steps", None
        ),
        block_length=_request_field_or_default(request, "block_length", None),
        num_to_transfer=_request_field_or_default(request, "num_to_transfer", None),
        max_transfer_per_step=_request_field_or_default(
            request, "max_transfer_per_step", None
        ),
        editing_threshold=_request_field_or_default(request, "editing_threshold", None),
        max_post_steps=_request_field_or_default(request, "max_post_steps", None),
        stability_steps=_request_field_or_default(request, "stability_steps", None),
        diffusion_full_canvas=_request_field_or_default(
            request, "diffusion_full_canvas", None
        ),
        diffusion_min_canvas_length=_request_field_or_default(
            request, "diffusion_min_canvas_length", None
        ),
        diffusion_max_canvas_length=_request_field_or_default(
            request, "diffusion_max_canvas_length", None
        ),
        diffusion_sampler=_request_field_or_default(request, "diffusion_sampler", None),
        threshold=_request_field_or_default(request, "threshold", None),
        min_threshold=_request_field_or_default(request, "min_threshold", None),
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        reasoning=template_reasoning,
        reasoning_effort=reasoning_effort,
        thinking_budget=_request_field_or_default(
            request, "thinking_budget", get_server_thinking_budget()
        ),
        thinking_start_token=_request_field_or_default(
            request, "thinking_start_token", get_server_thinking_start_token()
        ),
        thinking_end_token=_request_field_or_default(
            request, "thinking_end_token", get_server_thinking_end_token()
        ),
        tenant_id=tenant_id,
    )
    if processor is not None:
        args.logits_processors = structured_logits_processor_builder(request, processor)
    return args
