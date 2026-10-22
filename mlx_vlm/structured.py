import json
import logging
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_LLGUIDANCE_MASK_KERNEL = mx.fast.metal_kernel(
    name="mlx_vlm_llguidance_mask",
    input_names=["logits", "mask"],
    output_names=["out"],
    source="""
        uint batch = thread_position_in_grid.y;
        uint token = thread_position_in_grid.x;
        uint word = token >> 5;
        uint bit = token & 31;
        bool allowed = word < mask_shape[1] &&
            ((as_type<uint>(mask[batch * mask_shape[1] + word]) >> bit) & 1u);
        uint offset = batch * logits_shape[1] + token;
        out[offset] = allowed ? logits[offset] : -metal::numeric_limits<T>::infinity();
    """,
)


def _apply_llguidance_mask(logits: mx.array, mask: mx.array) -> mx.array:
    if logits.ndim == 1:
        logits = logits[None, :]
    if mask.ndim == 1:
        mask = mask[None, :]
    return _LLGUIDANCE_MASK_KERNEL(
        inputs=[logits, mask],
        template=[("T", logits.dtype)],
        grid=(logits.shape[1], logits.shape[0], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
    )[0]


def _allocate_shared_bitmask(batch_size: int, vocab_size: int):
    """Allocate a CPU-writable mask backed by MLX shared memory."""
    mask = mx.full(
        (batch_size, (vocab_size + 31) // 32),
        -1,
        dtype=mx.int32,
    )
    mx.eval(mask)
    view = np.array(mask, copy=False)
    if not view.flags["C_CONTIGUOUS"] or not view.flags["WRITEABLE"]:
        raise RuntimeError("MLX bitmask must expose writable contiguous memory")
    return mask, view


class LLGuidanceLogitsProcessor:
    """MLX logits processor backed by llguidance.

    Accepts a single sequence or a batch. Expected shapes are input_ids as
    (seq_len,) or (batch, seq_len), and logits as (vocab,), (1, vocab), or
    (batch, vocab).
    """

    # Do not hold a latency-sensitive constrained token behind unrelated
    # prompt-prefill work after it has been decoded.
    requires_immediate_decode_yield = True

    def __init__(self, grammar: str, llg_tokenizer) -> None:
        self.grammar = grammar
        self.llg_tokenizer = llg_tokenizer
        self.is_first_token = True
        self._mask_cursor = -1

    def clone(self) -> "LLGuidanceLogitsProcessor":
        return LLGuidanceLogitsProcessor(self.grammar, self.llg_tokenizer)

    def reset(self):
        self.is_first_token = True
        self.ll_matchers = None
        self.bitmask = None
        self.bitmask_mx = None
        self._mask_buffers = None
        self._mask_cursor = -1

    def _setup(self, batch_size: int) -> None:
        from llguidance import LLMatcher

        self.ll_matchers = [
            LLMatcher(self.llg_tokenizer, self.grammar) for _ in range(batch_size)
        ]
        self._mask_buffers = [
            _allocate_shared_bitmask(batch_size, self.llg_tokenizer.vocab_size)
            for _ in range(2)
        ]
        self._mask_cursor = -1

    def _consume_tokens(self, last_tokens: list[int]) -> None:
        for i, last_token in enumerate(last_tokens):
            self.ll_matchers[i].consume_token(last_token)
            error = self.ll_matchers[i].get_error()
            if error:
                raise ValueError(f"LLGuidance matcher error: {error}")

    def _fill_next_token_mask(self) -> mx.array:
        import llguidance.numpy

        self._mask_cursor = (self._mask_cursor + 1) % len(self._mask_buffers)
        self.bitmask_mx, self.bitmask = self._mask_buffers[self._mask_cursor]
        for i, matcher in enumerate(self.ll_matchers):
            llguidance.numpy.fill_next_token_bitmask(matcher, self.bitmask, i)
        return self.bitmask_mx

    def _apply_bitmask(self, logits: mx.array) -> mx.array:
        return _apply_llguidance_mask(logits, self._fill_next_token_mask())

    def prepare_next_token_mask(self, last_token: int) -> mx.array:
        """Advance one matcher and return its packed MLX token mask."""
        if self.is_first_token:
            self._setup(1)
            self.is_first_token = False
        else:
            self._consume_tokens([last_token])
        return self._fill_next_token_mask()

    def process_last_token(self, last_token: int, logits: mx.array) -> mx.array:
        if logits.ndim == 1:
            return self.process_last_token(last_token, mx.expand_dims(logits, 0))[0]

        if self.is_first_token:
            self._setup(logits.shape[0])
            self.is_first_token = False
        else:
            self._consume_tokens([last_token])

        return self._apply_bitmask(logits)

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if input_ids.ndim == 1 and logits.ndim == 2 and logits.shape[0] == 1:
            input_ids = mx.expand_dims(input_ids, 0)
        elif logits.ndim == 1:
            return self(mx.expand_dims(input_ids, 0), mx.expand_dims(logits, 0))[0]

        batch_size = input_ids.shape[0]
        if self.is_first_token:
            self._setup(batch_size)
            self.is_first_token = False
        else:
            self._consume_tokens(input_ids[:, -1].tolist())

        return self._apply_bitmask(logits)


class ThinkingAwareLogitsProcessor:
    """Delay a logits processor until model thinking has ended.

    Structured grammars constrain the final answer, not the private thinking
    tokens. This wrapper keeps the wrapped processor inactive until it observes
    the configured thinking end token, then lets the wrapped processor constrain
    the next sampled token as the first token of the final answer.
    """

    def __init__(
        self,
        processor,
        tokenizer,
        thinking_start_token: str = "<think>",
        thinking_end_token: str = "</think>",
        enable_thinking: bool = False,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.thinking_start_token = thinking_start_token
        self.thinking_end_token = thinking_end_token
        self.enable_thinking = enable_thinking
        self._active = not enable_thinking
        self._active_context: list[int] = []

        self.thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        self.thinking_end_token_id = tokenizer.encode(
            thinking_end_token, add_special_tokens=False
        )[-1]

    def clone(self) -> "ThinkingAwareLogitsProcessor":
        processor = (
            self.processor.clone()
            if hasattr(self.processor, "clone")
            else self.processor
        )
        return ThinkingAwareLogitsProcessor(
            processor=processor,
            tokenizer=self.tokenizer,
            thinking_start_token=self.thinking_start_token,
            thinking_end_token=self.thinking_end_token,
            enable_thinking=self.enable_thinking,
        )

    def reset(self):
        self._active = not self.enable_thinking
        self._active_context = []
        if hasattr(self.processor, "reset"):
            self.processor.reset()

    def _activate_if_thinking_done(self, token_id: int) -> bool:
        if self._active:
            return False
        if token_id == self.thinking_end_token_id:
            self._active = True
            self._active_context = []
            return True
        return False

    def _delegate_process_last_token(
        self, last_token: int, logits: mx.array
    ) -> mx.array:
        if hasattr(self.processor, "process_last_token"):
            return self.processor.process_last_token(last_token, logits)

        self._active_context.append(last_token)
        return self.processor(mx.array(self._active_context), logits)

    def process_last_token(self, last_token: int, logits: mx.array) -> mx.array:
        if not self._active:
            activated = self._activate_if_thinking_done(last_token)
            if not activated:
                return logits
            return self._delegate_process_last_token(last_token, logits)

        return self._delegate_process_last_token(last_token, logits)

    def _last_token_from_input_ids(self, input_ids: mx.array) -> int | None:
        if input_ids.ndim == 0:
            return int(input_ids.item())
        if input_ids.ndim == 1:
            if input_ids.shape[0] == 0:
                return None
            return int(input_ids[-1].item())
        if input_ids.shape[-1] == 0:
            return None
        return int(input_ids.reshape(-1, input_ids.shape[-1])[0, -1].item())

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        if self._active:
            return self.processor(input_ids, logits)

        token_id = self._last_token_from_input_ids(input_ids)
        if token_id is not None and self._activate_if_thinking_done(token_id):
            return self.processor(input_ids, logits)

        return logits


def _serialize_schema(schema: str | dict[str, Any]) -> str:
    if isinstance(schema, str):
        return schema
    return json.dumps(schema)


# Building an llguidance tokenizer walks the entire vocab (~1.5s for a 150k
# token model), so we keep the result around for the lifetime of the process.
_llg_tokenizer_cache = {}


def _stop_token_ids(tokenizer) -> list[int]:
    """All token ids that end generation for this tokenizer.

    Models such as GLM-4.7 and Flint end a tool-call turn on a token other than
    ``eos_token_id`` (e.g. ``<|observation|>``). The grammar must allow every
    stop token, or a complete output cannot end and generation runs on.
    """
    stop_ids = []
    criteria = getattr(tokenizer, "stopping_criteria", None)
    for source in (
        getattr(criteria, "eos_token_ids", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if isinstance(source, int):
            source = [source]
        for token_id in source or ():
            if isinstance(token_id, int) and token_id not in stop_ids:
                stop_ids.append(token_id)
    return stop_ids


def _get_llg_tokenizer(tokenizer):
    import llguidance.hf

    stop_ids = _stop_token_ids(tokenizer)
    key = (id(tokenizer), tuple(stop_ids))
    llg_tokenizer = _llg_tokenizer_cache.get(key)
    if llg_tokenizer is None:
        if len(stop_ids) > 1:
            llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer, eos_token=stop_ids)
        else:
            llg_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        _llg_tokenizer_cache[key] = llg_tokenizer
    return llg_tokenizer


def build_json_schema_logits_processor(tokenizer, schema: str | dict[str, Any]):
    try:
        import llguidance as llg
        import llguidance.hf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "llguidance is required for structured response_format generation. "
            "Install mlx-vlm with llguidance available."
        ) from exc

    llg_tokenizer = _get_llg_tokenizer(tokenizer)

    schema_text = _serialize_schema(schema)
    if hasattr(llg, "JsonCompiler"):
        grammar = llg.JsonCompiler(
            separators=(", ", ": "),
            whitespace_pattern="",
        ).compile(schema_text)
    else:
        grammar = llg.grammar_from("json_schema", schema_text)
    return LLGuidanceLogitsProcessor(grammar, llg_tokenizer)


# Tool-call formats with a grammar builder. GLM-4.7 and Flint share the
# `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` format.
_TOOL_CALL_GRAMMAR_PARSERS = ("glm47",)

_JSON_VALUE_PATTERNS = {
    "integer": r"/-?(0|[1-9][0-9]*)/",
    "number": r"/-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/",
    "boolean": '"true" | "false"',
}


def _marker(tokenizer, text: str) -> str:
    """Reference a marker by token id when it is one token, else as text."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 1:
        return f"<[{token_ids[0]}]>"
    return json.dumps(text)


def _tool_function(tool: dict[str, Any]) -> dict[str, Any]:
    return tool.get("function", tool) if isinstance(tool, dict) else {}


def build_glm47_tool_call_grammar(tokenizer, tools: list[dict[str, Any]]) -> str:
    """Build an llguidance Lark grammar for GLM-4.7 / Flint tool calls.

    Text and `<think>` blocks stay free. Inside `<tool_call>`, the function
    name must be one of `tools`, each `<arg_key>` must be a parameter of that
    tool, and integer/number/boolean values must match their JSON type.
    """
    m = {
        name: _marker(tokenizer, text)
        for name, text in (
            ("think", "<think>"),
            ("think_end", "</think>"),
            ("call", "<tool_call>"),
            ("call_end", "</tool_call>"),
            ("key", "<arg_key>"),
            ("key_end", "</arg_key>"),
            ("value", "<arg_value>"),
            ("value_end", "</arg_value>"),
        )
    }

    rules = []
    functions = []
    for i, tool in enumerate(tools):
        fn = _tool_function(tool)
        name = fn.get("name")
        if not name:
            continue
        parameters = fn.get("parameters") or {}
        properties = parameters.get("properties") or {}
        if not isinstance(properties, dict) or not all(properties):
            raise ValueError(f"tool {name!r} has invalid parameter names")
        required = set(parameters.get("required") or ())
        # Arguments follow the schema order: required ones must appear, optional
        # ones may be skipped, and none can repeat. Order does not change a call.
        args = []
        for j, (key, schema) in enumerate(properties.items()):
            json_type = schema.get("type") if isinstance(schema, dict) else None
            pattern = _JSON_VALUE_PATTERNS.get(json_type)
            value = f"({pattern})" if pattern else "TEXT?"
            rules.append(
                f'arg_{i}_{j}: {m["key"]} {json.dumps(key)} {m["key_end"]} '
                f'{m["value"]} {value} {m["value_end"]}'
            )
            args.append(f"arg_{i}_{j}" if key in required else f"arg_{i}_{j}?")
        functions.append(" ".join([json.dumps(name), *args]))

    if not functions:
        raise ValueError("tools must contain at least one named function")

    return (
        "\n".join(
            [
                "%llguidance {}",
                "start: TEXT? (block TEXT?)*",
                "block: think | call",
                f'think: {m["think"]} TEXT? {m["think_end"]}',
                f'call: {m["call"]} function {m["call_end"]}',
                "function: " + "\n    | ".join(functions),
                *rules,
                r"TEXT: /(.|\n)+/",
            ]
        )
        + "\n"
    )


def build_tool_call_logits_processor(tokenizer, tool_parser_type, tools):
    """Return a processor that keeps tool calls inside the request's schemas.

    Returns None when the tool format has no grammar builder or no tools are
    given, so callers can fall back to unconstrained generation.
    """
    if tool_parser_type not in _TOOL_CALL_GRAMMAR_PARSERS or not tools:
        return None
    try:
        import llguidance as llg
    except ImportError:
        return None

    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        grammar = llg.grammar_from(
            "lark", build_glm47_tool_call_grammar(tokenizer, tools)
        )
        llg_tokenizer = _get_llg_tokenizer(tokenizer)
        error = llg.LLMatcher.validate_grammar(grammar, llg_tokenizer)
    except Exception as exc:  # client-supplied schemas can be malformed
        error = str(exc)
    if error:
        logger.warning("Tool-call grammar disabled for this request: %s", error)
        return None
    return LLGuidanceLogitsProcessor(grammar, llg_tokenizer)
