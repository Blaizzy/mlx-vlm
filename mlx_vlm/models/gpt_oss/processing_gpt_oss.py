"""gpt-oss harmony response format.

gpt-oss replies in harmony channels — ``analysis``/``commentary`` (reasoning)
and ``final`` (the answer) — wrapped in ``<|channel|>NAME<|message|>...<|end|>``
control tokens. The HF tokenizer already exposes the parser machinery
(``parse_response`` / ``get_response_parser``) but ships no template, so
``response_template`` is ``None`` and the raw channels leak into message content.

Registering a processor for the ``gpt_oss`` model type fills that slot at load
and collapses the repeated reasoning channels, so the server's existing
response-template path handles gpt-oss with no harmony-specific server code.
"""

from ..base import install_auto_processor_patch

HARMONY_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "fields": {
        "reasoning_content": {
            "open_pattern": r"<\|channel\|>(?:analysis|commentary)<\|message\|>",
            "close": ["<|end|>"],
            "content": "text",
            "repeats": True,
        },
        "content": {
            "open_pattern": r"<\|channel\|>final<\|message\|>",
            "close": ["<|return|>", "<|end|>"],
            "content": "text",
        },
    },
    "start_anchor": "<|start|>assistant",
}


def _join_channels(value):
    """Collapse a repeated template field to a single string.

    ``reasoning_content`` matches both the analysis and commentary channels, so
    it carries ``repeats`` and parses to a list even when only one channel is
    present. Dropping ``repeats`` is not an alternative: the parser then keeps
    the last match, so a reply with both channels loses the analysis entirely.
    """
    if isinstance(value, (list, tuple)):
        parts = [
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        ]
        return "\n".join(parts) if parts else None
    return value


def _wrap_parse_response(tokenizer):
    """Join repeated fields so callers always receive strings."""
    original = getattr(tokenizer, "parse_response", None)
    if original is None or getattr(original, "_harmony_joined", False):
        return

    def parse_response(*args, **kwargs):
        parsed = original(*args, **kwargs)
        if isinstance(parsed, dict):
            return {key: _join_channels(val) for key, val in parsed.items()}
        return parsed

    parse_response._harmony_joined = True
    try:
        tokenizer.parse_response = parse_response
    except (AttributeError, TypeError):
        pass


def _attach_harmony_template(processor):
    """Fill an empty ``response_template`` slot on a gpt-oss tokenizer."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer is not None and getattr(tokenizer, "response_template", None) is None:
        try:
            tokenizer.response_template = HARMONY_RESPONSE_TEMPLATE
        except (AttributeError, TypeError):
            pass
    if tokenizer is not None:
        _wrap_parse_response(tokenizer)
    return processor


class GptOssProcessor:
    """Load gpt-oss normally, then attach the harmony response template.

    gpt-oss is text-only, so its "processor" is the tokenizer itself; this
    returns exactly what transformers would and only fills the template slot.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoProcessor

        processor = _previous_auto_processor_from_pretrained.__func__(
            AutoProcessor, pretrained_model_name_or_path, **kwargs
        )
        return _attach_harmony_template(processor)


_previous_auto_processor_from_pretrained = install_auto_processor_patch(
    "gpt_oss", GptOssProcessor
)
