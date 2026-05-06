"""
Tool parser utilities for mlx-vlm.

Re-exports mlx_lm's _infer_tool_parser with additional model support,
and loads parsers from both mlx_lm.tool_parsers and mlx_vlm.tool_parsers.
"""

import importlib

from mlx_lm.tokenizer_utils import _infer_tool_parser as _mlx_lm_infer_tool_parser

# Additional patterns not covered by mlx_lm
_EXTRA_PATTERNS = [
    ("<|tool_call>", "gemma4"),
]


def _infer_tool_parser(chat_template):
    """Infer tool parser type, checking mlx_lm patterns first then extras."""
    result = _mlx_lm_infer_tool_parser(chat_template)
    if result is not None:
        return result

    if not isinstance(chat_template, str):
        return None

    for marker, parser_type in _EXTRA_PATTERNS:
        if marker in chat_template:
            return parser_type

    return None


def _infer_tool_parser_from_processor(processor):
    """Infer tool parser type from processor's chat template."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return _infer_tool_parser(tokenizer.chat_template)

    return None


def load_tool_module(tool_parser_type):
    """Load a tool parser module from mlx_vlm.tool_parsers or mlx_lm.tool_parsers."""
    if importlib.util.find_spec(f"mlx_vlm.tool_parsers.{tool_parser_type}"):
        return importlib.import_module(f"mlx_vlm.tool_parsers.{tool_parser_type}")
    return importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
