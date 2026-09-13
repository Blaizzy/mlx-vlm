"""Tool-calling subsystem for mlx-vlm.

A single, server-agnostic home for the tool-calling pipeline: parser selection
(:mod:`~mlx_vlm.tools.registry`), extraction into the neutral
:class:`~mlx_vlm.tools.types.ParseResult` (:mod:`~mlx_vlm.tools.extractor`), and
``tool_choice`` policy (:mod:`~mlx_vlm.tools.policy`). The parsers themselves
live in :mod:`mlx_vlm.tools.parsers`; the registry maps a chat template to one of
them. The OpenAI- and Anthropic-compatible servers depend on this package;
nothing here depends on a server.
"""

from .base import ToolParser
from .extractor import process_tool_calls
from .policy import _prepare_chat_tool_choice
from .registry import (
    SPECS,
    ParserSpec,
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
)
from .types import ParseResult

__all__ = [
    "ParseResult",
    "ParserSpec",
    "SPECS",
    "ToolParser",
    "process_tool_calls",
    "load_tool_module",
    "_infer_tool_parser",
    "_infer_tool_parser_from_processor",
    "_prepare_chat_tool_choice",
]
