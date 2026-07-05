"""
Tool parser utilities for mlx-vlm.

Infers the tool parser type from chat template markers and loads the
matching parser module from ``mlx_vlm.tool_parsers``.
"""

import importlib


_TEMPLATE_MARKERS = [
    (("<|tool_call>",), "gemma4"),
    (("<|START_ACTION|>",), "cohere2_moe"),
    (("]<]minimax[>[<tool_call>",), "minimax_m3"),
    (("<mm:think>",), "minimax_m3"),
    (("<minimax:tool_call>",), "minimax_m2"),
    (("<start_function_call>",), "function_gemma"),
    (("<longcat_tool_call>",), "longcat"),
    (("<arg_key>",), "glm47"),
    (("<|tool_list_start|>",), "pythonic"),
    (("<tool_call>\\n<function=",), "qwen3_coder"),
    (("<tool_call>\n<function=",), "qwen3_coder"),
    (("<|tool_calls_section_begin|>",), "kimi_k2"),
    (("[TOOL_CALLS]",), "mistral"),
    (("<tool_call>", "tool_call.name"), "json_tools"),
]


def _infer_tool_parser(chat_template):
    """Infer the tool parser type from the chat template."""
    if not isinstance(chat_template, str):
        return None

    for markers, parser_type in _TEMPLATE_MARKERS:
        if all(marker in chat_template for marker in markers):
            return parser_type

    return None


def _infer_tool_parser_from_processor(processor):
    """Infer tool parser type from processor's chat template."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return _infer_tool_parser(tokenizer.chat_template)

    return None


def load_tool_module(tool_parser_type):
    """Load a tool parser module from mlx_vlm.tool_parsers."""
    module_name = f"mlx_vlm.tool_parsers.{tool_parser_type}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            raise ValueError(f"Unknown tool parser type: {tool_parser_type!r}") from e
        raise
