"""Extract structured tool calls from a model's text output.

Format-agnostic: given the parser module selected for the model (see
:mod:`mlx_vlm.tools.registry`), it slices the text between the parser's markers,
delegates per-format decoding to ``parse_tool_call``, and emits OpenAI-shaped
calls.
"""

import json
import logging
import re
import uuid

from .base import ToolParser
from .types import ParseResult

logger = logging.getLogger("mlx_vlm.server")


def process_tool_calls(
    model_output: str, tool_module: ToolParser, tools
) -> ParseResult:
    """Parse tool calls from model output using the given tool parser module."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for match in matches:
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    parsed = tool_module.parse_tool_call(call, tools)
                    parsed_calls = parsed if isinstance(parsed, list) else [parsed]
                    for tool_call in parsed_calls:
                        args = tool_call["arguments"]
                        called_tools.append(
                            {
                                "type": "function",
                                "index": len(called_tools),
                                "id": str(uuid.uuid4()),
                                "function": {
                                    "name": tool_call["name"].strip(),
                                    "arguments": (
                                        args
                                        if isinstance(args, str)
                                        else json.dumps(args, ensure_ascii=False)
                                    ),
                                },
                            },
                        )
                except Exception as exc:
                    logger.warning("Invalid tool call %r: %s", call, exc)
    return ParseResult(calls=called_tools, remaining_text=remaining)
