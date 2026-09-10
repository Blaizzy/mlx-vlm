"""Map a model's chat template to the tool parser that owns it.

Each supported format is one :class:`ParserSpec` in :data:`SPECS`: the parser
module name (in :mod:`mlx_vlm.tools.parsers`), the chat-template markers that
identify the format, and a priority. Selection returns the highest-priority
spec whose markers all appear in the template, so a format-specific parser
outranks the generic ``json_tools`` fallback that shares its extraction
markers. ``priority`` — not list position — carries that relationship, so
specs can be reordered or inserted without changing routing.

Adding a model is one :class:`ParserSpec` here plus its parser module; an
explicit ``override`` name bypasses inference entirely.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("mlx_vlm.tools")


@dataclass(frozen=True)
class ParserSpec:
    """Routing entry for one tool-call format.

    ``template_markers`` is an OR of AND-groups: the spec matches a chat
    template when every marker in any single group is present.
    """

    name: str
    template_markers: Tuple[Tuple[str, ...], ...]
    priority: int = 100

    def matches(self, chat_template: str) -> bool:
        return any(
            all(marker in chat_template for marker in group)
            for group in self.template_markers
        )


# Kept in historical marker-table order; `priority` — not position — decides
# overlaps. json_tools is the generic `<tool_call>` JSON fallback at priority 0,
# so any format-specific parser that shares those markers wins.
SPECS: Tuple[ParserSpec, ...] = (
    ParserSpec("atem", (("<atem:function_calls>", "<atem:invoke"),)),
    ParserSpec("gemma4", (("<|tool_call>",),)),
    ParserSpec("cohere2_moe", (("<|START_ACTION|>",),)),
    ParserSpec("minimax_m3", (("]<]minimax[>[<tool_call>",), ("<mm:think>",))),
    ParserSpec("minimax_m2", (("<minimax:tool_call>",),)),
    ParserSpec("minicpm5", (("<function name=", "<param name="),)),
    ParserSpec("longcat", (("<longcat_tool_call>",),)),
    ParserSpec("glm47", (("<arg_key>",),)),
    ParserSpec(
        "pythonic",
        (("<|tool_call_start|>", "<|tool_call_end|>"), ("<|tool_list_start|>",)),
    ),
    ParserSpec(
        "qwen3_coder",
        (("<tool_call>\\n<function=",), ("<tool_call>\n<function=",)),
    ),
    ParserSpec("kimi_k2", (("<|tool_calls_section_begin|>",),)),
    ParserSpec("mistral", (("[TOOL_CALLS]",),)),
    ParserSpec("json_tools", (("<tool_call>", "tool_call.name"),), priority=0),
)

_BY_NAME = {spec.name: spec for spec in SPECS}


def _template_text(chat_template) -> Optional[str]:
    """Resolve a tokenizer ``chat_template`` to the string used for tool-call
    detection.

    A tokenizer may carry several named templates -- transformers exposes them
    as a dict ``{name: template}`` (or, from config, a list of
    ``{"name", "template"}``). The ``tool_use`` variant is the one rendered when
    tools are passed, so it is authoritative for detecting the tool-call format;
    fall back to ``default`` and then any variant.
    """
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, list):
        chat_template = {
            entry.get("name"): entry.get("template")
            for entry in chat_template
            if isinstance(entry, dict)
        }
    if isinstance(chat_template, dict):
        for key in ("tool_use", "default"):
            if isinstance(chat_template.get(key), str):
                return chat_template[key]
        return next((v for v in chat_template.values() if isinstance(v, str)), None)
    return None


def _infer_tool_parser(chat_template, override: Optional[str] = None) -> Optional[str]:
    """Return the tool parser name for a chat template, or ``None``.

    ``chat_template`` may be a plain string or a multi-variant dict/list (see
    :func:`_template_text`). When ``override`` is given it is validated against
    :data:`SPECS` and returned as-is, skipping template inference.
    """
    if override is not None:
        if override not in _BY_NAME:
            raise ValueError(f"Unknown tool parser override: {override!r}")
        return override

    text = _template_text(chat_template)
    if text is None:
        return None

    matches = [spec for spec in SPECS if spec.matches(text)]
    if not matches:
        logger.debug("No tool parser matched the chat template.")
        return None
    return max(matches, key=lambda spec: spec.priority).name


def _infer_tool_parser_from_processor(
    processor, override: Optional[str] = None
) -> Optional[str]:
    """Infer the tool parser name from a processor's chat template."""
    if override is not None:
        return _infer_tool_parser(None, override=override)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if getattr(tokenizer, "chat_template", None):
        return _infer_tool_parser(tokenizer.chat_template)
    return None


def load_tool_module(tool_parser_type):
    """Load a tool parser module from :mod:`mlx_vlm.tools.parsers`."""
    module_name = f"mlx_vlm.tools.parsers.{tool_parser_type}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            raise ValueError(f"Unknown tool parser type: {tool_parser_type!r}") from e
        raise
