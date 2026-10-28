"""Native LFM3 thinking/content parsing and XML function calls."""

from dataclasses import dataclass
from typing import Optional

# LFM3 uses the same XML function/parameter syntax and schema conversions.
from . import qwen3_coder

parse_tool_call = qwen3_coder.parse_tool_call
_convert_param_value = qwen3_coder._convert_param_value
_get_arguments_config = qwen3_coder._get_arguments_config
_string_types = qwen3_coder._string_types

tool_call_start = "<tool_call>"
tool_call_end = "</tool_call>"


@dataclass
class ResponseDelta:
    reasoning: Optional[str] = None
    content: Optional[str] = None
    thinking_closed: bool = False


class ResponseParser:
    """Split LFM3 output without depending on a tokenizer response template.

    The prompt may already contain the opening <think> or an empty, closed
    thinking block. Only the current assistant prefix establishes that state.
    Tool spans are hidden from text; the tool pipeline parses the raw output
    independently, after removing reasoning.
    """

    def __init__(self, *, prefix="", starts_in_thinking=False):
        _, anchor, suffix = prefix.rpartition("<|im_start|>assistant\n")
        if anchor:
            starts_in_thinking = suffix.rfind("<think>") > suffix.rfind("</think>")
        self.state = "reasoning" if starts_in_thinking else "content"
        self.thinking_done = bool(anchor and "</think>" in suffix)
        self.buffer = ""
        self.trim_newlines = False
        self.trim_whitespace = False
        self.done = False

    def feed(self, text, last=False):
        if self.done:
            return ResponseDelta()
        self.buffer += text or ""
        reasoning, content = [], []
        thinking_closed = False

        def emit(value):
            if self.trim_whitespace:
                value = value.lstrip()
                self.trim_whitespace = not bool(value)
            elif self.trim_newlines:
                value = value.lstrip("\n")
                self.trim_newlines = not bool(value)
            if value and self.state != "tool":
                (reasoning if self.state == "reasoning" else content).append(value)

        while self.buffer:
            if self.state == "tool":
                markers = (tool_call_end,)
            elif self.state == "reasoning":
                markers = ("</think>", "<|im_end|>")
            else:
                markers = (tool_call_start, "<|im_end|>")
                if not self.thinking_done:
                    markers += ("<think>",)
            matches = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if not matches:
                hold = 0
                if not last:
                    for marker in markers:
                        for size in range(
                            min(len(marker) - 1, len(self.buffer)), 0, -1
                        ):
                            if self.buffer.endswith(marker[:size]):
                                hold = max(hold, size)
                                break
                end = len(self.buffer) - hold
                emit(self.buffer[:end])
                self.buffer = self.buffer[end:]
                break

            index, marker = min(matches)
            emit(self.buffer[:index])
            self.buffer = self.buffer[index + len(marker) :]
            if marker == "<|im_end|>":
                self.done = True
                self.buffer = ""
            elif marker == "<think>":
                self.state = "reasoning"
                self.trim_newlines = True
            elif marker == "</think>":
                self.state = "content"
                self.thinking_done = True
                thinking_closed = True
                self.trim_newlines = False
                self.trim_whitespace = True
            elif marker == tool_call_start:
                self.state = "tool"
            else:
                self.state = "content"
                self.trim_newlines = False
                self.trim_whitespace = True

        if last:
            self.done = True
        return ResponseDelta(
            reasoning="".join(reasoning) or None,
            content="".join(content) or None,
            thinking_closed=thinking_closed,
        )
