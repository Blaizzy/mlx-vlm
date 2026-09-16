"""Tests for the Gemma 4 tool-call parser."""

import json
import unittest

import mlx_vlm.tools.parsers.gemma4 as gemma4
from mlx_vlm.tools import process_tool_calls
from mlx_vlm.tools.parsers.gemma4 import parse_tool_call

# Wire-format helpers
TC_START = "<|tool_call>"
TC_END = "<tool_call|>"
ESC = '<|"|>'


def _wrap(call_body: str) -> str:
    """Wrap a ``call:name{...}`` body in the Gemma 4 tool-call markers."""
    return f"{TC_START}{call_body}{TC_END}"


def _str(value: str) -> str:
    """Render a string value with the Gemma 4 escape delimiter."""
    return f"{ESC}{value}{ESC}"


class TestGemma4ToolParser(unittest.TestCase):
    # ── snake_case (regression: must keep working) ────────────────────────

    # ── hyphenated names (the bug this PR fixes) ──────────────────────────

    def test_hyphenated_name_with_nested_args(self):
        text = _wrap(
            f"call:edit-file{{path:{_str('test.txt')},edits:[{{newText:{_str('orange')},oldText:{_str('apple')}}}]}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "edit-file")
        args = json.loads(result["arguments"])
        self.assertEqual(args["path"], "test.txt")
        self.assertEqual(args["edits"], [{"newText": "orange", "oldText": "apple"}])

    # ── arguments type ────────────────────────────────────────────────────

    # ── DiffusionGemma payload tolerance ──────────────────────────────────

    def test_bare_function_syntax(self):
        result = parse_tool_call("get_weather{city:Austin}")
        self.assertEqual(result["name"], "get_weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Austin"})

    def test_process_tool_calls_ignores_non_call_prose(self):
        result = process_tool_calls("Like call: prince", gemma4, tools=None)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.remaining_text, "Like call: prince")

    # ── error path ────────────────────────────────────────────────────────

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("just a normal model response, no tool call here")


if __name__ == "__main__":
    unittest.main()
