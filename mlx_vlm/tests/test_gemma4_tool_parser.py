"""Tests for the Gemma 4 tool-call parser."""

import unittest

from mlx_vlm.tool_parsers.gemma4 import parse_tool_call

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

    def test_snake_case_simple_string(self):
        text = _wrap(f"call:get_weather{{city:{_str('Nuremberg')}}}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "get_weather")
        self.assertEqual(result["arguments"], {"city": "Nuremberg"})

    def test_snake_case_with_numeric_arg(self):
        text = _wrap("call:set_volume{level:42}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "set_volume")
        self.assertEqual(result["arguments"], {"level": 42})

    # ── hyphenated names (the bug this PR fixes) ──────────────────────────

    def test_hyphenated_name(self):
        text = _wrap(
            f"call:lobe-web-browsing____search{{search_query:{_str('Lustgarten Berlin')}}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "lobe-web-browsing____search")
        self.assertEqual(result["arguments"], {"search_query": "Lustgarten Berlin"})

    def test_single_hyphen_name(self):
        text = _wrap(f"call:get-weather{{city:{_str('Tokyo')}}}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "get-weather")
        self.assertEqual(result["arguments"], {"city": "Tokyo"})

    def test_hyphenated_name_with_nested_args(self):
        text = _wrap(
            f"call:edit-file{{path:{_str('test.txt')},edits:[{{newText:{_str('orange')},oldText:{_str('apple')}}}]}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "edit-file")
        self.assertEqual(result["arguments"]["path"], "test.txt")
        self.assertEqual(
            result["arguments"]["edits"],
            [{"newText": "orange", "oldText": "apple"}],
        )

    # ── error path ────────────────────────────────────────────────────────

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("just a normal model response, no tool call here")


if __name__ == "__main__":
    unittest.main()
