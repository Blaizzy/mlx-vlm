"""Tests for the Gemma 4 tool-call parser."""

import json
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
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Nuremberg"})

    def test_snake_case_with_numeric_arg(self):
        text = _wrap("call:set_volume{level:42}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "set_volume")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"level": 42})

    # ── hyphenated names (the bug this PR fixes) ──────────────────────────

    def test_hyphenated_name(self):
        text = _wrap(
            f"call:lobe-web-browsing____search{{search_query:{_str('Lustgarten Berlin')}}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "lobe-web-browsing____search")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"search_query": "Lustgarten Berlin"})

    def test_single_hyphen_name(self):
        text = _wrap(f"call:get-weather{{city:{_str('Tokyo')}}}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "get-weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Tokyo"})

    def test_hyphenated_name_with_nested_args(self):
        text = _wrap(
            f"call:edit-file{{path:{_str('test.txt')},edits:[{{newText:{_str('orange')},oldText:{_str('apple')}}}]}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "edit-file")
        args = json.loads(result["arguments"])
        self.assertEqual(args["path"], "test.txt")
        self.assertEqual(
            args["edits"],
            [{"newText": "orange", "oldText": "apple"}],
        )

    # ── arguments type ────────────────────────────────────────────────────

    def test_arguments_is_json_string(self):
        """Arguments should be a JSON string per OpenAI spec."""
        text = _wrap(f"call:get_weather{{city:{_str('Berlin')}}}")
        result = parse_tool_call(text)
        self.assertIsInstance(result["arguments"], str)
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed, {"city": "Berlin"})

    # ── JSON-formatted keys (models may output standard JSON) ─────────────

    def test_json_formatted_keys_simple(self):
        """Models may output standard JSON with quoted keys: {"key": value}."""
        text = _wrap('call:get_weather{"city": "Berlin"}')
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "get_weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Berlin"})

    def test_json_formatted_keys_multiple(self):
        """Multiple JSON-formatted keys."""
        text = _wrap('call:memory_add{"action": "add", "content": "test", "target": "memory"}')
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "memory_add")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"action": "add", "content": "test", "target": "memory"})

    def test_json_formatted_keys_with_nested_values(self):
        """JSON keys with nested object/array values."""
        text = _wrap('call:process_data{"items": [1, 2, 3], "config": {"enabled": true}}')
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "process_data")
        args = json.loads(result["arguments"])
        self.assertEqual(args["items"], [1, 2, 3])
        self.assertEqual(args["config"], {"enabled": True})

    def test_json_formatted_keys_with_unicode_escape(self):
        """JSON keys with unicode escape sequences."""
        text = _wrap(r'call:memory_add{"action": "add", "content": "2026-04-25 > \u672c\u5730\u73af\u5883"}')
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "memory_add")
        args = json.loads(result["arguments"])
        self.assertEqual(args["action"], "add")
        self.assertIn("本地环境", args["content"])

    def test_mixed_gemma_and_json_formats(self):
        """Mix of Gemma 4 escape strings and JSON-formatted keys."""
        text = _wrap(f'call:search{{"query":{_str("Berlin weather")}}}')
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "search")
        args = json.loads(result["arguments"])
        self.assertEqual(args["query"], "Berlin weather")

    # ── error path ────────────────────────────────────────────────────────

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("just a normal model response, no tool call here")


if __name__ == "__main__":
    unittest.main()
