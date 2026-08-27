"""Tests for the Gemma 4 tool-call parser."""

import json
import unittest

import mlx_vlm.tool_parsers.gemma4 as gemma4
from mlx_vlm.server.responses_state import process_tool_calls
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

    def test_colon_namespaced_name(self):
        text = _wrap(
            f"call:google-workspace:google-workspace{{action:{_str('list_events')}}}"
        )
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "google-workspace:google-workspace")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"action": "list_events"})

    def test_dotted_namespaced_name(self):
        text = _wrap(f"call:mcp.calendar:list_events{{limit:5}}")
        result = parse_tool_call(text)
        self.assertEqual(result["name"], "mcp.calendar:list_events")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"limit": 5})

    # ── arguments type ────────────────────────────────────────────────────

    def test_arguments_is_json_string(self):
        """Arguments should be a JSON string per OpenAI spec."""
        text = _wrap(f"call:get_weather{{city:{_str('Berlin')}}}")
        result = parse_tool_call(text)
        self.assertIsInstance(result["arguments"], str)
        parsed = json.loads(result["arguments"])
        self.assertEqual(parsed, {"city": "Berlin"})

    # ── DiffusionGemma payload tolerance ──────────────────────────────────

    def test_bare_call_syntax(self):
        result = parse_tool_call("call:get_weather{city:Austin}")
        self.assertEqual(result["name"], "get_weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Austin"})

    def test_bare_function_syntax(self):
        result = parse_tool_call("get_weather{city:Austin}")
        self.assertEqual(result["name"], "get_weather")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"city": "Austin"})

    def test_bare_namespaced_function_syntax(self):
        result = parse_tool_call("mcp.calendar:list_events{limit:5}")
        self.assertEqual(result["name"], "mcp.calendar:list_events")
        args = json.loads(result["arguments"])
        self.assertEqual(args, {"limit": 5})

    def test_process_tool_calls_handles_raw_diffusion_gemma_output(self):
        result = process_tool_calls(
            _wrap(f"call:get_weather{{city:{_str('Austin')}}}"),
            gemma4,
            tools=None,
        )

        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["calls"][0]["function"]["name"], "get_weather")
        args = json.loads(result["calls"][0]["function"]["arguments"])
        self.assertEqual(args, {"city": "Austin"})
        self.assertEqual(result["remaining_text"], "")

    def test_process_tool_calls_ignores_non_call_prose(self):
        result = process_tool_calls("Like call: prince", gemma4, tools=None)

        self.assertEqual(result["calls"], [])
        self.assertEqual(result["remaining_text"], "Like call: prince")

    # ── error path ────────────────────────────────────────────────────────

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("just a normal model response, no tool call here")


if __name__ == "__main__":
    unittest.main()
