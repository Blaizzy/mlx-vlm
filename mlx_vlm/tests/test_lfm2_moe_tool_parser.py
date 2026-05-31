"""Tests for the LFM2-MoE tool-call parser."""

import json
import unittest

from mlx_vlm.server.responses_state import process_tool_calls
from mlx_vlm.tool_parsers import _infer_tool_parser, lfm2_moe
from mlx_vlm.tool_parsers.lfm2_moe import parse_tool_call

TC_START = "<|tool_call_start|>"
TC_END = "<|tool_call_end|>"


def _wrap(call_body: str) -> str:
    return f"{TC_START}{call_body}{TC_END}"


class TestLFM2MoeToolParser(unittest.TestCase):
    def test_infers_lfm2_moe_from_tool_call_markers(self):
        template = "{{ '<|tool_call_start|>[' + call + ']<|tool_call_end|>' }}"
        self.assertEqual(_infer_tool_parser(template), "lfm2_moe")

    def test_simple_pythonic_call(self):
        result = parse_tool_call(
            _wrap(
                "[search_files(pattern='.', target='files', "
                "output_mode='files_only')]"
            )
        )
        self.assertEqual(result["name"], "search_files")
        self.assertEqual(
            json.loads(result["arguments"]),
            {"pattern": ".", "target": "files", "output_mode": "files_only"},
        )

    def test_nested_json_object_argument(self):
        result = parse_tool_call(
            '[execute_code(language=\'python\', payload={"code":"print(1)","safe":true})]'
        )
        self.assertEqual(result["name"], "execute_code")
        self.assertEqual(
            json.loads(result["arguments"]),
            {"language": "python", "payload": {"code": "print(1)", "safe": True}},
        )

    def test_hyphenated_tool_name(self):
        result = parse_tool_call("[browser-cdp(action='snapshot')]")
        self.assertEqual(result["name"], "browser-cdp")
        self.assertEqual(json.loads(result["arguments"]), {"action": "snapshot"})

    def test_multiple_calls_are_preserved_by_server_processing(self):
        text = _wrap("[first_tool(x=1), second_tool(query='docs, notes')]")
        result = process_tool_calls(text, lfm2_moe, tools=[])
        self.assertEqual(result["remaining_text"], "")
        self.assertEqual(
            [c["function"]["name"] for c in result["calls"]],
            ["first_tool", "second_tool"],
        )
        self.assertEqual(
            json.loads(result["calls"][0]["function"]["arguments"]), {"x": 1}
        )
        self.assertEqual(
            json.loads(result["calls"][1]["function"]["arguments"]),
            {"query": "docs, notes"},
        )

    def test_no_call_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("just a normal model response, no tool call here")


if __name__ == "__main__":
    unittest.main()
