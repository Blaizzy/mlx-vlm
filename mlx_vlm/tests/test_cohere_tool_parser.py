import json

from mlx_vlm.tool_parsers.cohere2_moe import parse_tool_call


def test_cohere_action_array_parses_to_openai_tool_calls():
    result = parse_tool_call(
        """
        [
          {"tool_call_id": "1", "tool_name": "grep", "parameters": {"pattern": "<\\|channel>"}},
          {"tool_call_id_id": "2", "tool_name": "read", "parameters": {"path": "file.py"}}
        ]
        """
    )

    assert [call["name"] for call in result] == ["grep", "read"]
    assert json.loads(result[0]["arguments"]) == {"pattern": "<|channel>"}
    assert json.loads(result[1]["arguments"]) == {"path": "file.py"}


def test_cohere_single_action_object_parses_to_openai_tool_call():
    result = parse_tool_call(
        '{"tool_call_id": "1", "tool_name": "grep", "parameters": {"pattern": "foo"}}'
    )

    assert result["name"] == "grep"
    assert json.loads(result["arguments"]) == {"pattern": "foo"}
