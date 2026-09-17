"""Shared contracts for every discovered tool parser and format-specific edge cases."""

import json
import pkgutil
from types import SimpleNamespace

import pytest

from mlx_vlm.tools import (
    SPECS,
    _infer_tool_parser,
    _infer_tool_parser_from_processor,
    load_tool_module,
    parsers,
    process_tool_calls,
)

PARSER_NAMES = sorted(
    module.name
    for module in pkgutil.iter_modules(parsers.__path__)
    if not module.ispkg and not module.name.startswith("_")
)
WEATHER_ARGS = {"city": "Paris", "days": 3}
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]
# Literal wire examples are independent of the parser's marker constants.
WIRE_CALLS = {
    "atem": 'to=self<|message|><atem:function_calls><atem:invoke name="get_weather">'
    '<atem:parameter name="city">Paris</atem:parameter>'
    '<atem:parameter name="days">3</atem:parameter></atem:invoke></atem:function_calls>',
    "cohere2_moe": '<|START_ACTION|>{"tool_name":"get_weather",'
    '"parameters":{"city":"Paris","days":3}}<|END_ACTION|>',
    "gemma4": '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>',
    "glm47": "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
    "json_tools": '<tool_call>{"name":"get_weather",'
    '"arguments":{"city":"Paris","days":3}}</tool_call>',
    "kimi_k2": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    '<|tool_call_argument_begin|>{"city":"Paris","days":3}'
    "<|tool_call_end|><|tool_calls_section_end|>",
    "longcat": "<longcat_tool_call>get_weather<longcat_arg_key>city</longcat_arg_key>"
    "<longcat_arg_value>Paris</longcat_arg_value><longcat_arg_key>days</longcat_arg_key>"
    "<longcat_arg_value>3</longcat_arg_value></longcat_tool_call>",
    "minicpm5": '<function name="get_weather"><param name="city">Paris</param>'
    '<param name="days">3</param></function>',
    "minimax_m2": '<minimax:tool_call><invoke name="get_weather">'
    '<parameter name="city">Paris</parameter><parameter name="days">3</parameter>'
    "</invoke></minimax:tool_call>",
    "minimax_m3": ']<]minimax[>[<tool_call>]<]minimax[>[<invoke name="get_weather">'
    "]<]minimax[>[<city>Paris]<]minimax[>[</city>]<]minimax[>[<days>3"
    "]<]minimax[>[</days>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
    "mistral": '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris", "days": 3}',
    "pythonic": '<|tool_call_start|>[get_weather(city="Paris", days=3)]<|tool_call_end|>',
    "qwen3_coder": "<tool_call>\n<function=get_weather><parameter=city>Paris</parameter>"
    "<parameter=days>3</parameter></function></tool_call>",
}


def _parse(name, text, tools=None):
    return load_tool_module(name).parse_tool_call(text, tools)


def _arguments(call):
    value = call["arguments"]
    return json.loads(value) if isinstance(value, str) else value


def test_every_parser_has_registration_and_wire_example():
    assert set(PARSER_NAMES) == {spec.name for spec in SPECS} == set(WIRE_CALLS)


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_contract(name):
    parser = load_tool_module(name)
    wire = WIRE_CALLS[name]
    body = wire.removeprefix(parser.tool_call_start).removesuffix(parser.tool_call_end)
    parsed = parser.parse_tool_call(body, WEATHER_TOOLS)
    calls = parsed if isinstance(parsed, list) else [parsed]
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert _arguments(calls[0]) == WEATHER_ARGS
    for count in (1, 2):
        # Newlines delimit Mistral calls, which have no closing marker.
        output = "Before\n" + "\n".join([wire] * count) + "\nAfter"
        result = process_tool_calls(output, parser, WEATHER_TOOLS)
        assert result.remaining_text.split() == ["Before", "After"]
        assert len(result.calls) == count
        assert len({call["id"] for call in result.calls}) == count
        for index, call in enumerate(result.calls):
            assert call["id"]
            assert call["type"] == "function"
            assert call["index"] == index
            assert call["function"]["name"] == "get_weather"
            assert json.loads(call["function"]["arguments"]) == WEATHER_ARGS
    result = process_tool_calls("Ordinary assistant prose.", parser, WEATHER_TOOLS)
    assert result.calls == []
    assert result.remaining_text == "Ordinary assistant prose."


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_selection(name):
    # Specific formats must outrank the generic JSON fallback.
    generic = "<tool_call> tool_call.name"
    template = WIRE_CALLS[name] + generic
    for value in (
        template,
        {"default": generic, "tool_use": template},
        [
            {"name": "default", "template": generic},
            {"name": "tool_use", "template": template},
        ],
    ):
        assert _infer_tool_parser(value) == name
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template))
    assert _infer_tool_parser_from_processor(processor) == name
    assert _infer_tool_parser("anything", override=name) == name


@pytest.mark.parametrize(
    "name,text,error",
    [
        ("atem", "not a tool call", "No ATEM function invocation"),
        ("gemma4", "just a normal model response, no tool call here", None),
        ("mistral", "not a tool call at all", None),
        (
            "pythonic",
            "[write_file(content='const player = { x: 0, y: 1 };)]",
            "Invalid Pythonic tool call",
        ),
        ("pythonic", "[write_file(content=get_content())]", "must be a literal value"),
    ],
)
def test_invalid_calls(name, text, error):
    with pytest.raises(ValueError, match=error):
        _parse(name, text)


@pytest.mark.parametrize("multiple", [False, True], ids=["object", "array"])
def test_cohere_action_parses_to_openai_tool_calls(multiple):
    pattern = "<|channel>" if multiple else "foo"
    action = {
        "tool_call_id": "1",
        "tool_name": "grep",
        "parameters": {"pattern": pattern},
    }
    second = {
        "tool_call_id_id": "2",
        "tool_name": "read",
        "parameters": {"path": "file.py"},
    }
    result = _parse(
        "cohere2_moe",
        json.dumps([action, second] if multiple else action).replace(
            "<|channel>", r"<\|channel>"
        ),
    )
    calls = result if multiple else [result]
    assert [call["name"] for call in calls] == (
        ["grep", "read"] if multiple else ["grep"]
    )
    assert json.loads(calls[0]["arguments"]) == {"pattern": pattern}
    if multiple:
        assert json.loads(calls[1]["arguments"]) == {"path": "file.py"}


@pytest.mark.parametrize(
    "text,name,args",
    [
        (
            '<|tool_call>call:edit-file{path:<|"|>test.txt<|"|>,edits:[{newText:<|"|>orange<|"|>,oldText:<|"|>apple<|"|>}]}<tool_call|>',
            "edit-file",
            {"path": "test.txt", "edits": [{"newText": "orange", "oldText": "apple"}]},
        ),
        ("get_weather{city:Austin}", "get_weather", {"city": "Austin"}),
    ],
)
def test_gemma_call_syntax(text, name, args):
    result = _parse("gemma4", text)
    assert result["name"] == name
    assert json.loads(result["arguments"]) == args


def test_gemma_ignores_non_call_prose():
    result = process_tool_calls(
        "Like call: prince", load_tool_module("gemma4"), tools=None
    )
    assert result.calls == []
    assert result.remaining_text == "Like call: prince"


def test_glm_tool_name_drops_the_trailing_newline():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zip": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            },
        }
    ]
    result = _parse(
        "glm47",
        "get_weather\n<arg_key>zip</arg_key>\n<arg_value>10001</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n",
        tools,
    )
    assert result["name"] == "get_weather"


@pytest.mark.parametrize(
    "output",
    [
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]',
        WIRE_CALLS["mistral"],
    ],
    ids=["v3", "v11"],
)
def test_mistral_process_tool_calls(output):
    result = process_tool_calls(output, load_tool_module("mistral"), WEATHER_TOOLS)
    assert result.remaining_text == ""
    assert len(result.calls) == 1
    assert result.calls[0]["function"]["name"] == "get_weather"
    assert json.loads(result.calls[0]["function"]["arguments"]) == WEATHER_ARGS


@pytest.mark.parametrize(
    "text,name,args",
    [
        (
            '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]',
            "write_file",
            {"path": "game.html", "content": '<canvas id="game">\n</canvas>'},
        ),
        (
            "[configure(options={'position': [0, 1], 'enabled': True})]",
            "configure",
            {"options": {"position": [0, 1], "enabled": True}},
        ),
    ],
)
def test_pythonic_literal_arguments(text, name, args):
    assert _parse("pythonic", text) == {"name": name, "arguments": args}


@pytest.mark.parametrize(
    "template",
    [None, {}, [], 123, {"tool_use": None}, {"default": "x", "tool_use": "y"}],
)
def test_non_routable_inputs_return_none(template):
    assert _infer_tool_parser(template) is None


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError):
        _infer_tool_parser("anything", override="does_not_exist")
