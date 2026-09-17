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


def _weather_tools(**fields):
    properties = {name: dict(type=kind) for name, kind in fields.items()}
    return [
        dict(
            type="function",
            function=dict(
                name="get_weather",
                parameters=dict(type="object", properties=properties),
            ),
        )
    ]


WEATHER_TOOLS = _weather_tools(city="string", days="integer")
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


WIRE_VARIANTS = {
    "mistral": [
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]'
    ],
}


def _parse(name, text, tools=None):
    return load_tool_module(name).parse_tool_call(text, tools)


def _call(name, **arguments):
    return dict(name=name, arguments=arguments)


def _arguments(call):
    value = call["arguments"]
    return json.loads(value) if isinstance(value, str) else value


def test_every_parser_has_registration_and_wire_example():
    assert set(PARSER_NAMES) == {spec.name for spec in SPECS} == set(WIRE_CALLS)
    assert set(WIRE_VARIANTS) <= set(PARSER_NAMES)


@pytest.mark.parametrize("name", PARSER_NAMES)
def test_parser_contract(name):
    parser = load_tool_module(name)
    for wire in [WIRE_CALLS[name], *WIRE_VARIANTS.get(name, ())]:
        body = wire.removeprefix(parser.tool_call_start).removesuffix(
            parser.tool_call_end
        )
        parsed = parser.parse_tool_call(body, WEATHER_TOOLS)
        calls = parsed if isinstance(parsed, list) else [parsed]
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert _arguments(calls[0]) == WEATHER_ARGS
        for count, with_prose in [(1, False), (1, True), (2, True)]:
            # A bare call exercises EOF; newlines delimit Mistral's repeated calls.
            output = "\n".join([wire] * count)
            if with_prose:
                output = f"Before\n{output}\nAfter"
            result = process_tool_calls(output, parser, WEATHER_TOOLS)
            if with_prose:
                assert result.remaining_text.split() == ["Before", "After"]
            else:
                assert result.remaining_text == ""
            assert len(result.calls) == count
            assert len({call["id"] for call in result.calls}) == count
            for index, call in enumerate(result.calls):
                assert call["id"]
                assert call["type"] == "function"
                assert call["index"] == index
                assert call["function"]["name"] == "get_weather"
                assert json.loads(call["function"]["arguments"]) == WEATHER_ARGS
    for text in ("Ordinary assistant prose.", "Like call: prince"):
        result = process_tool_calls(text, parser, tools=None)
        assert result.calls == []
        assert result.remaining_text == text


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


@pytest.mark.parametrize(
    "parser,argument_type,text,expected,tools",
    [
        (
            "gemma4",
            str,
            '<|tool_call>call:edit-file{path:<|"|>test.txt<|"|>,edits:[{newText:<|"|>orange<|"|>,oldText:<|"|>apple<|"|>}]}<tool_call|>',
            _call(
                "edit-file",
                path="test.txt",
                edits=[{"newText": "orange", "oldText": "apple"}],
            ),
            None,
        ),
        (
            "gemma4",
            str,
            "get_weather{city:Austin}",
            _call("get_weather", city="Austin"),
            None,
        ),
        (
            "pythonic",
            dict,
            '[write_file(path="game.html", content="<canvas id="game">\n</canvas>")]',
            _call(
                "write_file", path="game.html", content='<canvas id="game">\n</canvas>'
            ),
            None,
        ),
        (
            "pythonic",
            dict,
            "[configure(options={'position': [0, 1], 'enabled': True})]",
            _call("configure", options={"position": [0, 1], "enabled": True}),
            None,
        ),
        (
            "cohere2_moe",
            str,
            '{"tool_call_id":"1","tool_name":"grep","parameters":{"pattern":"foo"}}',
            _call("grep", pattern="foo"),
            None,
        ),
        (
            "cohere2_moe",
            str,
            r'[{"tool_call_id":"1","tool_name":"grep","parameters":{"pattern":"<\|channel>"}},'
            '{"tool_call_id_id":"2","tool_name":"read","parameters":{"path":"file.py"}}]',
            [_call("grep", pattern="<|channel>"), _call("read", path="file.py")],
            None,
        ),
        (
            "glm47",
            dict,
            "get_weather\n<arg_key>zip</arg_key>\n<arg_value>10001</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n",
            _call("get_weather", zip="10001", days=3),
            _weather_tools(zip="string", days="integer"),
        ),
    ],
    ids=[
        "gemma-nested",
        "gemma-bare",
        "pythonic-html",
        "pythonic-nested",
        "cohere-object",
        "cohere-array-escape",
        "glm-newline",
    ],
)
def test_parser_syntax(parser, argument_type, text, expected, tools):
    result = _parse(parser, text, tools)
    assert isinstance(result, type(expected))
    calls = result if isinstance(result, list) else [result]
    expected_calls = expected if isinstance(expected, list) else [expected]
    assert all(isinstance(call["arguments"], argument_type) for call in calls)
    assert [dict(call, arguments=_arguments(call)) for call in calls] == expected_calls


@pytest.mark.parametrize(
    "template",
    [None, {}, [], 123, {"tool_use": None}, {"default": "x", "tool_use": "y"}],
)
def test_non_routable_inputs_return_none(template):
    assert _infer_tool_parser(template) is None


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError):
        _infer_tool_parser("anything", override="does_not_exist")
