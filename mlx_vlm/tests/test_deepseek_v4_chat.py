"""DeepSeek-V4 chat formatting: thinking mode, reasoning effort and DSML tool calls."""

import json
from types import SimpleNamespace

import pytest

from mlx_vlm.models.deepseek_v4 import encoding_dsv4
from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import (
    DEFAULT_CHAT_TEMPLATE,
    DeepseekV4Processor,
)
from mlx_vlm.tools import load_tool_module, process_tool_calls
from mlx_vlm.tools.registry import _infer_tool_parser_from_processor

BOS = "<｜begin▁of▁sentence｜>"
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }
]


class FakeTokenizer:
    """Renders the processor's default Jinja template like a HF tokenizer would."""

    chat_template = DEFAULT_CHAT_TEMPLATE

    def __init__(self):
        import jinja2

        self._template = jinja2.Environment().from_string(DEFAULT_CHAT_TEMPLATE)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **_
    ):
        text = self._template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) for ch in text]


def make_processor():
    processor = object.__new__(DeepseekV4Processor)
    processor.tokenizer = FakeTokenizer()
    return processor


def render(messages, **kwargs):
    kwargs.setdefault("add_generation_prompt", True)
    return make_processor().apply_chat_template(messages, tokenize=False, **kwargs)


def test_single_turn_chat_matches_previous_default_template():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    legacy = FakeTokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    assert render(messages) == legacy
    assert render(messages).endswith("<｜User｜>What is 2+2?<｜Assistant｜></think>")


def test_multi_turn_history_uses_official_format():
    # The reference encoding closes the (empty) thinking block before earlier
    # assistant replies; the old default template omitted that `</think>`.
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert render(messages) == (
        f"{BOS}<｜User｜>Hi<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜>"
        "<｜User｜>What is 2+2?<｜Assistant｜></think>"
    )
    assert render(messages) == encoding_dsv4.encode_messages(
        messages, thinking_mode="chat"
    )


def test_thinking_mode_opens_think_block():
    prompt = render([{"role": "user", "content": "Hi"}], enable_thinking=True)
    assert prompt == f"{BOS}<｜User｜>Hi<｜Assistant｜><think>"


@pytest.mark.parametrize(
    "effort,expected",
    [("high", "high"), ("max", "max"), ("medium", "high"), ("xhigh", "max")],
)
def test_reasoning_effort_prefix_in_thinking_mode(effort, expected):
    prompt = render(
        [{"role": "user", "content": "Hi"}],
        enable_thinking=True,
        reasoning_effort=effort,
    )
    assert prompt.startswith(BOS + encoding_dsv4.REASONING_EFFORT_PROMPTS[expected])


def test_reasoning_effort_ignored_without_thinking():
    plain = render([{"role": "user", "content": "Hi"}])
    assert render([{"role": "user", "content": "Hi"}], reasoning_effort="max") == plain


def test_tools_are_rendered_into_system_prompt():
    prompt = render(
        [{"role": "user", "content": "Weather in Paris?"}], tools=WEATHER_TOOLS
    )
    assert "## Tools" in prompt and '"name": "get_weather"' in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert prompt.index("## Tools") < prompt.index("<｜User｜>Weather in Paris?")


def test_tools_join_existing_system_message():
    prompt = render(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather?"},
        ],
        tools=WEATHER_TOOLS,
    )
    assert prompt.startswith(BOS + "Be brief.\n\n## Tools")


def test_tool_call_round_trip_messages():
    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    # Servers may hand arguments over as a dict instead of a JSON string.
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris", "days": 3},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21C"},
    ]
    prompt = render(messages, tools=WEATHER_TOOLS)
    assert (
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>'
        in prompt
    )
    assert (
        '<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>' in prompt
    )
    assert "<tool_result>Sunny, 21C</tool_result>" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_image_blocks_still_become_placeholders():
    prompt = render(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
    )
    assert "What is this?" in prompt


def test_no_generation_prompt_when_not_requested():
    prompt = make_processor().apply_chat_template(
        [{"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=False
    )
    assert prompt == f"{BOS}<｜User｜>Hi"


def test_tokenize_returns_token_ids():
    ids = make_processor().apply_chat_template(
        [{"role": "user", "content": "Hi"}], tokenize=True, add_generation_prompt=True
    )
    assert ids == [ord(ch) for ch in f"{BOS}<｜User｜>Hi<｜Assistant｜></think>"]


def test_explicit_chat_template_keeps_jinja_path():
    processor = make_processor()
    messages = [{"role": "user", "content": "Hi"}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template="custom"
    )
    assert prompt == FakeTokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def test_model_provided_template_is_respected():
    # A checkpoint that ships its own chat_template.jinja keeps rendering through it.
    processor = make_processor()
    processor.tokenizer.chat_template = "{{ 'model template' }}"
    calls = []
    processor.tokenizer.apply_chat_template = (
        lambda messages, **kw: calls.append(kw) or "model template"
    )
    assert (
        processor.apply_chat_template(
            [{"role": "user", "content": "Hi"}], tokenize=False
        )
        == "model template"
    )
    assert calls


def test_processor_declares_tool_parser():
    assert DeepseekV4Processor.tool_parser_type == "deepseek_v4"
    assert _infer_tool_parser_from_processor(make_processor()) == "deepseek_v4"


def test_processor_hint_does_not_override_explicit_parser():
    assert (
        _infer_tool_parser_from_processor(make_processor(), override="json_tools")
        == "json_tools"
    )


def test_processor_without_hint_still_uses_template():
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(chat_template="<tool_call> tool_call.name")
    )
    assert _infer_tool_parser_from_processor(processor) == "json_tools"


def test_parser_handles_typed_and_untyped_parameters():
    parser = load_tool_module("deepseek_v4")
    body = (
        '<｜DSML｜invoke name="search">\n'
        '<｜DSML｜parameter name="query" string="true">mlx "vlm"</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="filters" string="false">{"lang": ["py"], "top": 5}</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="raw">not json</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>"
    )
    calls = parser.parse_tool_call(body)
    assert calls == [
        {
            "name": "search",
            "arguments": {
                "query": 'mlx "vlm"',
                "filters": {"lang": ["py"], "top": 5},
                "raw": "not json",
            },
        }
    ]


def test_parser_handles_multiple_invokes_and_no_arguments():
    parser = load_tool_module("deepseek_v4")
    body = '<｜DSML｜invoke name="now">\n</｜DSML｜invoke>\n<｜DSML｜invoke name="ping">\n</｜DSML｜invoke>'
    assert [c["name"] for c in parser.parse_tool_call(body)] == ["now", "ping"]
    assert parser.parse_tool_call(body)[0]["arguments"] == {}


def test_parser_rejects_text_without_invoke():
    with pytest.raises(ValueError, match="No DSML tool invocation"):
        load_tool_module("deepseek_v4").parse_tool_call("just prose")


def test_model_output_after_thinking_is_extracted():
    output = (
        "Paris weather needed.</think>Let me check.\n"
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )
    result = process_tool_calls(output, load_tool_module("deepseek_v4"), WEATHER_TOOLS)
    assert len(result.calls) == 1
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert "Let me check." in result.remaining_text
