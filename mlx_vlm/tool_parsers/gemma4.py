import json

import regex as re

_tool_call_regex = re.compile(r"call:(\w+)\{(.*?)\}", re.DOTALL)

tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"


def parse_tool_call(text, tools=None):
    match = _tool_call_regex.findall(text)
    if not match:
        raise ValueError("No function call found in tool call text.")
    func_name = match[0][0]
    args_str = match[0][1]
    arguments = {}
    escape = '<|"|>'
    while args_str:
        split = args_str.index(":")
        key = args_str[:split]
        args_str = args_str[split + 1 :]
        # Parse an escaped string value
        if args_str.startswith(escape):
            args_str = args_str[len(escape) :]
            split = args_str.index(escape)
            arguments[key] = args_str[:split]
            args_str = args_str[split + len(escape) :]
            if args_str.startswith(","):
                args_str = args_str[1:]
            continue
        # Parse a non-string value
        if "," in args_str:
            split = args_str.index(",")
        else:
            split = len(args_str)

        value = args_str[:split]
        args_str = args_str[split + 1 :]

        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value

    return dict(name=func_name, arguments=arguments)
