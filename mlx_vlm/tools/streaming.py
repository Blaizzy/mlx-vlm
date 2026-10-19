"""Incremental function arguments, with final validation by the model's parser."""

import json
import re
import uuid

_DECODER = json.JSONDecoder()


def _json_calls(text, name_key="name", args_key="arguments"):
    """Read complete metadata and the argument value's received JSON prefix."""
    text = text.lstrip()
    offset = 1 if text.startswith("[") else 0
    while offset < len(text):
        offset += len(text[offset:]) - len(text[offset:].lstrip(" \r\n\t,"))
        if text[offset : offset + 1] != "{":
            return
        offset += 1
        name, arguments = None, ""
        while offset < len(text):
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            if text[offset : offset + 1] == "}":
                offset += 1
                break
            try:
                key, end = _DECODER.raw_decode(text, offset)
            except ValueError:
                return
            offset = end + len(text[end:]) - len(text[end:].lstrip())
            if text[offset : offset + 1] != ":":
                return
            offset += 1
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            start = offset
            try:
                value, offset = _DECODER.raw_decode(text, start)
            except ValueError:
                if name and key == args_key and text[start : start + 1] == "{":
                    yield name, text[start:]
                return
            if key in (name_key, "name") and isinstance(value, str):
                name = value.strip()
            if key == args_key:
                # String-encoded arguments are decoded only once complete.
                arguments = value if isinstance(value, str) else text[start:offset]
            offset += len(text[offset:]) - len(text[offset:].lstrip())
            if text[offset : offset + 1] == ",":
                offset += 1
            elif text[offset : offset + 1] != "}":
                if name:
                    yield name, arguments
                return
        if name:
            yield name, arguments
        else:
            return


def _xml_call(text, tools, module):
    match = re.match(r"\s*<function=([^>]+)>", text)
    if not match:
        return []
    name = match[1]
    body = text[match.end() :]
    properties = module._get_arguments_config(name, tools)
    arguments = {}
    while body:
        param = re.match(r"\s*<parameter=([^>]+)>", body)
        if not param:
            break
        key = param[1]
        body = body[param.end() :]
        end = body.find("</parameter>")
        value = body if end < 0 else body[:end]
        value = value.removeprefix("\n").removesuffix("\n")
        if end < 0:
            # Keep possible closing-tag fragments out of string values.
            value = _without_marker_tail(value, "</parameter>").removesuffix("\n")
            if properties.get(key, {}).get(
                "type"
            ) in module._string_types and not "null".startswith(value.lower()):
                prefix = json.dumps(arguments, ensure_ascii=False)[:-1]
                separator = ", " if arguments else ""
                return [
                    (
                        name,
                        prefix
                        + separator
                        + json.dumps(key)
                        + ": "
                        + json.dumps(value, ensure_ascii=False)[:-1],
                    )
                ]
            break
        arguments[key] = module._convert_param_value(value, key, properties)
        body = body[end + len("</parameter>") :]
    return [(name, json.dumps(arguments, ensure_ascii=False)[:-1])]


def _without_marker_tail(text, marker):
    for size in range(min(len(text), len(marker) - 1), 0, -1):
        if text.endswith(marker[:size]):
            return text[:-size]
    return text


def _gemma_call(text, module):
    match = re.match(r"\s*(?:call:|:)?([\w.:-]+)\{", text)
    if not match:
        return []
    arguments = {}
    entries = module._split_top_level(text[match.end() :])
    for index, entry in enumerate(entries):
        colon = module._find_top_level_colon(entry)
        if colon < 0:
            break
        key, value = entry[:colon].strip(), entry[colon + 1 :].lstrip()
        if value.startswith(module._ESCAPE):
            value_text = value[len(module._ESCAPE) :]
            if module._ESCAPE not in value_text:
                value_text = _without_marker_tail(value_text, module._ESCAPE)
                prefix = json.dumps(arguments)[:-1] + (", " if arguments else "")
                return [
                    (
                        match[1],
                        prefix + json.dumps(key) + ": " + json.dumps(value_text)[:-1],
                    )
                ]
        elif index == len(entries) - 1:
            break
        arguments[key] = module._parse_value(value)
    return [(match[1], json.dumps(arguments)[:-1])]


class ToolCallDeltaState:
    def __init__(self, module, tools):
        self.module = module
        self.parser_name = getattr(module, "__name__", "").rsplit(".", 1)[-1]
        self.tools = tools
        self.buffer = ""
        self.inside = False
        self.offset = 0
        self.calls = {}

    def feed(self, text):
        if not self.tools or self.parser_name not in (
            "json_tools",
            "cohere2_moe",
            "qwen3_coder",
            "gemma4",
        ):
            return []
        self.buffer += text or ""
        deltas = []
        start = self.module.tool_call_start
        end = self.module.tool_call_end or "\n"
        while self.buffer:
            if not self.inside:
                position = self.buffer.find(start)
                if position < 0:
                    self.buffer = self.buffer[-max(1, len(start) - 1) :]
                    break
                self.buffer = self.buffer[position + len(start) :]
                self.inside = True
            position = self.buffer.find(end)
            body = self.buffer if position < 0 else self.buffer[:position]
            if position < 0:
                body = _without_marker_tail(body, end)
                name = self.parser_name
                if name == "json_tools":
                    candidates = _json_calls(body)
                elif name == "cohere2_moe":
                    body = _without_marker_tail(body, "\\|").replace("\\|", "|")
                    candidates = _json_calls(body, "tool_name", "parameters")
                elif name == "qwen3_coder":
                    candidates = _xml_call(body, self.tools, self.module)
                elif name == "gemma4":
                    candidates = _gemma_call(body, self.module)
                else:
                    candidates = []
            else:
                try:
                    parsed = self.module.parse_tool_call(body.strip(), self.tools)
                except (ValueError, KeyError, TypeError):
                    if len(self.calls) > self.offset:
                        raise
                    parsed = []
                candidates = [parsed] if isinstance(parsed, dict) else parsed
                candidates = [
                    (call["name"].strip(), call["arguments"]) for call in candidates
                ]
            count = 0
            for count, (name, arguments) in enumerate(candidates, 1):
                index = self.offset + count - 1
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                if position >= 0 and index in self.calls:
                    sent = self.calls[index]["function"]["arguments"]
                    if not arguments.startswith(sent):
                        # JSON formatting is immaterial, but streamed bytes cannot
                        # be retracted. Recover the original complete argument value.
                        fields = (
                            ("tool_name", "parameters")
                            if self.parser_name == "cohere2_moe"
                            else ("name", "arguments")
                        )
                        raw_body = (
                            body.replace("\\|", "|")
                            if self.parser_name == "cohere2_moe"
                            else body
                        )
                        raw = list(_json_calls(raw_body, *fields))
                        if (
                            count <= len(raw)
                            and raw[count - 1][1].startswith(sent)
                            and json.loads(raw[count - 1][1]) == json.loads(arguments)
                        ):
                            arguments = raw[count - 1][1]
                delta = self._update(index, name, arguments)
                if delta:
                    deltas.append(delta)
            if position < 0:
                break
            self.offset += count
            self.buffer = self.buffer[position + len(end) :]
            self.inside = False
        return deltas

    def _update(self, index, name, arguments):
        previous = self.calls.get(index)
        if previous is None:
            call = {
                "index": index,
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            self.calls[index] = call
            return {**call, "function": dict(call["function"])}
        function = previous["function"]
        if function["name"] != name or not arguments.startswith(function["arguments"]):
            raise ValueError("Tool arguments changed after they were streamed.")
        fragment = arguments[len(function["arguments"]) :]
        function["arguments"] = arguments
        if fragment:
            return {"index": index, "function": {"arguments": fragment}}
        return None

    def finish(self, calls):
        if len(calls) < len(self.calls):
            raise ValueError("Generation ended with an incomplete tool call.")
        deltas = []
        for index, call in enumerate(calls):
            previous = self.calls.get(index)
            if previous is not None:
                if previous["function"]["name"] != call["function"][
                    "name"
                ] or json.loads(previous["function"]["arguments"]) != json.loads(
                    call["function"]["arguments"]
                ):
                    raise ValueError(
                        "Streamed tool arguments do not match the completed call."
                    )
                call.update(id=previous["id"], function=dict(previous["function"]))
            else:
                deltas.append(call)
        return deltas
