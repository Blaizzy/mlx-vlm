# Vendored from mlx-lm 0.31.3 (mlx_lm/tool_parsers/qwen3_coder.py).
# Copyright © 2025 Apple Inc.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
from typing import Any, Optional

import regex as re

_function_regex = re.compile(r"<function=(.*?)</function>$", re.DOTALL)
# A parameter ends at </parameter>. Only an unclosed last parameter ends at
# the end of the function body: splitting at an inner "<parameter=" would
# corrupt closed values that contain that text.
_parameter_regex = re.compile(r"<parameter=(.*?)(</parameter>|\Z)", re.DOTALL)

_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> dict:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}
            return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Convert parameter value based on its type in the schema."""
    if param_value.lower() == "null":
        return None

    if param_name not in param_config:
        return param_value
    param = param_config[param_name]

    if isinstance(param, dict) and "type" in param:
        param_type = str(param["type"]).strip().lower()
    else:
        # Untyped: decode objects and arrays, keep scalars such as "123" as text.
        try:
            value = json.loads(param_value)
        except json.JSONDecodeError:
            return param_value
        return value if isinstance(value, (dict, list)) else param_value
    if param_type in _string_types:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        return int(param_value)
    elif param_type.startswith("num") or param_type.startswith("float"):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        return (
            float_param_value
            if (float_param_value - int_param_value) != 0
            else int_param_value
        )
    elif param_type in _bool_types:
        return param_value.lower() == "true"
    else:
        if (
            param_type in _obj_types
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value)
            except json.JSONDecodeError:
                return ast.literal_eval(param_value)

        return ast.literal_eval(param_value)


def _parse_xml_function_call(function_call_str: str, tools: Optional[Any]):
    end_index = function_call_str.index(">")
    function_name = function_call_str[:end_index]
    param_config = _get_arguments_config(function_name, tools)
    parameters = function_call_str[end_index + 1 :]
    param_dict = {}
    for match_text, closer in _parameter_regex.findall(parameters):
        if (idx := match_text.find(">")) == -1:
            continue
        param_name = match_text[:idx]
        param_value = match_text[idx + 1 :]
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        try:
            param_dict[param_name] = _convert_param_value(
                param_value, param_name, param_config
            )
        except (ValueError, SyntaxError, TypeError):
            # An unclosed last parameter may be cut short; drop it as main did
            # rather than fail the whole call.
            if closer:
                raise
    return dict(name=function_name, arguments=param_dict)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    match = _function_regex.findall(model_output)
    if not match:
        raise ValueError("No function provided.")
    return _parse_xml_function_call(match[0], tools)
