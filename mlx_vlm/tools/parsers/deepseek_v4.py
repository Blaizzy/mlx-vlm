"""DeepSeek-V4 DSML tool calls.

DeepSeek-V4 emits tool calls as a DSML block (see
``mlx_vlm/models/deepseek_v4/encoding_dsv4.py``)::

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
    <｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

``string="true"`` values are taken verbatim; any other value is JSON (falling
back to the raw text when it does not parse). Whitespace between tags is
tolerated, unlike the strict reference parser.
"""

import json
import re
from typing import Any, List, Optional

DSML = "｜DSML｜"

tool_call_start = f"<{DSML}tool_calls>"
tool_call_end = f"</{DSML}tool_calls>"

_INVOKE_RE = re.compile(
    rf'<{DSML}invoke\s+name="([^"]+)"\s*>(.*?)</{DSML}invoke>', re.DOTALL
)
_PARAMETER_RE = re.compile(
    rf'<{DSML}parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>'
    rf"(.*?)</{DSML}parameter>",
    re.DOTALL,
)


def _decode_value(value: str, is_string: Optional[str]) -> Any:
    if is_string == "true":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_call(text: str, tools: Any = None) -> List[dict]:
    calls = []
    for name, body in _INVOKE_RE.findall(text):
        arguments = {}
        for key, is_string, value in _PARAMETER_RE.findall(body):
            if key in arguments:
                raise ValueError(f"Duplicate DSML parameter: {key!r}")
            arguments[key] = _decode_value(value, is_string)
        calls.append(dict(name=name, arguments=arguments))
    if not calls:
        raise ValueError("No DSML tool invocation found.")
    return calls
