import json
from typing import Any

# Apertus 1.5 emits a JSON list of single-key objects, name -> arguments:
#   <|tools_prefix|>[{"get_weather": {"city": "Paris"}}, ...]<|tools_suffix|>
# <|tools_suffix|> is an EOS token, so generation stops before it and the
# call runs to the end of the output.
tool_call_start = "<|tools_prefix|>"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    try:
        parsed = json.loads(text.strip().removesuffix("<|tools_suffix|>"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse tool call from: {text}") from exc

    calls = parsed if isinstance(parsed, list) else [parsed]
    if calls and all(isinstance(c, dict) and len(c) == 1 for c in calls):
        return [
            dict(name=name, arguments=arguments)
            for c in calls
            for name, arguments in c.items()
        ]
    raise ValueError(f"Could not parse tool call from: {text}")
