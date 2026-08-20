# Vendored from mlx-lm 0.31.3 (mlx_lm/tool_parsers/json_tools.py).
# Copyright © 2025 Apple Inc.

import json

tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(text, tools=None):
    return json.loads(text.strip())
