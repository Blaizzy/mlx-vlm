import pytest

from mlx_vlm.tools.parsers import atem

ATEM_TEMPLATE = """
<atem:function_calls>
<atem:invoke name="$FUNCTION_NAME">
<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>
</atem:invoke>
</atem:function_calls>
"""


def test_rejects_text_without_an_atem_invocation():
    with pytest.raises(ValueError, match="No ATEM function invocation"):
        atem.parse_tool_call("not a tool call")
