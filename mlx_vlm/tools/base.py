"""The contract every tool parser satisfies.

Parsers live as modules in :mod:`mlx_vlm.tools.parsers`; each exposes two marker
strings and a ``parse_tool_call`` callable. ``tool_call_end`` is the empty
string for formats whose call ends at a newline rather than a closing marker.
The extractor (:mod:`mlx_vlm.tools.extractor`) drives any module shaped like
this; the :class:`ParserSpec` in :mod:`mlx_vlm.tools.registry` maps a chat
template to the module that owns it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Union, runtime_checkable


@runtime_checkable
class ToolParser(Protocol):
    tool_call_start: str
    tool_call_end: str

    def parse_tool_call(
        self, text: str, tools: Optional[Any] = None
    ) -> Union[dict, List[dict]]: ...
