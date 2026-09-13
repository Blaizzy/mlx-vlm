"""Tool parser modules for mlx-vlm.

Each module here implements one tool-call format: two marker strings
(``tool_call_start`` / ``tool_call_end``) and a ``parse_tool_call`` callable.
The registry (:mod:`mlx_vlm.tools.registry`) maps a chat template to one of
these modules; the extractor drives it.
"""
