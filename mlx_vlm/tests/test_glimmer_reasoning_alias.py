"""Muse Glimmer reasoning_effort -> reasoning_strength alias.

Glimmer's chat template renders the system reasoning prompt from the
``reasoning_strength`` kwarg (``render_reasoning()``, default "high"):
the server passes ``reasoning_effort`` (OpenAI standard), so without the
alias the requested effort is silently dropped and the template always
renders "high".
"""

from mlx_vlm.server.generation import GenerationArguments


def test_no_reasoning_strength_when_effort_unset():
    kw = GenerationArguments(enable_thinking=True, reasoning=True).to_template_kwargs()

    assert "reasoning_strength" not in kw
    assert "reasoning_effort" not in kw
