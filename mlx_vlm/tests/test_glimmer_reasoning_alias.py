"""Muse Glimmer reasoning_effort -> reasoning_strength alias.

Glimmer's chat template renders the system reasoning prompt from the
``reasoning_strength`` kwarg (``render_reasoning()``, default "high"):
the server passes ``reasoning_effort`` (OpenAI standard), so without the
alias the requested effort is silently dropped and the template always
renders "high".
"""

from mlx_vlm.server.generation import GenerationArguments


def test_reasoning_effort_emits_reasoning_strength_alias():
    kw = GenerationArguments(
        enable_thinking=True,
        reasoning=True,
        reasoning_effort="low",
    ).to_template_kwargs()

    assert kw["reasoning_effort"] == "low"
    assert kw["reasoning_strength"] == "low"


def test_reasoning_strength_alias_preserves_high():
    kw = GenerationArguments(
        enable_thinking=True,
        reasoning=True,
        reasoning_effort="high",
    ).to_template_kwargs()

    assert kw["reasoning_strength"] == "high"


def test_no_reasoning_strength_when_effort_unset():
    kw = GenerationArguments(enable_thinking=True, reasoning=True).to_template_kwargs()

    assert "reasoning_strength" not in kw
    assert "reasoning_effort" not in kw
