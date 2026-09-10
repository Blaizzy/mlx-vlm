from __future__ import annotations

from typing import Any

from ci.components.base import ComponentContext, ComponentRegistration


def _planners(context: ComponentContext) -> tuple[Any, ...]:
    from ci.docs_change import DocsChange

    return (DocsChange(),)


def _output():
    from ci.bot import DocsChangeOutput

    return DocsChangeOutput()


REGISTRATION = ComponentRegistration(
    name="docs_change",
    components=frozenset({"docs_change"}),
    planner_factory=_planners,
    output_factory=_output,
    work=frozenset(),
)
