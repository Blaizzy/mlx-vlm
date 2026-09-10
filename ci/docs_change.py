from __future__ import annotations

from typing import Any, Sequence

from ci.change_rules import ChangeContext, ChangeMatch


class DocsChange:
    """Plan trusted GitHub-hosted documentation validation."""

    name = "docs_change"

    def plan(
        self, matches: Sequence[ChangeMatch], context: ChangeContext
    ) -> dict[str, Any]:
        paths = sorted({match.path for match in matches})
        return {
            "component": self.name,
            "checks": [
                {
                    "id": "docs",
                    "work_type": "Docs",
                    "component": self.name,
                    "execution_target": "github_hosted",
                    "changed_paths": paths,
                }
            ],
            "jobs": [],
            "gates": [],
            "blocked": [],
        }
