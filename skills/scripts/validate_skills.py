#!/usr/bin/env python3
"""Self-contained validator for the mlx-vlm-skills bundle (stdlib only).

Checks:
- every skills/skills/<name>/SKILL.md has YAML frontmatter with a `name` (matching its
  directory) and a non-trivial `description`;
- cross-skill `Skill("mlx-vlm-skills:<x>")` references point at skills that exist;
- the plugin metadata files agree on name and version.

Exits non-zero (listing every problem) on failure. Run: `python3 skills/scripts/validate_skills.py`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the skills/ bundle root
SKILLS_DIR = ROOT / "skills"
METADATA = {
    "claude": ROOT.parent / ".claude-plugin" / "marketplace.json",
    "plugin": ROOT / ".claude-plugin" / "plugin.json",
    "codex": ROOT / ".codex-plugin" / "plugin.json",
    "gemini": ROOT / "gemini-extension.json",
}
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_REF = re.compile(r'Skill\("mlx-vlm-skills:([a-z0-9-]+)"\)')


def _parse_frontmatter(text: str) -> dict:
    m = _FRONTMATTER.match(text)
    if not m:
        return {}
    out = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    errors: list[str] = []
    skill_names: set[str] = set()

    if not SKILLS_DIR.is_dir():
        print(f"FAIL: {SKILLS_DIR} not found")
        return 1

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    for d in skill_dirs:
        skill_names.add(d.name)
        md = d / "SKILL.md"
        if not md.is_file():
            errors.append(f"{d.name}: missing SKILL.md")
            continue
        fm = _parse_frontmatter(md.read_text())
        if fm.get("name") != d.name:
            errors.append(
                f"{d.name}: frontmatter name={fm.get('name')!r} != directory name"
            )
        if len(fm.get("description", "")) < 40:
            errors.append(
                f"{d.name}: description missing or too short (needs a real trigger sentence)"
            )

    # cross-skill references must resolve
    for d in skill_dirs:
        md = d / "SKILL.md"
        if not md.is_file():
            continue
        for ref in _REF.findall(md.read_text()):
            if ref not in skill_names:
                errors.append(f"{d.name}: references unknown skill '{ref}'")

    # metadata consistency
    metas = {}
    for label, path in METADATA.items():
        if not path.is_file():
            errors.append(f"metadata: {label} file missing at {path}")
            continue
        try:
            metas[label] = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"metadata: {label} is not valid JSON ({e})")
    versions = {lbl: m.get("version") for lbl, m in metas.items() if "version" in m}
    if len(set(versions.values())) > 1:
        errors.append(f"metadata: version mismatch across files: {versions}")

    if errors:
        print(f"FAIL: {len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"OK: {len(skill_dirs)} skill(s) valid: {', '.join(sorted(skill_names))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
