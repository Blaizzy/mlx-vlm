"""Keep the Nemotron VoiceChat audio dependency floor in sync with its import."""

from __future__ import annotations

import ast
import re
from pathlib import Path

MIN_MLX_AUDIO_VERSION = (0, 4, 8)
CODEC_MODULE = "mlx_audio.codec.models.nemotron_voicechat"
REQUIREMENT = re.compile(r"^mlx-audio>=(\d+)\.(\d+)\.(\d+)$")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mlx_audio_lower_bound(requirements_path: Path) -> tuple[int, ...]:
    entries = [
        line
        for raw_line in requirements_path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.split("#", 1)[0].strip()).startswith("mlx-audio")
    ]
    assert (
        len(entries) == 1
    ), f"expected exactly one mlx-audio requirement, found {len(entries)}"
    match = REQUIREMENT.fullmatch(entries[0])
    assert match is not None, (
        "mlx-audio requirement must be an explicit semantic-version lower bound; "
        f"found {entries[0]!r}"
    )
    return tuple(int(part) for part in match.groups())


def test_nemotron_voicechat_dependency_floor() -> None:
    """Require the published mlx-audio release that provides the codec import."""
    root = _repository_root()
    config_path = root / "mlx_vlm" / "models" / "nemotron_voicechat" / "config.py"
    requirements_path = root / "requirements.txt"

    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module == CODEC_MODULE
        for node in ast.walk(tree)
    ), f"Nemotron VoiceChat config must import {CODEC_MODULE}"
    lower_bound = _mlx_audio_lower_bound(requirements_path)
    assert lower_bound >= MIN_MLX_AUDIO_VERSION, (
        "Nemotron VoiceChat requires mlx-audio>=0.4.8; "
        f"found mlx-audio>={'.'.join(map(str, lower_bound))}"
    )


if __name__ == "__main__":
    test_nemotron_voicechat_dependency_floor()
    print("PASS: Nemotron VoiceChat dependency floor")
