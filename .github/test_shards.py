"""Keep related MLX tests together so each runner can reuse compiled kernels."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "mlx_vlm" / "tests"

# The first shard covers portable interfaces on Linux; model and Metal checks
# stay on macOS. Balance using Actions timings, not test counts. The last Mac
# shard picks up new files automatically so adding a suite cannot drop coverage.
GROUPS = (
    {
        "test_server.py",
        "test_cli.py",
        "test_sampling_defaults.py",
        "test_tool_streaming.py",
    },
    {
        "test_cache.py",
        "test_moe_offload.py",
        "test_processors.py",
        "test_training_api.py",
    },
    {
        "test_models.py",
        "test_extraction_models.py",
        "test_generate.py",
        "test_audio_models.py",
        "test_image_generation_models.py",
        "test_training_metrics.py",
    },
)


def shards():
    files = {
        str(path.relative_to(TESTS))
        for pattern in ("test_*.py", "*_test.py")
        for path in TESTS.rglob(pattern)
    }
    assigned = set().union(*GROUPS)
    if len(assigned) != sum(map(len, GROUPS)):
        raise ValueError("A test file is assigned to more than one shard")
    if missing := assigned - files:
        raise ValueError(f"Shard configuration references missing tests: {missing}")
    # Start the long training contract early so it can compile while other
    # workers drain the model cases instead of leaving it as a final straggler.
    return [
        sorted(group, key=lambda name: (name != "test_training_api.py", name))
        for group in (*GROUPS, files - assigned)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int, choices=range(1, len(GROUPS) + 2))
    args = parser.parse_args()
    paths = [str(TESTS / name) for name in shards()[args.shard - 1]]
    if not paths:
        parser.error("Shard has no test files")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "3",
            "--dist=worksteal",
            "--durations=30",
            f"--junitxml={ROOT}/test-results/shard-{args.shard}.xml",
            *paths,
        ],
        cwd=ROOT / "mlx_vlm",
    )
    sys.exit(result.returncode)
