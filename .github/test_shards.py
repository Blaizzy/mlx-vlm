"""Keep related MLX tests together so each runner can reuse compiled kernels."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "mlx_vlm" / "tests"

# Balance using Actions timings, not test counts. The last shard also picks up
# new test files automatically, so adding a suite never silently drops coverage.
GROUPS = (
    {
        "test_cache.py",
        "test_moe_offload.py",
        "test_model_ops.py",
        "test_generate.py",
        "test_processors.py",
        "test_cli.py",
        "test_tool_streaming.py",
        "test_sampling_defaults.py",
    },
    {
        "test_models.py",
        "test_extraction_models.py",
        "test_server.py",
        "test_audio_models.py",
        "test_training_api.py",
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
    return [sorted(group) for group in (*GROUPS, files - assigned)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=int, choices=range(1, len(GROUPS) + 2))
    parser.add_argument(
        "--list", action="store_true", help="List files without running"
    )
    args = parser.parse_args()
    paths = [str(TESTS / name) for name in shards()[args.shard - 1]]
    if not paths:
        parser.error("Shard has no test files")
    if args.list:
        print("\n".join(paths))
    else:
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
