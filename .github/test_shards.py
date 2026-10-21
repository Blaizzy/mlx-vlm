"""Keep related MLX tests together so each runner can reuse compiled kernels."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "mlx_vlm" / "tests"

# Keep RAM-sensitive cache tests away from the larger model contracts.
# Portable interfaces and generation run on Linux; other groups use three Macs.
GROUPS = {
    "training-ops": {"test_cache.py", "test_moe_offload.py", "test_model_ops.py"},
    "media": {
        "test_extraction_models.py",
        "test_audio_models.py",
        "test_image_generation_models.py",
        "test_video_generation_models.py",
        "test_diffusion_models.py",
        "test_flux3.py",
        "test_flux3_upsampler.py",
        "test_llada_image.py",
        "test_moss_image.py",
        "test_muse_spark_video.py",
    },
    "interfaces": {
        "test_server.py",
        "test_server_stt_realtime.py",
        "test_cli.py",
        "test_sampling_defaults.py",
        "test_tool_streaming.py",
        "test_generate.py",
    },
}


def shards():
    files = {
        str(path.relative_to(TESTS))
        for pattern in ("test_*.py", "*_test.py")
        for path in TESTS.rglob(pattern)
    }
    groups = {name: set(files) for name, files in GROUPS.items()}
    groups["training-ops"].update(
        name for name in files if Path(name).name.startswith("test_train")
    )
    groups["interfaces"].update(
        name for name in files if Path(name).name.startswith("test_server")
    )
    assigned = set().union(*groups.values())
    if len(assigned) != sum(map(len, groups.values())):
        raise ValueError("A test file is assigned to more than one shard")
    if missing := assigned - files:
        raise ValueError(f"Shard configuration references missing tests: {missing}")
    # New model suites join this group automatically instead of losing coverage.
    groups["models"] = files - assigned
    # Start training compilation early instead of leaving it as a straggler.
    return {
        name: sorted(group, key=lambda path: (path != "test_training_api.py", path))
        for name, group in groups.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    groups = shards()
    parser.add_argument("group", choices=groups)
    args = parser.parse_args()
    paths = [str(TESTS / name) for name in groups[args.group]]
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
            f"--junitxml={ROOT}/test-results/{args.group}.xml",
            *paths,
        ],
        cwd=ROOT / "mlx_vlm",
    )
    sys.exit(result.returncode)
