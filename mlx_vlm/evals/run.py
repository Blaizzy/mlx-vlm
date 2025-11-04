import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Define available benchmarks and their default configurations
BENCHMARKS = {
    "mmstar": {
        "script": "mlx_vlm/evals/mmstar.py",
        "dataset": "Lin-Chen/MMStar",
        "split": "val",
        "default_output": "results/mmstar",
    },
    "mmmu": {
        "script": "mlx_vlm/evals/mmmu.py",
        "dataset": "MMMU/MMMU",
        "split": "validation",
        "default_output": "results/mmmu",
    },
    "mathvista": {
        "script": "mlx_vlm/evals/math_vista.py",
        "dataset": "AI4Math/MathVista",
        "split": "testmini",
        "default_output": "results/mathvista",
    },
    "ocrbench": {
        "script": "mlx_vlm/evals/ocrbench.py",
        "dataset": "echo840/OCRBench",
        "split": "test",
        "default_output": "results/ocrbench",
    },
}


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_benchmark(
    benchmark_name: str,
    model: str,
    adapter_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    max_tokens: int = 3000,
    temperature: float = 0.0,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    extra_args: Optional[List[str]] = None,
) -> int:
    """
    Run a single benchmark evaluation.

    Args:
        benchmark_name: Name of the benchmark to run
        model: Model path or name
        adapter_path: Optional adapter path
        max_samples: Maximum samples to evaluate
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        output_dir: Custom output directory
        verbose: Enable verbose output
        extra_args: Additional arguments to pass to the benchmark script

    Returns:
        Return code from the benchmark script
    """
    if benchmark_name not in BENCHMARKS:
        logging.error(f"Unknown benchmark: {benchmark_name}")
        logging.error(f"Available benchmarks: {', '.join(BENCHMARKS.keys())}")
        return 1

    config = BENCHMARKS[benchmark_name]
    script_path = config["script"]

    if not Path(script_path).exists():
        logging.error(f"Benchmark script not found: {script_path}")
        return 1

    # Build command
    cmd = [
        sys.executable,
        "-m",
        (
            f"mlx_vlm.evals.{benchmark_name}"
            if benchmark_name != "mathvista"
            else "mlx_vlm.evals.math_vista"
        ),
        "--model",
        model,
        "--dataset",
        config["dataset"],
        "--split",
        config["split"],
    ]

    # Add optional arguments
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    cmd.extend(["--max-tokens", str(max_tokens)])
    cmd.extend(["--temperature", str(temperature)])

    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    else:
        cmd.extend(["--output-dir", config["default_output"]])

    if verbose:
        cmd.append("--verbose")

    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)

    # Log the command
    logging.info(f"\n{'='*80}")
    logging.info(f"Running {benchmark_name.upper()} Benchmark")
    logging.info(f"{'='*80}")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info(f"{'='*80}\n")

    # Run the benchmark
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logging.error(f"Error running {benchmark_name}: {e}")
        return 1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for MLX-VLM benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Benchmarks:
  mmstar      - MMStar (Multi-Modal Star Benchmark)
  mmmu        - MMMU (Massive Multi-discipline Multimodal Understanding)
  mathvista   - MathVista (Mathematical Reasoning)
  ocrbench    - OCRBench (OCR Benchmark)
  all         - Run all benchmarks

Examples:
  # Run all benchmarks
  python evaluate.py --model mlx-community/Qwen2-VL-2B-Instruct-bf16 --benchmarks all

  # Run specific benchmarks
  python evaluate.py --model mlx-community/Qwen2-VL-2B-Instruct-bf16 --benchmarks mmstar mathvista

  # Run with custom settings
  python evaluate.py --model my-model --benchmarks all --max-samples 100 --temperature 0.7

  # Run with adapter
  python evaluate.py --model base-model --adapter-path ./adapters --benchmarks mmmu
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or Hugging Face model name",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(BENCHMARKS.keys()) + ["all"],
        help="Benchmarks to run (default: all)",
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path for trained adapter weights",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per benchmark (for debugging)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum number of tokens to generate (default: 3000)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0 for greedy)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for all results (default: results/<benchmark_name>)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other benchmarks if one fails",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    # Determine which benchmarks to run
    if "all" in args.benchmarks:
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        benchmarks_to_run = args.benchmarks

    logging.info(f"\n{'='*80}")
    logging.info("MLX-VLM Benchmark Evaluation Suite")
    logging.info(f"{'='*80}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Benchmarks: {', '.join(benchmarks_to_run)}")
    if args.adapter_path:
        logging.info(f"Adapter: {args.adapter_path}")
    if args.max_samples:
        logging.info(f"Max Samples: {args.max_samples}")
    logging.info(f"Max Tokens: {args.max_tokens}")
    logging.info(f"Temperature: {args.temperature}")
    logging.info(f"{'='*80}\n")

    # Track results
    results = {}
    failed_benchmarks = []

    # Run each benchmark
    for i, benchmark_name in enumerate(benchmarks_to_run, 1):
        logging.info(f"\n[{i}/{len(benchmarks_to_run)}] Starting {benchmark_name}...")

        # Determine output directory
        if args.output_dir:
            output_dir = f"{args.output_dir}/{benchmark_name}"
        else:
            output_dir = BENCHMARKS[benchmark_name]["default_output"]

        # Run the benchmark
        return_code = run_benchmark(
            benchmark_name=benchmark_name,
            model=args.model,
            adapter_path=args.adapter_path,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_dir=output_dir,
            verbose=args.verbose,
        )

        results[benchmark_name] = return_code

        if return_code != 0:
            failed_benchmarks.append(benchmark_name)
            logging.error(f"✗ {benchmark_name} failed with return code {return_code}")

            if not args.continue_on_error:
                logging.error(
                    "Stopping due to error. Use --continue-on-error to continue."
                )
                break
        else:
            logging.info(f"✓ {benchmark_name} completed successfully")

    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("Evaluation Summary")
    logging.info(f"{'='*80}")

    successful = [name for name, code in results.items() if code == 0]
    failed = [name for name, code in results.items() if code != 0]

    logging.info(f"Total benchmarks: {len(results)}")
    logging.info(f"Successful: {len(successful)}")
    logging.info(f"Failed: {len(failed)}")

    if successful:
        logging.info(f"\n✓ Successful benchmarks:")
        for name in successful:
            output_dir = args.output_dir or BENCHMARKS[name]["default_output"]
            logging.info(f"  - {name} (results in {output_dir})")

    if failed:
        logging.error(f"\n✗ Failed benchmarks:")
        for name in failed:
            logging.error(f"  - {name} (return code: {results[name]})")

    logging.info(f"{'='*80}\n")

    # Return overall success/failure
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
