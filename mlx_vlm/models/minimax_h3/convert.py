from __future__ import annotations

import argparse

from .weights import convert_minimax_h3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one local official MiniMax-H3 partition to MLX."
    )
    parser.add_argument("--source", required=True, help="Official local checkpoint")
    parser.add_argument("--output", required=True, help="Converted MLX directory")
    parser.add_argument(
        "--partition",
        required=True,
        choices=("fl2va", "ref2va"),
        help="Select exactly one task transformer",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Omit Qwen vision weights (FL2VA/T2VA conversion only)",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="Cast floating tensors; default preserves source dtypes",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Source revision recorded in the conversion manifest",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = convert_minimax_h3(
        args.source,
        args.output,
        partition=args.partition,
        text_only=args.text_only,
        dtype=args.dtype,
        source_revision=args.source_revision,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(
        f"partition={report.partition} source_bytes={report.source_bytes} "
        f"converted_bytes={report.converted_bytes} dry_run={report.dry_run}"
    )
    if report.tensor_counts:
        for component, count in sorted(report.tensor_counts.items()):
            print(f"{component}: {count} tensors")


if __name__ == "__main__":
    main()
