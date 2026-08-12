from __future__ import annotations

import argparse

from .download import partition_for_workflow, resolve_model_path
from .weights import convert_minimax_h3


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one official MiniMax-H3 workflow to MLX."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Official local checkpoint or Hugging Face repository ID",
    )
    parser.add_argument("--output", required=True, help="Converted MLX directory")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--workflow",
        choices=("t2va", "fl2va", "ref2va"),
        help="Select one Diffusers workflow and its required transformer",
    )
    selection.add_argument(
        "--partition",
        choices=("fl2va", "ref2va"),
        help="Select a task partition (legacy alias for --workflow)",
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
        help="Hub revision to download and record in the conversion manifest",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    workflow = args.workflow or args.partition
    partition = partition_for_workflow(workflow)
    source = resolve_model_path(
        args.source,
        workflow=workflow,
        revision=args.source_revision,
    )
    report = convert_minimax_h3(
        source,
        args.output,
        partition=partition,
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
