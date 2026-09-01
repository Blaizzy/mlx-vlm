"""Extract a model's native MTP tensors into a standalone MLX drafter.

Dispatches by base ``model_type`` through the ``MTP_SPLITTERS`` registry. With
no ``--model-type`` the base type is detected from the source config and MTP
tensors are confirmed present.
"""

import argparse
from pathlib import Path
from typing import Optional

from .speculative.drafters.mtp_split import (
    MTP_SPLITTERS,
    detect_mtp_splitter,
    get_mtp_splitter,
)
from .utils import get_model_path


def split_mtp(
    source: str,
    output: str,
    *,
    model_type: Optional[str] = None,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: int = 64,
    dequantize: bool = False,
) -> Path:
    if dequantize and q_bits is not None:
        raise ValueError(
            "Choose either dense BF16 MTP extraction or MTP quantization, not both."
        )

    if model_type is not None:
        splitter = get_mtp_splitter(model_type)
        if splitter is None:
            raise ValueError(
                f"No MTP splitter registered for model_type={model_type!r}. "
                f"Known: {sorted(MTP_SPLITTERS)}"
            )
    else:
        source_path = get_model_path(
            source, revision=revision, force_download=force_download
        )
        splitter = detect_mtp_splitter(source_path)
        if splitter is None:
            raise ValueError(
                f"No native MTP tensors / registered splitter for {source!r}. "
                f"Known base model types: {sorted(MTP_SPLITTERS)}. "
                f"Pass --model-type to force a specific splitter."
            )

    return splitter.split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
        q_bits=q_bits,
        q_group_size=q_group_size,
        dequantize=dequantize,
    )


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-type",
        default=None,
        help=f"Force a splitter (default: detect). One of {sorted(MTP_SPLITTERS)}.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--q-bits", type=int, default=None)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--dequantize",
        "--full-precision",
        dest="dequantize",
        action="store_true",
        help=(
            "Reconstruct block-FP8 MTP tensors as dense BF16 instead of "
            "native MLX quantized weights."
        ),
    )
    return parser


def main():
    args = configure_parser().parse_args()
    output = split_mtp(**vars(args))
    print(f"Wrote MTP drafter to {output}")


if __name__ == "__main__":
    main()
