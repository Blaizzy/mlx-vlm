import argparse
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter
from .qwen3_5_mtp import Qwen3_5MTPDraftModel


class Qwen3_5MTPSplitter(MTPSplitter):
    output_model_type = "qwen3_5_mtp"
    draft_model_cls = Qwen3_5MTPDraftModel
    tie_word_embeddings_default = True
    depth_field = "mtp_num_hidden_layers"
    block_size_extra = 2
    supports_mlx_source = True

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith("mtp.")

    def on_mlx_source(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return {
            (key[len("mtp.") :] if key.startswith("mtp.") else key): value
            for key, value in tensors.items()
        }

    def quantization(self, tensors, source_config, text_config, quant_opts):
        if not any(key.endswith(".scales") for key in tensors):
            return None
        quantization = source_config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = source_config.get("quantization")
        return quantization


def split_qwen3_5_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write Qwen3.5 native MTP tensors into a standalone drafter folder."""
    return Qwen3_5MTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Qwen3.5 native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_qwen3_5_mtp(**vars(args))
    print(f"Wrote Qwen3.5 MTP drafter to {output}")


if __name__ == "__main__":
    main()
