import argparse
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter
from .inkling_mtp import InklingMTPDraftModel

_MTP_PREFIX = "model.mtp."
_NORM_KEY = "model.llm.norm.weight"


class InklingMTPSplitter(MTPSplitter):
    output_model_type = "inkling_mtp"
    draft_model_cls = InklingMTPDraftModel
    tie_word_embeddings_default = False
    block_size_extra = 2
    supports_mlx_source = True

    def read_text_config(self, source_config: dict) -> dict:
        text_config = dict(source_config.get("text_config") or {})
        if not text_config:
            raise ValueError("source config does not contain a text_config.")
        mtp_config = source_config.get("mtp_config") or {}
        text_config.setdefault(
            "num_mtp_layers",
            mtp_config.get("num_nextn_predict_layers")
            or text_config.get("num_nextn_predict_layers"),
        )
        text_config.setdefault(
            "mtp_local_layer_ids",
            mtp_config.get("local_layer_ids") or text_config.get("local_layer_ids"),
        )
        return text_config

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith(_MTP_PREFIX) or key == _NORM_KEY

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        for key, value in tensors.items():
            if key.startswith(_MTP_PREFIX):
                out[key[len(_MTP_PREFIX) :]] = value
            elif key == _NORM_KEY:
                out["norm.weight"] = value
            else:
                out[key] = value
        return out

    def depth(self, text_config: dict) -> int:
        return int(
            text_config.get("num_mtp_layers")
            or text_config.get("num_nextn_predict_layers")
            or 1
        )

    def extra_config(self, text_config: dict) -> dict:
        return {
            "num_mtp_layers": self.depth(text_config),
            "mtp_local_layer_ids": text_config.get("mtp_local_layer_ids"),
        }


def split_inkling_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write Inkling native MTP tensors into a standalone drafter folder."""
    return InklingMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Inkling native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_inkling_mtp(**vars(args))
    print(f"Wrote Inkling MTP drafter to {output}")


if __name__ == "__main__":
    main()
