import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import mlx.core as mx

from ....fp8 import make_quantization_config
from ....models.glm5_next.config import TextConfig
from ..mtp_split import MTPSplitter
from .glm5_next_mtp import Glm5NextMTPDraftModel


class Glm5NextMTPSplitter(MTPSplitter):
    output_model_type = "glm5_next_mtp"
    draft_model_cls = Glm5NextMTPDraftModel
    tie_word_embeddings_default = False
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    )

    @staticmethod
    def _layer_prefixes(text_config: dict):
        layer_idx = int(text_config["num_hidden_layers"])
        return (
            f"model.language_model.layers.{layer_idx}.",
            f"language_model.model.layers.{layer_idx}.",
            f"model.layers.{layer_idx}.",
            f"layers.{layer_idx}.",
        )

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith(self._layer_prefixes(text_config))

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        prefixes = self._layer_prefixes(text_config)
        renamed = {}
        for key, value in tensors.items():
            prefix = next((p for p in prefixes if key.startswith(p)), None)
            if prefix is None:
                continue
            key = key[len(prefix) :]
            if key.startswith(("enorm.", "hnorm.", "eh_proj.")):
                destination = key
            elif key.startswith("shared_head.norm."):
                destination = "shared_head_norm." + key[len("shared_head.norm.") :]
            else:
                destination = "mtp_block." + key
            renamed[destination] = value
        return renamed

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(args=TextConfig.from_dict(text_config))

    def quantization_from_source(self, tensors, source_config):
        if not any(key.endswith(".scales") for key in tensors):
            return None
        quantization = source_config.get("quantization")
        if quantization is None:
            quantization = make_quantization_config(source_config)
        return quantization


def split_glm5_next_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write GLM-5-Next's native layer-45 MTP drafter checkpoint."""
    return Glm5NextMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split GLM-5-Next native MTP tensors into an MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_glm5_next_mtp(**vars(args))
    print(f"Wrote GLM-5-Next MTP drafter to {output}")


if __name__ == "__main__":
    main()
