import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import mlx.core as mx

from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config
from ..mtp_split import MTPSplitter
from .deepseek_v4_dspark import DeepseekV4DsparkDraftModel


def _quantization_from_weights(weights: Dict[str, mx.array]) -> Optional[dict]:
    """Per-module quantization config for the sanitized DSpark layout."""
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {"group_size": 64, "bits": 8, "mode": "affine"}

    for key in weights:
        if not key.endswith(".scales"):
            continue
        module = key[: -len(".scales")]
        if "ffn.switch_mlp." in module and module.endswith("_proj"):
            quantization[module] = mxfp4
        elif (
            module.endswith("main_proj")
            or "ffn.shared_experts." in module
            or ".attn.w" in module
        ):
            quantization[module] = mxfp8

    return quantization if len(quantization) > 3 else None


class DeepseekV4DsparkSplitter(MTPSplitter):
    """Extract DeepSeek-V4's native DSpark head (``mtp.<stage>.*``, stages 0..N)
    into a standalone ``deepseek_v4_dspark`` drafter."""

    output_model_type = "deepseek_v4_dspark"
    draft_model_cls = DeepseekV4DsparkDraftModel
    require_text_config = False
    tie_word_embeddings_default = False
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    )

    def __init__(self):
        self._n_mtp_layers = 1

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith("mtp.")

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(args=DeepseekV4Config.from_dict(text_config))

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        stages = {
            int(match.group(1))
            for key in tensors
            if (match := re.match(r"stages\.(\d+)\.", key))
        }
        self._n_mtp_layers = (max(stages) + 1) if stages else 1

    def quantization_from_source(self, tensors, source_config):
        return _quantization_from_weights(tensors)

    def extra_config(self, text_config: dict) -> dict:
        dspark_block = int(text_config.get("dspark_block_size", 0) or 0)
        return {
            "n_mtp_layers": self._n_mtp_layers,
            "target_layer_ids": list(
                text_config.get("dspark_target_layer_ids", []) or []
            ),
            "mask_token_id": int(text_config.get("dspark_noise_token_id", 0) or 0),
            "markov_rank": int(text_config.get("dspark_markov_rank", 256) or 256),
            "block_size": (dspark_block + 1) if dspark_block else 0,
        }


def split_deepseek_v4_dspark(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write DeepSeek-V4 DSpark MTP tensors into a standalone drafter folder."""
    return DeepseekV4DsparkSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-V4 DSpark MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v4_dspark(**vars(args))
    print(f"Wrote DeepSeek-V4 DSpark drafter to {output}")


if __name__ == "__main__":
    main()
