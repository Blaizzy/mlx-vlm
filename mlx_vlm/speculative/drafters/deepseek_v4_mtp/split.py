import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.deepseek_v4.config import ModelConfig as DeepseekV4Config
from ....utils import _load_safetensors
from ..mtp_split import MTPSplitter
from .deepseek_v4_mtp import DeepseekV4MTPDraftModel


def _module_from_scales_key(key: str) -> str:
    return key[: -len(".scales")]


def _quantization_from_weights(weights: Dict[str, mx.array]) -> Optional[dict]:
    mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
    quantization = {"group_size": 64, "bits": 8, "mode": "affine"}

    for key in weights:
        if not key.endswith(".scales"):
            continue
        module = _module_from_scales_key(key)
        if "decoder.ffn.switch_mlp." in module and module.endswith("_proj"):
            quantization[module] = mxfp4
        elif (
            module in ("e_proj", "h_proj")
            or "decoder.ffn.shared_experts." in module
            or "decoder.attn.w" in module
        ):
            quantization[module] = mxfp8

    return quantization if len(quantization) > 3 else None


class DeepseekV4MTPSplitter(MTPSplitter):
    output_model_type = "deepseek_v4_mtp"
    draft_model_cls = DeepseekV4MTPDraftModel
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

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith("mtp.")

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        try:
            with safe_open(file, framework="mlx") as f:
                return {key: mx.array(f.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            shard = _load_safetensors(str(file))
            return {key: shard[key] for key in keys}

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(args=DeepseekV4Config.from_dict(text_config))

    def quantization_from_source(self, tensors, source_config):
        return _quantization_from_weights(tensors)


def split_deepseek_v4_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write DeepSeek-V4 native MTP tensors into a standalone drafter folder."""
    return DeepseekV4MTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split DeepSeek-V4 native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v4_mtp(**vars(args))
    print(f"Wrote DeepSeek-V4 MTP drafter to {output}")


if __name__ == "__main__":
    main()
