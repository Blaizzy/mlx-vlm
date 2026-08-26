import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import mlx.core as mx
from safetensors import safe_open

from ....models.qwen4_exp.config import TextConfig as Qwen4ExpTextConfig
from ....utils import _load_safetensors
from ..mtp_split import MTPSplitter
from .qwen4_exp_mtp import Qwen4ExpMTPDraftModel

# Every RMSNorm in the Qwen4-Exp text model stores a zero-centered weight, so it
# scales by ``1 + weight``. The trunk's loader offsets them by suffix; the two MTP
# input-fusion norms end in `norm_embedding` / `norm_hidden` rather than `norm`,
# so they are listed here explicitly.
_ZERO_CENTERED_SUFFIXES = (
    ".pre_fc_norm_embedding.weight",
    ".pre_fc_norm_hidden.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".hc_norm.weight",
)


class Qwen4ExpMTPSplitter(MTPSplitter):
    output_model_type = "qwen4_exp_mtp"
    draft_model_cls = Qwen4ExpMTPDraftModel
    require_text_config = True
    tie_word_embeddings_default = False
    depth_field = "mtp_num_hidden_layers"
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

    def should_quantize_key(self, key: str) -> bool:
        # The trunk's `quant_predicate` carves these out of 4-bit: four output
        # features weighting every sublayer's contribution to every residual
        # stream, and the shared-expert gate. Both are tiny -- together well under
        # a megabyte in bf16 -- and far too sensitive to round to 4 bits, so the
        # drafter keeps them at full precision rather than diverging from the
        # target it has to agree with.
        if key.endswith("block_inject_weight.weight") or key.endswith(
            "shared_expert_gate.weight"
        ):
            return False
        return super().should_quantize_key(key)

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        try:
            with safe_open(file, framework="mlx") as f:
                return {key: mx.array(f.get_tensor(key)) for key in keys}
        except (AttributeError, RuntimeError, TypeError):
            shard = _load_safetensors(str(file))
            return {key: shard[key] for key in keys}

    def rename(self, tensors: Dict[str, mx.array], text_config: dict):
        offset = {}
        for key, value in tensors.items():
            if key.endswith(_ZERO_CENTERED_SUFFIXES) and value.ndim == 1:
                value = value + 1.0
            offset[key] = value
        return offset

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(
            args=Qwen4ExpTextConfig.from_dict(text_config),
            layers=[None],
        )


def split_qwen4_exp_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write Qwen4-Exp native MTP tensors into a standalone drafter folder."""
    return Qwen4ExpMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split Qwen4-Exp native MTP tensors into a standalone MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_qwen4_exp_mtp(**vars(args))
    print(f"Wrote Qwen4-Exp MTP drafter to {output}")


if __name__ == "__main__":
    main()
