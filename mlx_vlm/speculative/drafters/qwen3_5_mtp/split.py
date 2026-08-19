import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ....models.qwen3_5.fp8 import make_quantization_config
from ..mtp_split import MTPSplitter
from .qwen3_5_mtp import Qwen3_5MTPDraftModel

# top-level ``mtp.*`` norms that follow the zero-centered (weight + 1.0) RMSNorm
# convention shared by Qwen3.5 and Qwen3-Next
_QWEN_MTP_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


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

    def quantization_from_source(self, tensors, source_config):
        if not any(key.endswith(".scales") for key in tensors):
            return None
        quantization = source_config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = source_config.get("quantization")
        if quantization is None:
            quantization = make_quantization_config(source_config)
        return quantization


class Qwen3NextMTPSplitter(MTPSplitter):
    # Qwen3-Next ships the same top-level ``mtp.*`` block as Qwen3.5 and reuses
    # the qwen3_5_mtp drafter runtime, but its MoE stores separate up/down/gate
    # experts (not Qwen3.5's fused ``gate_up_proj``), so expert stacking is bespoke.
    output_model_type = "qwen3_5_mtp"
    draft_model_cls = None
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

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        out: Dict[str, mx.array] = {}
        for key, value in tensors.items():
            new_key = key[len("mtp.") :] if key.startswith("mtp.") else key
            if key.startswith("mtp.") and any(
                new_key.endswith(sfx) for sfx in _QWEN_MTP_NORM_SUFFIXES
            ):
                if value.ndim == 1 and mx.issubdtype(value.dtype, mx.floating):
                    value = value + 1.0
            out[new_key] = value
        return out

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        n_experts = int(text_config.get("num_experts", 0) or 0)
        if not n_experts:
            return
        pattern = re.compile(
            r"(.*\.experts)\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
        )
        prefixes = {m.group(1) for k in tensors if (m := pattern.match(k))}
        for prefix in prefixes:
            base = prefix[: -len(".experts")]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                keys = [f"{prefix}.{e}.{proj}.weight" for e in range(n_experts)]
                if all(k in tensors for k in keys):
                    tensors[f"{base}.switch_mlp.{proj}.weight"] = mx.stack(
                        [tensors.pop(k) for k in keys]
                    )

    def quantization_from_source(self, tensors, source_config):
        if not any(key.endswith(".scales") for key in tensors):
            return None
        quantization = source_config.get("mtplx_mtp_quantization")
        if quantization is None:
            quantization = source_config.get("quantization")
        if quantization is None:
            quantization = make_quantization_config(source_config)
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
