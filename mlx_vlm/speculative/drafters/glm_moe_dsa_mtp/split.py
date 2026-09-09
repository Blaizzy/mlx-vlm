import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter


class GlmMoeDsaMTPSplitter(MTPSplitter):
    output_model_type = "glm_moe_dsa_mtp"
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

    def _nextn_prefix(self, text_config: dict) -> str:
        return f"model.layers.{int(text_config['num_hidden_layers'])}."

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith(self._nextn_prefix(text_config))

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        prefix = self._nextn_prefix(text_config)
        renamed = {}
        for key, tensor in tensors.items():
            name = key[len(prefix) :]
            if name == "shared_head.norm.weight":
                name = "shared_head_norm.weight"
            renamed[name] = tensor
        return renamed

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        key = "self_attn.kv_b_proj.weight"
        if key in tensors:
            weight = tensors.pop(key)
            num_heads = int(text_config["num_attention_heads"])
            nope = int(text_config["qk_nope_head_dim"])
            value_dim = int(text_config["v_head_dim"])
            weight = weight.reshape(num_heads, nope + value_dim, -1)
            tensors["self_attn.embed_q.weight"] = mx.contiguous(
                weight[:, :nope, :].swapaxes(-1, -2)
            )
            tensors["self_attn.unembed_out.weight"] = mx.contiguous(weight[:, nope:, :])

        pattern = re.compile(
            r"(.*\.experts)\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
        )
        prefixes = {match.group(1) for key in tensors if (match := pattern.match(key))}
        num_experts = int(text_config["n_routed_experts"])
        for prefix in prefixes:
            mlp_prefix = prefix[: -len(".experts")]
            for projection in ("gate_proj", "up_proj", "down_proj"):
                keys = [
                    f"{prefix}.{expert}.{projection}.weight"
                    for expert in range(num_experts)
                ]
                if all(key in tensors for key in keys):
                    tensors[f"{mlp_prefix}.switch_mlp.{projection}.weight"] = mx.stack(
                        [tensors.pop(key) for key in keys]
                    )

        for key in list(tensors):
            tensors[f"mtp.{key}"] = tensors.pop(key)


def split_glm_moe_dsa_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: int = 64,
) -> Path:
    return GlmMoeDsaMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
        q_bits=q_bits,
        q_group_size=q_group_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split GLM-MoE-DSA native MTP tensors into an MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--q-bits", type=int, default=None)
    parser.add_argument("--q-group-size", type=int, default=64)
    return parser


def main():
    args = build_parser().parse_args()
    print(f"Wrote MTP drafter to {split_glm_moe_dsa_mtp(**vars(args))}")


if __name__ == "__main__":
    main()
