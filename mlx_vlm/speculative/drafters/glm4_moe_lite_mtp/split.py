import argparse
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter

# GLM-4.7-Flash ships one trained nextn (MTP) layer as
# ``model.layers.<num_hidden_layers>.*`` (a dedicated ``embed_tokens``, the
# ``enorm`` / ``hnorm`` / ``eh_proj`` projections, an MLA ``self_attn``, a
# 64-expert MoE + shared expert, and an untied ``shared_head``). It is split
# into the flat layout the drafter loads:
#
#     model.embed_tokens.weight        <- model.layers.<N>.embed_tokens.weight
#     model.{enorm,hnorm,eh_proj}      <- model.layers.<N>.{enorm,hnorm,eh_proj}
#     model.mtp_block.*                <- model.layers.<N>.{input_layernorm,
#                                          post_attention_layernorm,self_attn.*,mlp.*}
#     model.shared_head_norm.weight    <- model.layers.<N>.shared_head.norm.weight
#     lm_head.weight                   <- model.layers.<N>.shared_head.head.weight
_TOP_LEVEL = ("enorm.weight", "hnorm.weight", "eh_proj.weight")
_SHARED_HEAD_NORM = "shared_head.norm.weight"
_SHARED_HEAD_HEAD = "shared_head.head.weight"

_MTP_ATTN_PREFIX = "model.mtp_block.self_attn"
_MTP_MLP_PREFIX = "model.mtp_block.mlp"

# Registered attention buffers that are recomputed at build time; they have no
# home in the flat layout, so a strict load would reject them if kept.
_SKIP_BUFFER_SUFFIXES = ("rotary_emb.inv_freq",)


def _is_nonparameter_buffer(rest: str) -> bool:
    return any(rest.endswith(suffix) for suffix in _SKIP_BUFFER_SUFFIXES)


def _flat_name(rest: str) -> str:
    """Map a ``model.layers.<N>.`` suffix to the flat drafter layout."""
    if rest == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if rest in _TOP_LEVEL:
        return f"model.{rest}"
    if rest == _SHARED_HEAD_NORM:
        return "model.shared_head_norm.weight"
    if rest == _SHARED_HEAD_HEAD:
        return "lm_head.weight"
    return f"model.mtp_block.{rest}"


def _flatten_nextn_weights(
    selected: Dict[str, mx.array], nextn_prefix: str
) -> Dict[str, mx.array]:
    flat: Dict[str, mx.array] = {}
    for key, tensor in selected.items():
        rest = key[len(nextn_prefix) :]
        if _is_nonparameter_buffer(rest):
            continue
        flat_key = _flat_name(rest)
        # The noaux_tc router correction bias is fp32 in the source shard and
        # must stay fp32 (casting it breaks routing); everything else is bf16.
        if "e_score_correction_bias" in flat_key:
            flat[flat_key] = tensor
        else:
            flat[flat_key] = tensor.astype(mx.bfloat16)
    return flat


def _split_kv_b_proj(weights: Dict[str, mx.array], text_config: dict) -> None:
    """Split the fused MLA ``kv_b_proj`` into absorbed ``embed_q`` / ``unembed_out``.

    No-op when the weights are already in the absorbed layout.
    """
    weight_key = f"{_MTP_ATTN_PREFIX}.kv_b_proj.weight"
    if weight_key not in weights:
        return

    num_heads = int(text_config["num_attention_heads"])
    qk_nope = int(text_config["qk_nope_head_dim"])
    head_dim = qk_nope + int(text_config["v_head_dim"])

    v = weights.pop(weight_key).reshape(num_heads, head_dim, -1)
    weights[f"{_MTP_ATTN_PREFIX}.embed_q.weight"] = mx.contiguous(
        v[:, :qk_nope, :].swapaxes(-1, -2)
    )
    weights[f"{_MTP_ATTN_PREFIX}.unembed_out.weight"] = mx.contiguous(v[:, qk_nope:, :])


def _stack_experts(weights: Dict[str, mx.array], text_config: dict) -> None:
    """Stack per-expert MoE tensors into the ``switch_mlp`` layout."""
    n_experts = int(text_config["n_routed_experts"])
    for proj in ("gate_proj", "down_proj", "up_proj"):
        first = f"{_MTP_MLP_PREFIX}.experts.0.{proj}.weight"
        if first not in weights:
            continue
        expert_keys = [
            f"{_MTP_MLP_PREFIX}.experts.{e}.{proj}.weight" for e in range(n_experts)
        ]
        missing = [key for key in expert_keys if key not in weights]
        if missing:
            raise ValueError(
                f"nextn checkpoint is missing expert tensors {missing[:3]}: "
                f"config n_routed_experts={n_experts} expects experts "
                f"0..{n_experts - 1} for {proj}."
            )
        weights[f"{_MTP_MLP_PREFIX}.switch_mlp.{proj}.weight"] = mx.stack(
            [weights.pop(key) for key in expert_keys]
        )


class Glm4MoeLiteMTPSplitter(MTPSplitter):
    output_model_type = "glm4_moe_lite_mtp"
    draft_model_cls = None
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

    def read_text_config(self, source_config: dict) -> dict:
        text_config = dict(source_config.get("text_config") or source_config)
        text_config.pop("quantization", None)
        text_config.pop("quantization_config", None)
        return text_config

    def _nextn_prefix(self, text_config: dict) -> str:
        return f"model.layers.{int(text_config['num_hidden_layers'])}."

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.startswith(self._nextn_prefix(text_config))

    def rename(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        return _flatten_nextn_weights(tensors, self._nextn_prefix(text_config))

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        _split_kv_b_proj(tensors, text_config)
        _stack_experts(tensors, text_config)


def split_glm4_moe_lite_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
    q_bits: Optional[int] = None,
    q_group_size: int = 64,
) -> Path:
    """Write GLM-4.7-Flash native MTP tensors into a standalone drafter folder."""
    return Glm4MoeLiteMTPSplitter().split(
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
        description="Split GLM-4.7-Flash native MTP tensors into a standalone MLX drafter."
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
    output = split_glm4_moe_lite_mtp(**vars(args))
    print(f"Wrote GLM-4.7-Flash MTP drafter to {output}")


if __name__ == "__main__":
    main()
