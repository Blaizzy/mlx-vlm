import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path

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


def _safetensor_files(model_path: Path) -> List[Path]:
    return [
        Path(path)
        for path in glob.glob(str(model_path / "*.safetensors"))
        if not path.endswith("consolidated.safetensors")
    ]


def _weight_map(model_path: Path) -> Dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        data = json.load(f)
    return data.get("weight_map", {})


def _iter_nextn_keys(
    model_path: Path, nextn_prefix: str
) -> Iterable[tuple[Path, List[str]]]:
    """Yield (shard, keys) for the shards that hold the nextn layer tensors.

    Uses the weight index when present (so only the handful of shards with
    nextn tensors are touched), otherwise scans the local safetensors files.
    """
    weight_map = _weight_map(model_path)
    if weight_map:
        by_file: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith(nextn_prefix):
                by_file.setdefault(filename, []).append(key)
        if by_file:
            for filename, keys in by_file.items():
                yield model_path / filename, keys
            return

    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            keys = [key for key in f.keys() if key.startswith(nextn_prefix)]
        if keys:
            yield file, keys


def _load_selected_tensors(file: Path, keys: List[str]) -> Dict[str, mx.array]:
    tensors = {}
    try:
        with safe_open(file, framework="mlx") as f:
            for key in keys:
                tensors[key] = mx.array(f.get_tensor(key))
    except (AttributeError, RuntimeError, TypeError):
        shard = mx.load(str(file))
        tensors = {key: shard[key] for key in keys}
    return tensors


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


def _quantize(
    weights: Dict[str, mx.array], bits: int, group_size: int
) -> Optional[dict]:
    """Affine-quantize the projection weights in place, matching mlx-lm convert.

    Skips the router gate (``mlp.gate.weight`` stays full precision for routing
    stability), norms, and the fp32 correction bias. Returns the quantization
    config to record, or ``None`` if nothing was quantized.
    """
    quantized_any = False
    for key in list(weights):
        if not key.endswith(".weight") or key.endswith("mlp.gate.weight"):
            continue
        weight = weights[key]
        if weight.ndim < 2 or weight.shape[-1] % group_size != 0:
            continue
        wq, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        weights[key] = wq
        weights[key[: -len(".weight")] + ".scales"] = scales
        weights[key[: -len(".weight")] + ".biases"] = biases
        quantized_any = True

    if not quantized_any:
        return None
    return {"group_size": group_size, "bits": bits, "mode": "affine"}


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
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = dict(source_config.get("text_config") or source_config)

    nextn_prefix = f"model.layers.{int(text_config['num_hidden_layers'])}."
    selected = {}
    for file, keys in _iter_nextn_keys(source_path, nextn_prefix):
        selected.update(_load_selected_tensors(file, keys))

    if not selected:
        raise ValueError(f"No {nextn_prefix}* tensors found in {source_path}.")

    weights = _flatten_nextn_weights(selected, nextn_prefix)
    _split_kv_b_proj(weights, text_config)
    _stack_experts(weights, text_config)

    quantization = None
    if q_bits is not None:
        quantization = _quantize(weights, q_bits, q_group_size)

    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    depth = int(text_config.get("num_nextn_predict_layers", 1))
    text_config.pop("quantization", None)
    text_config.pop("quantization_config", None)
    draft_config = {
        "model_type": "glm4_moe_lite_mtp",
        "text_config": text_config,
        "block_size": int(block_size or depth + 1),
        "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
    }
    if quantization is not None:
        draft_config["quantization"] = quantization
        draft_config["quantization_config"] = quantization

    with open(output_path / "config.json", "w") as f:
        json.dump(dict(sorted(draft_config.items())), f, indent=2)

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        src = source_path / name
        if src.exists():
            shutil.copy(src, output_path / name)

    return output_path


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
