import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from ..mtp_split import MTPSplitter


class Glm5NextMTPSplitter(MTPSplitter):
    output_model_type = "glm5_next_mtp"
    draft_model_cls = None
    tie_word_embeddings_default = False
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1
    supports_mlx_source = False

    def _layer_idx(self, text_config: dict) -> int:
        return int(text_config["num_hidden_layers"])

    def select_keys(self, key: str, text_config: dict) -> bool:
        return f"layers.{self._layer_idx(text_config)}." in key

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        marker = f"layers.{self._layer_idx(text_config)}."
        out: Dict[str, mx.array] = {}
        for key, value in tensors.items():
            rest = key[key.index(marker) + len(marker) :]
            if rest.startswith("shared_head.norm."):
                rest = "shared_head_norm." + rest[len("shared_head.norm.") :]
            out[rest] = value

        kb = "self_attn.kv_b_proj.weight"
        if kb in out:
            v = out.pop(kb)
            nope = int(text_config["qk_nope_head_dim"])
            vhd = int(text_config["v_head_dim"])
            nheads = int(text_config["num_attention_heads"])
            v = v.reshape(nheads, nope + vhd, -1)
            out["self_attn.embed_q.weight"] = mx.contiguous(
                v[:, :nope, :].swapaxes(-1, -2)
            )
            out["self_attn.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])

        return {f"mtp.{key}": value for key, value in out.items()}

    def postprocess(self, tensors: Dict[str, mx.array], text_config: dict) -> None:
        n_experts = int(
            text_config.get("n_routed_experts", text_config.get("num_experts", 0)) or 0
        )
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
        return source_config.get("quantization") or source_config.get(
            "quantization_config"
        )


def split_glm5_next_mtp(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    block_size: Optional[int] = None,
    force_download: bool = False,
) -> Path:
    """Write the GLM-5-Next native nextn layer into a standalone drafter folder."""
    return Glm5NextMTPSplitter().split(
        source,
        output,
        revision=revision,
        block_size=block_size,
        force_download=force_download,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split the GLM-5-Next nextn layer into a standalone MLX drafter."
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
