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

    @staticmethod
    def _dequantize_fine_grained_fp8(
        tensors: Dict[str, mx.array], block_size: int = 128
    ) -> Dict[str, mx.array]:
        scale_keys = [key for key in tensors if key.endswith("weight_scale_inv")]
        if not scale_keys:
            return tensors

        converted = dict(tensors)
        for scale_key in scale_keys:
            weight_key = scale_key[: -len("_scale_inv")]
            if weight_key not in converted:
                raise ValueError(f"Missing FP8 weight for scale tensor {scale_key!r}.")
            weight = converted.pop(weight_key)
            scale_inv = converted.pop(scale_key)
            if weight.dtype != mx.uint8 or weight.ndim < 2:
                raise ValueError(
                    "GLM fine-grained FP8 weights must be uint8 matrices; "
                    f"got dtype={weight.dtype}, shape={weight.shape}."
                )

            *batch_shape, rows, cols = weight.shape
            expected_scale_shape = (
                *batch_shape,
                (rows + block_size - 1) // block_size,
                (cols + block_size - 1) // block_size,
            )
            if scale_inv.shape != expected_scale_shape:
                raise ValueError(
                    "GLM fine-grained FP8 scale shape does not match its weight: "
                    f"weight={weight.shape}, scales={scale_inv.shape}, "
                    f"expected={expected_scale_shape}."
                )

            pad_rows = (-rows) % block_size
            pad_cols = (-cols) % block_size
            decoded = mx.from_fp8(weight, dtype=mx.bfloat16)
            if pad_rows or pad_cols:
                decoded = mx.pad(
                    decoded,
                    [(0, 0)] * len(batch_shape) + [(0, pad_rows), (0, pad_cols)],
                )
            decoded = decoded.reshape(
                *batch_shape,
                (rows + pad_rows) // block_size,
                block_size,
                (cols + pad_cols) // block_size,
                block_size,
            )
            scales = scale_inv.reshape(
                *batch_shape,
                (rows + pad_rows) // block_size,
                1,
                (cols + pad_cols) // block_size,
                1,
            )
            converted[weight_key] = (decoded * scales).reshape(
                *batch_shape, rows + pad_rows, cols + pad_cols
            )[..., :rows, :cols]
        return converted

    def run_sanitize(
        self, tensors: Dict[str, mx.array], text_config: dict
    ) -> Dict[str, mx.array]:
        tensors = self._dequantize_fine_grained_fp8(tensors)
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
