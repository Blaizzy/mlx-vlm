"""Split a DeepSeek-V4-{Flash,Pro}-DSpark checkpoint's ``mtp.*`` draft stack into a
standalone ``deepseek_v4_dspark`` drafter folder.

DSpark extends DeepSeek native MTP, so the draft blocks ship under the ``mtp.*`` namespace
of the base fp8/fp4 checkpoint (alongside the shared ``embed``/``head``). This mirrors
``deepseek_v4_mtp.split`` but, like the reference dspark-mlx loader, **dequantizes the draft
to bf16** rather than keeping it quantized: the ported ``DSparkAttention`` does manual
grouped ``wo_a`` math (no quantized matmul), and the reference itself runs the drafter in
bf16/fp32. The structural ``mtp.N.* -> blocks.N.*`` remap + expert stacking is left to
``DeepseekV4DSparkDraftModel.sanitize`` at load time.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
from safetensors import safe_open

from ....utils import get_model_path
from ..deepseek_v4_mtp.split import (
    _iter_mtp_keys,
    _load_selected_tensors,
    _safetensor_files,
    _text_config,
)


def _load_shared_keys(model_path: Path, keys) -> Dict[str, mx.array]:
    """Find the shared top-level tensors (``embed.weight`` / ``head.weight``) across shards."""
    found: Dict[str, mx.array] = {}
    for file in _safetensor_files(model_path):
        with safe_open(file, framework="mlx") as f:
            present = [k for k in keys if k in f.keys() and k not in found]
        if present:
            found.update(_load_selected_tensors(file, present))
    return found


def _dequant_to_bf16(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Dequantize the fp8(e4m3, 128x128 block) / fp4(e2m1, per-32 group) draft weights to
    bf16, dropping the ``.scale`` siblings. Mirrors the base model's scale handling
    (``DeepseekV4MoE`` quant config: mxfp4 experts, mxfp8 attention/shared) inverted via
    ``mx.dequantize``. Unquantized tensors (norms, hc params, sinks, embed/head) pass through.
    """
    out: Dict[str, mx.array] = {}
    for k, v in weights.items():
        if k.endswith(".scale"):
            continue
        out[k] = v

    for k, v in weights.items():
        if not k.endswith(".scale"):
            continue
        wk = k[: -len(".scale")] + ".weight"
        weight = weights.get(wk)
        if weight is None:
            out[k] = v
            continue
        is_fp4_expert = (
            "ffn.experts." in wk
            and ".shared_experts." not in wk
            and weight.dtype in (mx.int8, mx.uint8)
            and v.shape[-1] * 16 == weight.shape[-1]
        )
        if is_fp4_expert:
            out[wk] = mx.dequantize(
                weight.view(mx.uint32), v, group_size=32, bits=4, mode="mxfp4"
            ).astype(mx.bfloat16)
        elif weight.dtype == mx.uint8:
            scales = mx.repeat(mx.repeat(v, 4, -1), 128, 0)
            out[wk] = mx.dequantize(
                weight.view(mx.uint32), scales, group_size=32, bits=8, mode="mxfp8"
            ).astype(mx.bfloat16)
        else:
            out[wk] = weight  # already dequantized / unquantized
    return out


def _n_mtp_layers(keys) -> int:
    stages = {int(k.split(".")[1]) for k in keys if k.startswith("mtp.")}
    return (max(stages) + 1) if stages else 0


def split_deepseek_v4_dspark(
    source: str,
    output: str,
    *,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Write the DeepSeek-V4 DSpark draft (``mtp.*`` + ``embed``/``head``) to a standalone
    bf16 drafter folder routable as ``model_type='deepseek_v4_dspark'``."""
    source_path = get_model_path(
        source, revision=revision, force_download=force_download
    )
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(source_path / "config.json") as f:
        source_config = json.load(f)
    text_config = _text_config(source_config)

    selected = {}
    for file, keys in _iter_mtp_keys(source_path):
        keys = [
            k for k in keys if not k.endswith("tid2eid")
        ]  # draft blocks score-route
        selected.update(_load_selected_tensors(file, keys))
    if not selected:
        raise ValueError(f"No mtp.* tensors found in {source_path}.")
    # embed/head are shared with the base; pull them from wherever they live.
    selected.update(_load_shared_keys(source_path, ("embed.weight", "head.weight")))

    selected = _dequant_to_bf16(selected)
    n_mtp = _n_mtp_layers(selected)

    mx.save_safetensors(
        str(output_path / "model.safetensors"),
        selected,
        metadata={"format": "mlx"},
    )

    draft_config = {
        k: v
        for k, v in text_config.items()
        if k
        not in ("quantization", "quantization_config", "architectures", "model_type")
    }
    draft_config["model_type"] = "deepseek_v4_dspark"
    if n_mtp:
        draft_config["n_mtp_layers"] = n_mtp
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
        description="Split a DeepSeek-V4-DSpark checkpoint's mtp.* draft into a "
        "standalone bf16 MLX drafter."
    )
    parser.add_argument("--model", "--source", dest="source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    output = split_deepseek_v4_dspark(**vars(args))
    print(f"Wrote DeepSeek-V4 DSpark drafter to {output}")


if __name__ == "__main__":
    main()
