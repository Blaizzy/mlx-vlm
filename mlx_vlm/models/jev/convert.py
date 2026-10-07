"""Prepare a JEV decision checkpoint from an MLX Qwen3.5/3.8 base and a JEV adapter.

JEV models (autotrust/JEV-9B, JEV-27B, JEV-27B-VL) ship the unchanged base model
plus a System 1 LoRA. The adapter's ``lm_head`` LoRA changes only the rows of the
decision tokens, so those rows, with the bias from ``decision_head.json``, form a
small float32 decision head. This script copies the base checkpoint, adds the
backbone LoRA unmerged (the base model stays System 2) and the decision head, and
records the decision settings in ``config.json``.

Usage:
    python -m mlx_vlm.models.jev.convert \\
        --model mlx-community/Qwen3.8-27B-4bit \\
        --adapter autotrust/JEV-27B-VL \\
        --mlx-path JEV-27B-VL-4bit

The decision rows are read from the base ``lm_head``, dequantized when the base
is quantized; the rest of the base is copied unchanged.
"""

import argparse
import json
import math
import re
import shutil
import string
from pathlib import Path

import mlx.core as mx

from ...utils import get_model_path

LORA_KEY = re.compile(
    r"(?:^|\.)model\.(?:language_model\.)?layers\.(\d+)\."
    r"(self_attn|linear_attn|mlp)\.(\w+)\.lora_([AB])\.weight$"
)
LM_HEAD_KEYS = ("language_model.lm_head", "lm_head")
MAX_CHOICES = 256


def choice_labels(tokenizer, count=MAX_CHOICES):
    """Single-token option labels A-Z, then AA, AB, ..., as JEV's server reads them."""
    labels = []
    for label in list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
    ]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        context = tokenizer.encode(f"x\n{label}) y", add_special_tokens=False)
        if len(ids) == 1 and ids[0] != tokenizer.unk_token_id and ids[0] in context:
            labels.append((label, ids[0]))
        if len(labels) == count:
            break
    return labels


def adapter_lora(weights, alpha, rank, rslora=False):
    """Map PEFT LoRA tensors to MLX names and drop zero rank padding."""
    lora, used, unknown = {}, 0, []
    for key, value in weights.items():
        match = LORA_KEY.search(key)
        if match is None:
            if ".lora_" in key and ".lm_head." not in key:
                unknown.append(key)
            continue
        layer, block, name, side = match.groups()
        prefix = f"language_model.model.layers.{layer}.{block}.{name}"
        value = value.T  # A: (r, in) -> (in, r); B: (out, r) -> (r, out)
        axis = 1 if side == "A" else 0
        nonzero = mx.any(value != 0, axis=1 - axis)
        live = [i for i, flag in enumerate(nonzero.tolist()) if flag]
        used = max(used, live[-1] + 1 if live else 0)
        lora[f"{prefix}.lora_{side.lower()}"] = value
    if unknown:
        raise ValueError(
            f"Unsupported LoRA tensors outside the decoder layers: {unknown[:3]}"
        )
    for key, value in lora.items():
        lora[key] = value[:, :used] if key.endswith("lora_a") else value[:used]
    return lora, used, alpha / (math.sqrt(rank) if rslora else rank)


def lm_head_rows(path, ids):
    """Rows of the lm_head weight for token ``ids``, dequantized if needed."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        weight_map = {
            key: shard.name
            for shard in path.glob("*.safetensors")
            for key in mx.load(str(shard))
        }
    name = next((n for n in LM_HEAD_KEYS if f"{n}.weight" in weight_map), None)
    if name is None:
        raise ValueError(f"No lm_head weight found in {path}")
    tensors = mx.load(str(path / weight_map[f"{name}.weight"]))
    rows = mx.array(ids)
    weight = tensors[f"{name}.weight"][rows]
    if f"{name}.scales" in weight_map:
        config = json.loads((path / "config.json").read_text())
        quantization = config.get("quantization") or config["quantization_config"]
        quantization = quantization.get(name) or quantization
        if f"{name}.scales" not in tensors:
            tensors.update(mx.load(str(path / weight_map[f"{name}.scales"])))
        biases = tensors.get(f"{name}.biases")
        weight = mx.dequantize(
            weight,
            tensors[f"{name}.scales"][rows],
            None if biases is None else biases[rows],
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
        )
    return weight.astype(mx.float32)


def convert(model, adapter, mlx_path, adapter_subfolder="adapter_vllm"):
    from transformers import AutoTokenizer

    base = get_model_path(model)
    root = get_model_path(
        adapter,
        allow_patterns=[
            f"{adapter_subfolder}/*",
            str(Path(adapter_subfolder).parent / "calibration.json"),
        ],
    )
    folder = root / adapter_subfolder
    peft = json.loads((folder / "adapter_config.json").read_text())
    head = json.loads((folder / "decision_head.json").read_text())
    calibration = json.loads((folder.parent / "calibration.json").read_text())
    weights = mx.load(str(folder / "adapter_model.safetensors"))
    if peft.get("rank_pattern") or peft.get("alpha_pattern"):
        raise ValueError("Per-module LoRA ranks or alphas are not supported")

    lora, rank, scale = adapter_lora(
        weights, peft["lora_alpha"], peft["r"], peft.get("use_rslora", False)
    )
    if not lora:
        raise ValueError(f"No backbone LoRA tensors found in {folder}")

    slots = head["slots"]["ranges"]
    start, end = slots["choice"]
    if end != len(head["verbalizer_ids"]):
        raise ValueError("Expected the choice slots to be the last decision slots")
    labels = choice_labels(AutoTokenizer.from_pretrained(base))
    if [i for _, i in labels[: end - start]] != head["verbalizer_ids"][start:end]:
        raise ValueError("The tokenizer's first choice labels differ from the head")
    ids = head["verbalizer_ids"] + [i for _, i in labels[end - start :]]

    rows = lm_head_rows(base, ids)
    lm_a = weights["base_model.model.lm_head.lora_A.weight"].astype(mx.float32)
    lm_b = weights["base_model.model.lm_head.lora_B.weight"].astype(mx.float32)
    rows = rows + scale * (lm_b[mx.array(ids)] @ lm_a)
    bias = mx.array(head["bias"] + [0.0] * (len(ids) - len(head["bias"])))

    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.is_file() and item.name != "README.md":
            shutil.copy2(item, out / item.name)

    extra = {
        **lora,
        "decision_head.weight": rows,
        "decision_head.bias": bias.astype(mx.float32),
    }
    mx.save_safetensors(str(out / "jev.safetensors"), extra, metadata={"format": "mlx"})
    index = out / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
    else:
        data = {
            "metadata": {},
            "weight_map": {
                key: shard.name
                for shard in out.glob("model*.safetensors")
                for key in mx.load(str(shard))
            },
        }
    data["weight_map"].update(dict.fromkeys(extra, "jev.safetensors"))
    index.write_text(json.dumps(data, indent=4))

    config = json.loads((base / "config.json").read_text())
    config["model_type"] = "jev"
    config["decision_config"] = {
        "protocol": "jev27-bare-v1",
        "template_version": head["slots"].get("template_version", "bare-v1"),
        "slots": slots,
        "choice_labels": [label for label, _ in labels],
        "temperature_by_type": {
            kind: calibration["per_kind"].get(kind, 1.0)
            for kind in ("noul", "choice", "score")
        },
        "adapter": {
            "rank": rank,
            "scale": scale,
            "target_modules": sorted(
                {key.split(".")[-2] for key in lora if key.endswith("lora_a")}
            ),
        },
    }
    (out / "config.json").write_text(json.dumps(config, indent=4))
    print(
        f"Wrote {out}: LoRA rank {rank} on {len(lora) // 2} projections, "
        f"{len(ids)}-row decision head"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="MLX Qwen3.5/3.8 base")
    parser.add_argument("--adapter", required=True, help="JEV repository or folder")
    parser.add_argument("--mlx-path", required=True, help="Output directory")
    parser.add_argument(
        "--adapter-subfolder",
        default="adapter_vllm",
        help="Adapter folder inside --adapter (vl/adapter_vllm for JEV-9B images)",
    )
    args = parser.parse_args()
    convert(args.model, args.adapter, args.mlx_path, args.adapter_subfolder)


if __name__ == "__main__":
    main()
