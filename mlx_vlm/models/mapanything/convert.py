"""Convert an official MapAnything checkpoint to an MLX model directory.

The Hub releases (``facebook/map-anything``, ``facebook/map-anything-apache``
and the ``-v1`` variants) ship a float32 ``model.safetensors`` in PyTorch
layout and a ``config.json`` without ``model_type``. This writes
``config.json`` (with ``model_type``), ``model.safetensors`` (MLX layout; the
trunk linears in ``--dtype``, everything else float32) and
``preprocessor_config.json``. No PyTorch is needed.

float16 is the default: it is what the reference's mixed precision picks on
Apple GPUs, runs as fast as bfloat16 and keeps three more mantissa bits (the
largest trunk activation measured was ~450, far below the float16 limit).

Usage:
    python -m mlx_vlm.models.mapanything.convert \\
        --hf-path facebook/map-anything --mlx-path map-anything-fp16
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import mlx.core as mx

from ...utils import get_model_path
from .config import ModelConfig
from .mapanything import Model

_GLOBAL_ENCODERS = (
    "depth_scale_encoder",
    "cam_rot_encoder",
    "cam_trans_encoder",
    "cam_trans_scale_encoder",
)
_RENAMES = [
    (re.compile(r"^encoder\.model\."), "encoder."),
    (
        re.compile(r"^dense_head\.0\.input_process\.(\d)\.0\.0\."),
        r"dense_head.projects.\1.",
    ),
    (
        re.compile(r"^dense_head\.0\.input_process\.(\d)\.0\.1\."),
        r"dense_head.resize_layers.\1.",
    ),
    (
        re.compile(r"^dense_head\.0\.input_process\.(\d)\.1\."),
        lambda m: f"dense_head.scratch.layer{int(m[1]) + 1}_rn.",
    ),
    (re.compile(r"^dense_head\.0\.scratch\."), "dense_head.scratch."),
    (re.compile(r"^dense_head\.1\."), "dense_head."),
    (re.compile(r"^scale_head\.mlp\.(\d+)\.0\."), r"scale_head.mlp.\1."),
]
_GLOBAL_KEY = re.compile(rf"^({'|'.join(_GLOBAL_ENCODERS)})\.encoder\.((?:0\.)*)(\d)\.")
# 1x1 convolutions that run as linears.
_LINEAR_CONVS = re.compile(r"^pose_head\.(proj|res_conv\.\d+\.res_conv\d)\.weight$")
_CONV_TRANSPOSE = re.compile(r"^dense_head\.resize_layers\.[01]\.weight$")
# Trunk linears stored in the conversion dtype (weights and biases, as autocast).
_TRUNK_MATMULS = re.compile(
    r"^(encoder\.blocks\.\d+\.(attn|mlp)\.\w+"
    r"|info_sharing\.(self_attention_blocks\.\d+\.(attn|mlp)\.\w+|proj_embed))"
    r"\.(weight|bias)$"
)


def _rename(key: str, num_global_layers: int) -> str:
    match = _GLOBAL_KEY.match(key)
    if match:
        name, zeros, last = match.groups()
        index = 0 if last == "0" else num_global_layers - 1 - len(zeros) // 2
        return f"{name}.layers.{index}." + key[match.end() :]
    for pattern, replacement in _RENAMES:
        key = pattern.sub(replacement, key)
    return key


def to_mlx_layout(
    weights: Dict[str, mx.array], config: ModelConfig, dtype: str = "float16"
) -> Dict[str, mx.array]:
    """PyTorch-layout MapAnything weights -> MLX keys and layouts; the trunk
    linears in ``dtype``, all other tensors in float32."""
    num_global_layers = len(config.geometric_input_config.global_intermediate_dims) + 1
    cast = getattr(mx, dtype)
    out = {}
    for key, value in weights.items():
        key = _rename(key, num_global_layers)
        if _LINEAR_CONVS.match(key):
            value = value.reshape(value.shape[:2])
        elif _CONV_TRANSPOSE.match(key):
            value = value.transpose(1, 2, 3, 0)
        elif value.ndim == 4:
            value = value.transpose(0, 2, 3, 1)
        value = value.astype(cast if _TRUNK_MATMULS.match(key) else mx.float32)
        out[key] = value
    return out


def load_official(path: Path, dtype: str = "float16"):
    """Official checkpoint directory -> (config dict with ``model_type``,
    MLX weights)."""
    config = json.loads((path / "config.json").read_text())
    config["model_type"] = "mapanything"
    weights = mx.load(str(path / "model.safetensors"))
    return config, to_mlx_layout(weights, ModelConfig.from_dict(config), dtype)


def convert(hf_path: str, mlx_path: str, dtype: str = "float16") -> Path:
    source = get_model_path(hf_path)
    config, weights = load_official(source, dtype)
    model = Model(ModelConfig.from_dict(config))
    model.load_weights(list(model.sanitize(weights).items()))

    out = Path(mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out / "model.safetensors"), weights, metadata={"format": "mlx"}
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "preprocessor_config.json").write_text(
        json.dumps({"resize_mode": "fixed_mapping", "resolution_set": 518}, indent=2)
        + "\n"
    )
    print(f"Saved {len(weights)} tensors to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", default="facebook/map-anything")
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    args = parser.parse_args()
    convert(args.hf_path, args.mlx_path, args.dtype)


if __name__ == "__main__":
    main()
