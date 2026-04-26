"""Convert facebook/sapiens-pose-bbox-detector (mmdet RTMDet .pth) to MLX.

Usage:
  python -m mlx_vlm.models.rtmdet.convert \
      --hf-repo facebook/sapiens-pose-bbox-detector \
      --out ./rtmdet-person-mlx \
      --dtype bfloat16
"""

import argparse
import dataclasses
import json
from pathlib import Path

import mlx.core as mx
import torch
from huggingface_hub import hf_hub_download

from .config import RTMDetConfig
from .rtmdet import Model


def _to_mx(tensor: torch.Tensor) -> mx.array:
    return mx.array(tensor.detach().cpu().numpy())


def convert(hf_repo: str, out_dir: Path, dtype: str = "bfloat16",
            arch: str = "m", ckpt_filename: str = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_filename is None:
        # default for the sapiens pose bbox detector
        ckpt_filename = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    pth = hf_hub_download(hf_repo, ckpt_filename)
    print(f"[convert] loading {pth}")
    obj = torch.load(pth, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj.get("model", obj))

    # Convert each tensor to mx.array, filter tracking counters
    weights = {k: _to_mx(v) for k, v in sd.items() if "num_batches_tracked" not in k}
    sanitized = Model.sanitize(weights)

    target_dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
    cast = {k: v.astype(target_dtype) if mx.issubdtype(v.dtype, mx.floating) else v
            for k, v in sanitized.items()}
    mx.save_safetensors(str(out_dir / "model.safetensors"), cast)

    cfg = RTMDetConfig(arch=arch)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.pop("text_config", None)
    cfg_dict.pop("vision_config", None)
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    (out_dir / "preprocessor_config.json").write_text(json.dumps({
        "image_size": [640, 640],
        "pad_value": [114, 114, 114],
    }, indent=2))

    print(f"[convert] wrote {len(cast)} tensors to {out_dir}/model.safetensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="facebook/sapiens-pose-bbox-detector")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--arch", default="m", choices=["tiny", "s", "m", "l", "x"])
    ap.add_argument("--ckpt", default=None, help="override checkpoint filename")
    args = ap.parse_args()
    convert(args.hf_repo, args.out, args.dtype, args.arch, args.ckpt)


if __name__ == "__main__":
    main()
