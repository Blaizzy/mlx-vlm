"""Convert the OmniParser icon_detect PyTorch checkpoint to MLX safetensors.

The reference checkpoint is microsoft/OmniParser-v2.0 ``icon_detect/model.pt``
(Ultralytics YOLO11-family, nc=1). Conversion transposes 4-D conv weights
from PyTorch (O, I, H, W) to MLX (O, H, W, I) layout.

Requires torch at conversion time only; inference needs no PyTorch.

Usage:
    python -m mlx_vlm.models.yolo11.convert \
        --ckpt ~/models/OmniParser-v2.0/icon_detect/model.pt --output ./omniparser-icon-detect-mlx
"""

import argparse
import json
from pathlib import Path


def convert(ckpt_path: str, output_dir: str):
    import mlx.core as mx
    import torch

    from .yolo11 import YOLO11, load_weights

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_pt = ckpt["model"].float()
    nc = int(model_pt.nc)
    state = model_pt.state_dict()

    mlx_weights = {}
    n_conv = 0
    for key, tensor in state.items():
        if "num_batches_tracked" in key:
            continue
        if key.endswith("dfl.conv.weight"):
            key = key.replace("dfl.conv.weight", "dfl.weight")
        if tensor.ndim == 4:
            tensor = tensor.permute(0, 2, 3, 1).contiguous()
            n_conv += 1
        mlx_weights[key] = mx.array(tensor.numpy())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "model.safetensors"
    mx.save_safetensors(weights_path, mlx_weights)

    config = {
        "model_type": "yolo11",
        "nc": nc,
        "ch": [256, 512, 512],
        "reg_max": 16,
        "names": {int(k): str(v) for k, v in model_pt.names.items()},
        "stride": [8, 16, 32],
    }
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"Converted {len(mlx_weights)} tensors ({n_conv} conv transposed) "
        f"-> {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)"
    )

    # Verify: load into the MLX model and run a forward pass.
    model = YOLO11(nc=nc)
    load_weights(model, mx.load(str(weights_path)))
    x = mx.random.normal((1, 640, 640, 3))
    preds = model(x)
    mx.eval(preds)
    print(f"Forward OK: output shape {tuple(preds.shape)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to icon_detect/model.pt (OmniParser-v2.0)",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    convert(args.ckpt, args.output)


if __name__ == "__main__":
    main()
