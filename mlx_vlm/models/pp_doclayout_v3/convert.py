"""Convert PP-DocLayoutV3 checkpoints to MLX format.

Load safetensors directly with MLX, rename keys, transpose Conv2d weights
from NCHW to NHWC, and cast to the requested dtype (float32 by default).
Drops batch counters and the training-only denoising embedding.

Usage:
    python -m mlx_vlm.models.pp_doclayout_v3.convert \\
        --hf-path PaddlePaddle/PP-DocLayoutV3_safetensors \\
        --output ./pp-doclayout-v3-mlx \\
        --dtype float32
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .pp_doclayout_v3 import Model


def convert(hf_path: str, output: str, dtype: str = "float32") -> Path:
    try:
        import mlx.core as mx
        from huggingface_hub import snapshot_download

        from ...utils import MODEL_CONVERSION_DTYPES
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        sys.exit(2)

    if dtype not in MODEL_CONVERSION_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; expected one of {MODEL_CONVERSION_DTYPES}"
        )
    mlx_dtype = getattr(mx, dtype)

    src = Path(hf_path)
    if not src.exists():
        src = Path(snapshot_download(hf_path))
    ckpt_file = src / "model.safetensors"
    cfg_file = src / "config.json"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"No model.safetensors in {src}")
    if not cfg_file.is_file():
        raise FileNotFoundError(f"No config.json in {src}")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    raw = mx.load(str(ckpt_file))
    print(f"Loaded {len(raw)} tensors from {ckpt_file}")
    sanitized: Dict[str, mx.array] = {
        k: v.astype(mlx_dtype) for k, v in Model.sanitize(raw).items()
    }
    n_dropped = len(raw) - len(sanitized)
    print(f"  renamed {len(sanitized)} tensors, dropped {n_dropped}, cast to {dtype}")

    weights_path = out / "model.safetensors"
    mx.save_safetensors(str(weights_path), sanitized, metadata={"format": "mlx"})
    print(f"  wrote {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    shutil.copy2(cfg_file, out / "config.json")
    print("  copied config.json")

    _verify(out)
    return out


def _verify(out: Path) -> None:
    """Smoke-test via the framework loader + a random forward."""
    import mlx.core as mx

    from ...utils import load_model

    model = load_model(out)
    model.eval()
    pixel = mx.random.normal((1, 256, 256, 3), dtype=mx.float32)
    out_dict = model(pixel)
    mx.eval(out_dict["logits"], out_dict["pred_boxes"], out_dict["order_logits"])
    print(
        f"  verified forward: logits {tuple(out_dict['logits'].shape)}, "
        f"boxes {tuple(out_dict['pred_boxes'].shape)}, "
        f"order {tuple(out_dict['order_logits'].shape)}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    from ...utils import MODEL_CONVERSION_DTYPES

    parser = argparse.ArgumentParser(description="Convert PP-DocLayoutV3 -> MLX.")
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float32", choices=MODEL_CONVERSION_DTYPES)
    args = parser.parse_args(argv)
    convert(args.hf_path, args.output, args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
