"""Convert RF-DETR PyTorch checkpoints to MLX safetensors format.

Downloads official weights directly from Roboflow's GCS bucket.
Requires only torch + safetensors for conversion (no rfdetr package).

Usage:
    python -m mlx_vlm.models.rfdetr.convert --variant base --output ./rfdetr-base-mlx
    python -m mlx_vlm.models.rfdetr.convert --variant seg-small --output ./rfdetr-seg-small-mlx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

MODEL_VARIANTS = {
    "base": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "filename": "rf-detr-base.pth",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 560,
            "dec_layers": 3,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 14,
            "num_windows": 4,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [2, 5, 8, 11],
            "projector_scale": ["P4"],
        },
    },
    "small": {
        "url": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "filename": "rf-detr-small.pth",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 512,
            "dec_layers": 3,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 14,
            "num_windows": 4,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [2, 5, 8, 11],
            "projector_scale": ["P4"],
        },
    },
    "large": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "filename": "rf-detr-large.pth",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_base",
            "hidden_dim": 256,
            "resolution": 704,
            "dec_layers": 6,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 14,
            "num_windows": 4,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [2, 5, 8, 11],
            "projector_scale": ["P4"],
        },
    },
    "seg-small": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "filename": "rf-detr-seg-small.pt",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 384,
            "dec_layers": 4,
            "num_queries": 100,
            "num_classes": 90,
            "patch_size": 12,
            "num_windows": 2,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [3, 6, 9, 12],
            "projector_scale": ["P4"],
            "segmentation": True,
            "positional_encoding_size": 32,
        },
    },
    "seg-large": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "filename": "rf-detr-seg-large.pt",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 504,
            "dec_layers": 5,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 12,
            "num_windows": 2,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [3, 6, 9, 12],
            "projector_scale": ["P4"],
            "segmentation": True,
            "positional_encoding_size": 42,
            "seg_num_blocks": 5,
        },
    },
    "seg-xlarge": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "filename": "rf-detr-seg-xl.pt",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 624,
            "dec_layers": 6,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 12,
            "num_windows": 2,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [3, 6, 9, 12],
            "projector_scale": ["P4"],
            "segmentation": True,
            "positional_encoding_size": 52,
            "seg_num_blocks": 6,
        },
    },
    "seg-2xlarge": {
        "url": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
        "filename": "rf-detr-seg-2xl.pt",
        "config": {
            "model_type": "rf-detr",
            "encoder": "dinov2_windowed_small",
            "hidden_dim": 256,
            "resolution": 768,
            "dec_layers": 6,
            "num_queries": 300,
            "num_classes": 90,
            "patch_size": 12,
            "num_windows": 2,
            "group_detr": 13,
            "sa_nheads": 8,
            "ca_nheads": 16,
            "dec_n_points": 2,
            "two_stage": True,
            "bbox_reparam": True,
            "lite_refpoint_refine": True,
            "layer_norm": True,
            "out_feature_indexes": [3, 6, 9, 12],
            "projector_scale": ["P4"],
            "segmentation": True,
            "positional_encoding_size": 64,
            "seg_num_blocks": 6,
        },
    },
}


def _download(url: str, dest: str) -> str:
    """Download file with progress."""
    if os.path.exists(dest):
        print(f"  Using cached {dest}")
        return dest
    print(f"  Downloading {url}...")
    urlretrieve(url, dest)
    print(f"  Saved to {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    return dest


def convert(variant: str, output_dir: str):
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("Error: torch and safetensors required for conversion.")
        print("  pip install torch safetensors")
        sys.exit(1)

    if variant not in MODEL_VARIANTS:
        print(f"Unknown variant: {variant}")
        print(f"Available: {', '.join(MODEL_VARIANTS.keys())}")
        sys.exit(1)

    info = MODEL_VARIANTS[variant]
    config = info["config"]

    # Download checkpoint
    cache_dir = os.path.expanduser("~/.cache/rfdetr-mlx")
    os.makedirs(cache_dir, exist_ok=True)
    pth_path = os.path.join(cache_dir, info["filename"])
    print(f"Step 1: Download checkpoint")
    _download(info["url"], pth_path)

    # Load PyTorch checkpoint
    print(f"Step 2: Extract weights")
    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    print(f"  {len(state_dict)} tensors extracted")

    # Add model. prefix for sanitize() compatibility
    prefixed = {f"model.{k}": v for k, v in state_dict.items()}

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Step 3: Save MLX format")
    weights_path = output_path / "model.safetensors"
    save_file(prefixed, str(weights_path))
    print(f"  {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  {config_path}")

    preprocessor_config = {
        "config": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "post_process_config": {"num_select": 300},
    }
    with open(output_path / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f, indent=2)

    # Verify
    print(f"Step 4: Verify MLX loading")
    import mlx.core as mx

    from .config import ModelConfig
    from .rfdetr import Model

    mc = ModelConfig.from_dict(config)
    model = Model(mc)
    weights = mx.load(str(weights_path))
    sanitized = Model.sanitize(weights)
    model.load_weights(list(sanitized.items()))

    res = config["resolution"]
    outputs = model(mx.random.normal((1, res, res, 3)))
    mx.eval(outputs["pred_logits"])
    print(f"  Forward pass OK: {outputs['pred_logits'].shape}")

    print(f"\nDone! Load with:")
    print(f"  config = json.load(open('{config_path}'))")
    print(f"  model = Model(ModelConfig.from_dict(config))")
    print(
        f"  model.load_weights(list(Model.sanitize(mx.load('{weights_path}')).items()))"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert RF-DETR weights to MLX")
    parser.add_argument(
        "--variant",
        default="base",
        choices=list(MODEL_VARIANTS.keys()),
        help="Model variant",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    convert(args.variant, args.output)


if __name__ == "__main__":
    main()
