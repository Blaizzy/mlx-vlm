"""Torch closeness test for SAM3 MLX port.

Compares HF PyTorch Sam3Model outputs with MLX Model outputs
on the same image + text input.
"""

import json
import sys
import time
from pathlib import Path

# Ensure mlx_vlm is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from io import BytesIO

# Import mlx_vlm BEFORE transformers to avoid module shadowing
from mlx_vlm.utils import load_model, get_model_path
from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor as MLXSam3Processor

from transformers import AutoModel, AutoProcessor


def load_test_image():
    """Load a COCO test image (two cats on a couch)."""
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def load_hf_model():
    """Load HF PyTorch model."""
    print("Loading HF PyTorch model...")
    model = AutoModel.from_pretrained("facebook/sam3", dtype=torch.float32)
    model.eval()
    # Load HF processor directly to avoid our auto-processor patch
    from transformers.models.sam3_video.processing_sam3_video import Sam3VideoProcessor
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
    return model, processor


def load_mlx_model():
    """Load MLX model."""
    print("Loading MLX model...")
    model_path = get_model_path("facebook/sam3")
    model = load_model(model_path)
    processor = MLXSam3Processor.from_pretrained(str(model_path))
    return model, processor


def check_closeness(name, mlx_arr, torch_arr, rtol=1e-3, atol=1e-3):
    """Compare MLX and PyTorch arrays."""
    mlx_np = np.array(mlx_arr) if not isinstance(mlx_arr, np.ndarray) else mlx_arr
    torch_np = torch_arr.detach().cpu().float().numpy() if isinstance(torch_arr, torch.Tensor) else torch_arr

    # Handle shape mismatches
    if mlx_np.shape != torch_np.shape:
        print(f"  {name}: SHAPE MISMATCH mlx={mlx_np.shape} torch={torch_np.shape}")
        return False

    max_diff = np.max(np.abs(mlx_np - torch_np))
    mean_diff = np.mean(np.abs(mlx_np - torch_np))
    is_close = np.allclose(mlx_np, torch_np, rtol=rtol, atol=atol)

    status = "PASS" if is_close else "FAIL"
    print(f"  {name}: {status} | max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} shape={mlx_np.shape}")
    return is_close


def test_full_pipeline():
    """Compare full detection pipeline outputs."""
    image = load_test_image()
    text = "a cat"

    # --- HF PyTorch ---
    hf_model, hf_proc = load_hf_model()
    detector = hf_model.detector_model

    # Process image and text separately
    hf_img_inputs = hf_proc(images=image, return_tensors="pt")
    hf_txt_inputs = hf_proc.tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=32, truncation=True
    )
    hf_inputs = {**hf_img_inputs, **hf_txt_inputs}
    print(f"\nHF input keys: {list(hf_inputs.keys())}")
    for k, v in hf_inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")

    print("\nRunning HF forward...")
    t0 = time.time()
    with torch.no_grad():
        hf_out = detector(
            pixel_values=hf_inputs["pixel_values"],
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
        )
    print(f"HF forward: {time.time() - t0:.2f}s")

    print(f"HF output type: {type(hf_out).__name__}")
    print(f"HF output fields: {[f for f in dir(hf_out) if not f.startswith('_')]}")

    hf_logits = hf_out.pred_logits  # (B, Q, 1 or num_classes)
    hf_boxes = hf_out.pred_boxes  # (B, Q, 4)
    hf_masks = hf_out.pred_masks  # (B, Q, H, W)
    hf_presence = hf_out.presence_logits  # (B, Q, 1)

    print(f"\nHF pred_logits: {hf_logits.shape}")
    print(f"HF pred_boxes: {hf_boxes.shape}")
    print(f"HF pred_masks: {hf_masks.shape}")
    if hf_presence is not None:
        print(f"HF presence_logits: {hf_presence.shape}")

    # --- MLX ---
    mlx_model, mlx_proc = load_mlx_model()

    # Use same preprocessed inputs
    pixel_values_np = hf_inputs["pixel_values"].numpy()
    input_ids_np = hf_inputs["input_ids"].numpy()
    attention_mask_np = hf_inputs["attention_mask"].numpy()

    # HF pixel_values is (B, C, H, W) -> MLX needs (B, H, W, C)
    pixel_values_mlx = mx.array(pixel_values_np.transpose(0, 2, 3, 1))
    input_ids_mlx = mx.array(input_ids_np)
    attention_mask_mlx = mx.array(attention_mask_np)

    print(f"\nMLX pixel_values: {pixel_values_mlx.shape}")
    print(f"MLX input_ids: {input_ids_mlx.shape}")

    print("\nRunning MLX forward...")
    t0 = time.time()
    mlx_out = mlx_model.detect(
        pixel_values_mlx,
        input_ids_mlx,
        attention_mask_mlx,
    )
    mx.eval(mlx_out)
    print(f"MLX forward: {time.time() - t0:.2f}s")

    mlx_logits = mlx_out["pred_logits"]
    mlx_boxes = mlx_out["pred_boxes"]
    mlx_masks = mlx_out["pred_masks"]

    print(f"\nMLX pred_logits: {mlx_logits.shape}")
    print(f"MLX pred_boxes: {mlx_boxes.shape}")
    print(f"MLX pred_masks: {mlx_masks.shape}")

    # --- Compare ---
    print("\n=== Closeness Test Results ===")

    # Flatten MLX logits to match HF shape
    mlx_logits_np = np.array(mlx_logits).squeeze(-1) if mlx_logits.ndim == 3 else np.array(mlx_logits)
    hf_logits_np = hf_logits.detach().numpy()
    check_closeness("pred_logits", mlx_logits_np, hf_logits_np, rtol=1e-2, atol=1e-2)
    check_closeness("pred_boxes", mlx_boxes, hf_boxes, rtol=1e-2, atol=1e-2)

    # Masks may have different spatial sizes; compare if matching
    mlx_masks_np = np.array(mlx_masks)
    hf_masks_np = hf_masks.detach().numpy()
    if mlx_masks_np.shape == hf_masks_np.shape:
        check_closeness("pred_masks", mlx_masks_np, hf_masks_np, rtol=1e-1, atol=1e-1)
    else:
        print(f"  pred_masks: shapes differ mlx={mlx_masks_np.shape} torch={hf_masks_np.shape}")

    # Compare detection scores (sigmoid of logits * presence)
    hf_scores_np = 1 / (1 + np.exp(-hf_logits_np))
    if hf_presence is not None:
        hf_pres_np = hf_presence.detach().numpy()
        hf_pres_sig = 1 / (1 + np.exp(-hf_pres_np))
        hf_scores_np = hf_scores_np * hf_pres_sig  # broadcast (1,200) * (1,1)

    mlx_scores_np = 1 / (1 + np.exp(-mlx_logits_np))

    check_closeness("detection_scores", mlx_scores_np, hf_scores_np, rtol=1e-1, atol=1e-1)

    # Show top detections from both
    hf_top5 = np.argsort(hf_scores_np[0])[-5:][::-1]
    mlx_top5 = np.argsort(mlx_scores_np[0])[-5:][::-1]

    print("\nTop 5 HF detections:")
    for i in hf_top5:
        box = hf_out.pred_boxes[0, i].numpy()
        print(f"  Query {i}: score={hf_scores_np[0, i]:.4f} box={box}")

    mlx_boxes_np = np.array(mlx_boxes)
    print("\nTop 5 MLX detections:")
    for i in mlx_top5:
        box = mlx_boxes_np[0, int(i)]
        print(f"  Query {i}: score={mlx_scores_np[0, int(i)]:.4f} box={box}")

    return {
        "hf_out": hf_out,
        "mlx_out": mlx_out,
        "hf_scores": hf_scores_np,
        "mlx_scores": mlx_scores_np,
        "image": image,
        "hf_model": hf_model,
        "mlx_model": mlx_model,
        "hf_proc": hf_proc,
    }


def save_prediction_image(results, output_path="sam3_predictions.png"):
    """Save image with prediction overlays from both models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image = results["image"]
    W, H = image.size

    hf_scores = results["hf_scores"][0]
    mlx_scores = results["mlx_scores"][0]

    hf_boxes = results["hf_out"].pred_boxes[0].detach().numpy()
    mlx_boxes = np.array(results["mlx_out"]["pred_boxes"][0])

    hf_masks = results["hf_out"].pred_masks[0].detach().numpy()
    mlx_masks = np.array(results["mlx_out"]["pred_masks"][0])

    threshold = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, scores, boxes, masks, title in [
        (axes[0], hf_scores, hf_boxes, hf_masks, "HF PyTorch"),
        (axes[1], mlx_scores, mlx_boxes, mlx_masks, "MLX"),
    ]:
        ax.imshow(image)
        ax.set_title(f"{title} (threshold={threshold})", fontsize=14)

        keep = scores > threshold
        kept_scores = scores[keep]
        kept_boxes = boxes[keep]
        kept_masks = masks[keep]

        # Sort by score descending
        order = np.argsort(kept_scores)[::-1]

        colors = plt.cm.Set1(np.linspace(0, 1, max(len(order), 1)))

        # Overlay masks
        combined_mask = np.zeros((H, W, 4))
        for idx, i in enumerate(order[:10]):  # max 10
            mask = kept_masks[i]
            # Resize mask to image size
            mask_pil = Image.fromarray(mask.astype(np.float32))
            mask_pil = mask_pil.resize((W, H), Image.BILINEAR)
            mask_np = np.array(mask_pil)
            binary = mask_np > 0

            color = colors[idx % len(colors)]
            for c in range(3):
                combined_mask[:, :, c] += binary * color[c] * 0.4
            combined_mask[:, :, 3] += binary * 0.4

        combined_mask = np.clip(combined_mask, 0, 1)
        ax.imshow(combined_mask, alpha=0.6)

        # Draw boxes
        for idx, i in enumerate(order[:10]):
            cx, cy, w, h = kept_boxes[i]
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            bw = w * W
            bh = h * H
            color = colors[idx % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), bw, bh, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5,
                f"{kept_scores[i]:.2f}",
                color="white",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved prediction image: {output_path}")


def save_video_tracking(mlx_model, mlx_proc, output_path="sam3_tracking.png"):
    """Simulate video tracking on multiple frames and save visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlx_vlm.models.sam3.generate import Sam3Predictor

    predictor = Sam3Predictor(mlx_model, mlx_proc, score_threshold=0.1)

    # Use 4 different COCO images as "video frames"
    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000285138.jpg",  # kitchen
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # airplane
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # zebras
    ]
    prompts = ["a cat", "a bottle", "an airplane", "a zebra"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (url, prompt) in enumerate(zip(urls, prompts)):
        print(f"\nProcessing frame {idx+1}: '{prompt}'...")
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"  Failed to load image: {e}")
            axes[idx].text(0.5, 0.5, f"Failed: {e}", ha="center", va="center")
            continue

        result = predictor.predict(image, text_prompt=prompt, score_threshold=0.1)

        W, H = image.size
        ax = axes[idx]
        ax.imshow(image)

        # Overlay masks
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(result.scores), 1)))
        combined_mask = np.zeros((H, W, 4))

        for i in range(min(5, len(result.scores))):
            mask = result.masks[i]
            if mask.shape != (H, W):
                mask_pil = Image.fromarray(mask.astype(np.float32))
                mask_pil = mask_pil.resize((W, H), Image.BILINEAR)
                mask = np.array(mask_pil)
            binary = mask > 0
            color = colors[i % len(colors)]
            for c in range(3):
                combined_mask[:, :, c] += binary * color[c] * 0.4
            combined_mask[:, :, 3] += binary * 0.4

            # Draw box
            x1, y1, x2, y2 = result.boxes[i]
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, f"{result.scores[i]:.2f}",
                color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
            )

        combined_mask = np.clip(combined_mask, 0, 1)
        ax.imshow(combined_mask, alpha=0.6)
        ax.set_title(f'"{prompt}" ({len(result.scores)} detections)', fontsize=12)
        ax.axis("off")

    plt.suptitle("SAM3 MLX - Multi-Image Detection", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved tracking visualization: {output_path}")


if __name__ == "__main__":
    results = test_full_pipeline()
    save_prediction_image(results, "/Users/prince_canuma/Projects/mlx-vlm/sam3_predictions.png")
    mlx_proc = MLXSam3Processor.from_pretrained(str(get_model_path("facebook/sam3")))
    save_video_tracking(
        results["mlx_model"],
        mlx_proc,
        "/Users/prince_canuma/Projects/mlx-vlm/sam3_tracking.png",
    )
