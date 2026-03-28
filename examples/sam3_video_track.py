"""SAM3 MLX — Video tracking demo with full frame-by-frame output."""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

OUTPUT_DIR = Path(__file__).parent / "sam3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS_BGR = [
    (181, 120, 31),   # blue
    (13, 128, 255),   # orange
    (43, 161, 43),    # green
    (41, 38, 214),    # red
    (189, 102, 148),  # purple
    (74, 87, 140),    # brown
]


def _nms_results(result, iou_thresh=0.5):
    """Remove duplicate detections via NMS."""
    if len(result.scores) == 0:
        return result
    boxes, scores, masks = result.boxes, result.scores, result.masks
    order = np.argsort(-scores)
    keep = []
    for i in order:
        discard = False
        for j in keep:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            a_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            if inter / max(a_i + a_j - inter, 1e-6) > iou_thresh:
                discard = True
                break
        if not discard:
            keep.append(i)
    from mlx_vlm.models.sam3.generate import DetectionResult
    return DetectionResult(boxes=boxes[keep], masks=masks[keep], scores=scores[keep])


def main():
    from mlx_vlm.utils import load_model, get_model_path
    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    import mlx.core as mx
    import mlx.nn as nn

    # ── Load model ──────────────────────────────────────────────
    model_path = get_model_path("facebook/sam3")
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=0.15)

    # ── Open video ──────────────────────────────────────────────
    video_path = "/Users/prince_canuma/Downloads/car_video.mp4"
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")

    # ── Detect on frame 0 ──────────────────────────────────────
    ret, frame0_bgr = cap.read()
    frame0_rgb = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2RGB)
    frame0_pil = Image.fromarray(frame0_rgb)

    result = predictor.predict(frame0_pil, text_prompt="a car")

    # NMS
    result = _nms_results(result, iou_thresh=0.5)
    print(f"Frame 0: {len(result.scores)} cars detected")
    for i in range(len(result.scores)):
        x1, y1, x2, y2 = result.boxes[i]
        print(f"  [{result.scores[i]:.2f}] box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    if len(result.scores) == 0:
        print("No detections, exiting.")
        return

    # ── Process every Nth frame ─────────────────────────────────
    # Process every 2nd frame for speed, write every frame
    process_every = 2
    out_path = str(OUTPUT_DIR / "4_tracking_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Write frame 0 with detection overlay
    overlay0 = _draw_frame(frame0_bgr, result.masks, result.scores, result.boxes, 0)
    writer.write(overlay0)

    # Store latest masks/scores for non-processed frames
    latest_masks = result.masks
    latest_scores = result.scores
    latest_boxes = result.boxes

    # For subsequent frames, run detection per-frame (simpler than full tracker memory)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    for fi in range(1, total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if fi % process_every == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            res = predictor.predict(frame_pil, text_prompt="a car")
            res = _nms_results(res, iou_thresh=0.5)
            latest_masks = res.masks
            latest_scores = res.scores
            latest_boxes = res.boxes

            if fi % 20 == 0:
                print(f"  Frame {fi}/{total_frames}: {len(res.scores)} detections")

        overlay = _draw_frame(frame_bgr, latest_masks, latest_scores, latest_boxes, fi)
        writer.write(overlay)

    writer.release()
    cap.release()
    print(f"\nSaved: {out_path}")

    # ── Also save a grid image of key frames ────────────────────
    _save_grid(video_path, predictor, _nms_results)


def _draw_frame(frame_bgr, masks, scores, boxes, frame_idx):
    """Overlay masks and boxes on a BGR frame."""
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    for i in range(len(scores)):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        mask = masks[i]

        # Resize mask to frame size
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0

        # Mask overlay
        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                (out[:, :, c].astype(np.float32) * 0.55 + color[c] * 0.45).astype(np.uint8),
                out[:, :, c],
            )

        # Mask contour
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 2)

        # Box
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"car {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, max(y1 - th - 10, 0)), (x1 + tw + 6, max(y1, th + 10)), color, -1)
        cv2.putText(out, label, (x1 + 3, max(y1 - 4, th + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Frame counter
    cv2.putText(out, f"Frame {frame_idx}", (W - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


def _save_grid(video_path, predictor, nms_fn):
    """Save a 2x3 grid of key frames with overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    indices = [0, int(fps * 2), int(fps * 4), int(fps * 6), int(fps * 8), total - 1]
    indices = [min(i, total - 1) for i in indices]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    axes = axes.flatten()

    for ax_idx, fi in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame_bgr = cap.read()
        if not ret:
            axes[ax_idx].axis("off")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        result = predictor.predict(frame_pil, text_prompt="a car")
        result = nms_fn(result, iou_thresh=0.5)

        overlay = _draw_frame(frame_bgr, result.masks, result.scores, result.boxes, fi)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        axes[ax_idx].imshow(overlay_rgb)
        n = len(result.scores)
        scores_str = ", ".join(f"{s:.2f}" for s in result.scores[:3])
        axes[ax_idx].set_title(f"Frame {fi} — {n} cars [{scores_str}]", fontsize=11)
        axes[ax_idx].axis("off")

    cap.release()

    plt.suptitle('SAM3 MLX — Video Tracking "a car"', fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "4_tracking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
