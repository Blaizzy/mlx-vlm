"""SAM3 MLX — Full inference demo: Object Detection, Instance Segmentation, Video Tracking."""

import numpy as np
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path(__file__).parent / "sam3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = [
    (0.12, 0.47, 0.71),  # blue
    (1.00, 0.50, 0.05),  # orange
    (0.17, 0.63, 0.17),  # green
    (0.84, 0.15, 0.16),  # red
    (0.58, 0.40, 0.74),  # purple
    (0.55, 0.34, 0.29),  # brown
    (0.89, 0.47, 0.76),  # pink
    (0.50, 0.50, 0.50),  # gray
    (0.74, 0.74, 0.13),  # olive
    (0.09, 0.75, 0.81),  # cyan
]


def load_model():
    from mlx_vlm.utils import load_model, get_model_path
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

    model_path = get_model_path("facebook/sam3")
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    return model, processor


def fetch_image(url: str) -> Image.Image:
    return Image.open(BytesIO(requests.get(url, timeout=15).content)).convert("RGB")


def nms_results(result, iou_thresh=0.5):
    """Apply NMS to remove duplicate detections."""
    if len(result.scores) == 0:
        return result

    boxes = result.boxes
    scores = result.scores
    masks = result.masks

    order = np.argsort(-scores)
    keep = []
    for i in order:
        discard = False
        for j in keep:
            # Compute IoU
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            a_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            iou = inter / max(a_i + a_j - inter, 1e-6)
            if iou > iou_thresh:
                discard = True
                break
        if not discard:
            keep.append(i)

    from mlx_vlm.models.sam3.generate import DetectionResult
    return DetectionResult(
        boxes=boxes[keep],
        masks=masks[keep],
        scores=scores[keep],
    )


def draw_overlay(ax, image, result, title=""):
    """Draw masks + boxes on axes."""
    W, H = image.size
    ax.imshow(image)

    for i in range(len(result.scores)):
        color = COLORS[i % len(COLORS)]

        # Mask overlay
        mask = result.masks[i]
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H), Image.BILINEAR))
        binary = mask > 0
        overlay = np.zeros((H, W, 4))
        for c in range(3):
            overlay[:, :, c] = binary * color[c]
        overlay[:, :, 3] = binary * 0.5
        ax.imshow(overlay)

        # Box
        x1, y1, x2, y2 = result.boxes[i]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 8, 12), f"{result.scores[i]:.2f}",
            color="white", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
        )

    ax.set_title(title, fontsize=14)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────
# 1. Object Detection
# ─────────────────────────────────────────────────────────────────
def demo_detection(predictor):
    print("\n=== 1. Object Detection ===")
    image = fetch_image("http://images.cocodataset.org/val2017/000000039769.jpg")

    result = predictor.predict(image, text_prompt="a cat")
    result = nms_results(result)
    print(f'Prompt: "a cat" — {len(result.scores)} detections')
    for i in range(len(result.scores)):
        x1, y1, x2, y2 = result.boxes[i]
        print(f"  [{result.scores[i]:.2f}] box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # Detection: boxes only, no masks
    ax.imshow(image)
    for i in range(len(result.scores)):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = result.boxes[i]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 8, 12), f"cat {result.scores[i]:.2f}",
            color="white", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.85),
        )
    ax.set_title('Object Detection — "a cat"', fontsize=15)
    ax.axis("off")
    plt.tight_layout()
    out = OUTPUT_DIR / "1_detection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# 2. Instance Segmentation
# ─────────────────────────────────────────────────────────────────
def demo_segmentation(predictor):
    print("\n=== 2. Instance Segmentation ===")
    image = fetch_image("http://images.cocodataset.org/val2017/000000039769.jpg")

    result = predictor.predict(image, text_prompt="a cat")
    result = nms_results(result)
    print(f'Prompt: "a cat" — {len(result.scores)} instances')
    for i in range(len(result.scores)):
        area = int(result.masks[i].sum())
        print(f"  [{result.scores[i]:.2f}] mask area={area}px")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    draw_overlay(ax, image, result, title='Instance Segmentation — "a cat"')
    plt.tight_layout()
    out = OUTPUT_DIR / "2_segmentation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# 3. Multi-prompt Detection
# ─────────────────────────────────────────────────────────────────
def demo_multi_prompt(predictor):
    print("\n=== 3. Multi-Prompt Detection ===")
    urls_prompts = [
        ("http://images.cocodataset.org/val2017/000000039769.jpg", "a cat"),
        ("http://images.cocodataset.org/val2017/000000397133.jpg", "an airplane"),
        ("http://images.cocodataset.org/val2017/000000252219.jpg", "a zebra"),
        ("http://images.cocodataset.org/val2017/000000087038.jpg", "a car"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for idx, (url, prompt) in enumerate(urls_prompts):
        print(f'  [{idx+1}/4] "{prompt}"...', end=" ", flush=True)
        try:
            image = fetch_image(url)
        except Exception as e:
            print(f"FAILED ({e})")
            axes[idx].text(0.5, 0.5, "Failed to load", ha="center", va="center", fontsize=14)
            axes[idx].set_title(f'"{prompt}"')
            axes[idx].axis("off")
            continue

        result = predictor.predict(image, text_prompt=prompt, score_threshold=0.1)
        result = nms_results(result)
        print(f"{len(result.scores)} detections")
        draw_overlay(axes[idx], image, result, title=f'"{prompt}" ({len(result.scores)} det.)')

    plt.suptitle("SAM3 MLX — Multi-Prompt Detection", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "3_multi_prompt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# 4. Video Tracking
# ─────────────────────────────────────────────────────────────────
def demo_tracking(model, processor):
    import cv2

    print("\n=== 4. Video Tracking ===")
    video_path = "/Users/prince_canuma/Downloads/car_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")

    # Sample frames evenly (every ~2 seconds)
    sample_interval = max(1, int(fps * 2))
    frame_indices = list(range(0, total_frames, sample_interval))[:6]

    frames = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    print(f"  Sampled {len(frames)} frames at indices {frame_indices}")

    if len(frames) < 2:
        print("  Not enough frames.")
        return

    from mlx_vlm.models.sam3.generate import Sam3VideoPredictor
    video_predictor = Sam3VideoPredictor(model, processor, score_threshold=0.15)
    video_predictor.set_video(frames)

    obj_id = video_predictor.add_text_prompt("a car", frame_idx=0)
    print(f'  Tracking object {obj_id} ("a car") across {len(frames)} frames...')

    results = video_predictor.propagate()

    ncols = min(len(results), 3)
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(results):
        if i >= len(axes):
            break
        frame = frames[r.frame_idx]
        W_f, H_f = frame.size
        ax = axes[i]
        ax.imshow(frame)

        for j in range(len(r.scores)):
            mask = r.masks[j]
            if mask.shape[0] != H_f or mask.shape[1] != W_f:
                mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W_f, H_f), Image.BILINEAR))
            binary = mask > 0
            color = COLORS[j % len(COLORS)]
            overlay = np.zeros((H_f, W_f, 4))
            for c in range(3):
                overlay[:, :, c] = binary * color[c]
            overlay[:, :, 3] = binary * 0.5
            ax.imshow(overlay)

        n_obj = len(r.object_ids) if r.object_ids else 0
        scores_str = ", ".join(f"{s:.2f}" for s in r.scores[:3])
        ax.set_title(f"Frame {r.frame_idx} — {n_obj} obj [{scores_str}]", fontsize=11)
        ax.axis("off")

    # Hide empty axes
    for i in range(len(results), len(axes)):
        axes[i].axis("off")

    plt.suptitle('SAM3 MLX — Video Tracking "a car"', fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "4_tracking.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Also save as video with overlays
    _save_tracking_video(model, processor, frames, results, frame_indices, fps)


def _save_tracking_video(model, processor, frames, results, frame_indices, fps):
    """Save an MP4 video with mask overlays."""
    import cv2

    out_path = str(OUTPUT_DIR / "4_tracking_video.mp4")
    frame0 = np.array(frames[0])
    H, W = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 4.0, (W, H))

    for r in results:
        frame_np = np.array(frames[r.frame_idx]).copy()
        for j in range(len(r.scores)):
            mask = r.masks[j]
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = np.array(Image.fromarray(mask.astype(np.float32)).resize((W, H), Image.BILINEAR))
            binary = mask > 0
            color = COLORS[j % len(COLORS)]
            for c in range(3):
                frame_np[:, :, c] = np.where(
                    binary,
                    (frame_np[:, :, c] * 0.5 + int(color[c] * 255) * 0.5).astype(np.uint8),
                    frame_np[:, :, c],
                )
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Add text
        cv2.putText(
            frame_bgr,
            f"Frame {r.frame_idx}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
        )
        for j, s in enumerate(r.scores[:3]):
            cv2.putText(
                frame_bgr,
                f"obj {r.object_ids[j] if r.object_ids else j}: {s:.2f}",
                (20, 80 + j * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )

        writer.write(frame_bgr)

    writer.release()
    print(f"Saved video: {out_path}")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, processor = load_model()

    from mlx_vlm.models.sam3.generate import Sam3Predictor
    predictor = Sam3Predictor(model, processor, score_threshold=0.3)

    demo_detection(predictor)
    demo_segmentation(predictor)
    demo_multi_prompt(predictor)
    demo_tracking(model, processor)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
