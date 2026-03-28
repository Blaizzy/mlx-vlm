"""SAM3 video tracking CLI.

Usage:
    python -m mlx_vlm.models.sam3.track_video --video input.mp4 --prompt "a car"
    python -m mlx_vlm.models.sam3.track_video --video input.mp4 --prompt "a person" --every 4 --threshold 0.2 --output out.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

COLORS_BGR = [
    (181, 120, 31),
    (13, 128, 255),
    (43, 161, 43),
    (41, 38, 214),
    (189, 102, 148),
    (74, 87, 140),
]


def nms(boxes, scores, masks, iou_thresh=0.5):
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
    return boxes[keep], scores[keep], masks[keep]


def draw_frame(frame_bgr, masks, scores, boxes, prompt, H, W):
    out = frame_bgr.copy()
    for i in range(len(scores)):
        color = COLORS_BGR[i % len(COLORS_BGR)]
        mask = masks[i]
        if mask.shape[0] != H or mask.shape[1] != W:
            mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        binary = mask > 0

        for c in range(3):
            out[:, :, c] = np.where(
                binary,
                (out[:, :, c].astype(np.float32) * 0.55 + color[c] * 0.45).astype(np.uint8),
                out[:, :, c],
            )

        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)

        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{prompt} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, max(y1 - th - 10, 0)), (x1 + tw + 6, max(y1, th + 10)), color, -1)
        cv2.putText(out, label, (x1 + 3, max(y1 - 4, th + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return out


def main():
    parser = argparse.ArgumentParser(description="SAM3 video tracking")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g. 'a car')")
    parser.add_argument("--output", default=None, help="Output video path (default: <input>_tracked.mp4)")
    parser.add_argument("--model", default="facebook/sam3", help="Model path or HF repo")
    parser.add_argument("--threshold", type=float, default=0.15, help="Score threshold")
    parser.add_argument("--nms-thresh", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--every", type=int, default=2, help="Run detection every N frames")
    args = parser.parse_args()

    video_path = args.video
    if args.output is None:
        p = Path(video_path)
        args.output = str(p.parent / f"{p.stem}_tracked{p.suffix}")

    # Load model
    from mlx_vlm.utils import load_model, get_model_path
    from mlx_vlm.models.sam3.generate import Sam3Predictor
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

    print(f"Loading model: {args.model}")
    model_path = get_model_path(args.model)
    model = load_model(model_path)
    processor = Sam3Processor.from_pretrained(str(model_path))
    predictor = Sam3Predictor(model, processor, score_threshold=args.threshold)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")
    print(f'Prompt: "{args.prompt}", detect every {args.every} frames, threshold {args.threshold}')

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    latest_masks = np.array([])
    latest_scores = np.array([])
    latest_boxes = np.array([])

    for fi in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if fi % args.every == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = predictor.predict(frame_pil, text_prompt=args.prompt, score_threshold=args.threshold)
            if len(result.scores) > 0:
                latest_boxes, latest_scores, latest_masks = nms(
                    result.boxes, result.scores, result.masks, args.nms_thresh
                )
            else:
                latest_masks = np.array([])
                latest_scores = np.array([])
                latest_boxes = np.array([])

            if fi % 40 == 0:
                print(f"  Frame {fi}/{total_frames}: {len(latest_scores)} detections")

        out = draw_frame(frame_bgr, latest_masks, latest_scores, latest_boxes, args.prompt, H, W)
        writer.write(out)

    writer.release()
    cap.release()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
