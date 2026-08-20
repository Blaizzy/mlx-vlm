"""Video processing for SAM 3D Body MLX inference.

Processes each frame through the body estimation pipeline,
renders 3D keypoints and skeleton overlay, outputs annotated video.

Usage:
    python -m mlx_vlm.models.sam3d_body.video --input video.mp4 --output output.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Skeleton connections using MHR70 keypoint ordering (NOT COCO-17).
# MHR70: 0-4=head, 5-6=shoulders, 7-8=elbows, 9-10=hips, 11-12=knees,
# 13-14=ankles, 15-17=L foot, 18-20=R foot, 21-41=R hand(+wrist@41),
# 42-62=L hand(+wrist@62), 63-64=olecranon, 65-66=cubital fossa,
# 67-68=acromion, 69=neck
SKELETON_PAIRS = [
    # Head
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (1, 2),
    (3, 5),
    (4, 6),  # ears to shoulders
    # Torso
    (5, 6),
    (5, 9),
    (6, 10),
    (9, 10),
    # Left arm: shoulder(5) -> elbow(7) -> wrist(62)
    (5, 7),
    (7, 62),
    # Right arm: shoulder(6) -> elbow(8) -> wrist(41)
    (6, 8),
    (8, 41),
    # Left leg: hip(9) -> knee(11) -> ankle(13)
    (9, 11),
    (11, 13),
    # Right leg: hip(10) -> knee(12) -> ankle(14)
    (10, 12),
    (12, 14),
    # Left foot fan
    (13, 15),
    (13, 16),
    (13, 17),
    # Right foot fan
    (14, 18),
    (14, 19),
    (14, 20),
]

JOINT_COLORS = [
    (255, 0, 0),  # red - head/face
    (0, 255, 0),  # green - torso
    (0, 0, 255),  # blue - arms
    (255, 255, 0),  # yellow - legs
]


def bbox_iou(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def track_person(detections, prev_bbox, iou_threshold=0.3):
    """Pick the detection that best matches prev_bbox by IoU.

    Falls back to largest detection if no match exceeds the threshold.
    Returns the chosen bbox or None if no detections.
    """
    if not detections:
        return None
    if prev_bbox is None:
        return detections[0]  # first frame: largest person

    best_iou = 0.0
    best_box = None
    for det in detections:
        score = bbox_iou(det, prev_bbox)
        if score > best_iou:
            best_iou = score
            best_box = det

    if best_iou >= iou_threshold:
        return best_box
    return detections[0]  # lost track: fall back to largest


def project_keypoints_perspective(
    keypoints_3d, camera, bbox, img_w, img_h, fov_deg=60.0
):
    """Project 3D keypoints to 2D using full perspective projection.

    Replicates the PyTorch PerspectiveHead.perspective_projection() logic:
    1. Flip scale and ty (camera system difference)
    2. Convert (s, tx, ty) to camera translation (tx', ty', tz) using focal length + bbox
    3. Add translation to 3D points
    4. Perspective project using intrinsic matrix K

    camera: (3,) = [scale, tx, ty]
    keypoints_3d: (N, 3)
    bbox: [x1, y1, x2, y2]
    """
    import math

    pred_cam = camera.copy()
    pred_cam[[0, 2]] *= -1  # flip s and ty

    s, tx, ty = pred_cam

    bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    focal_length = img_h / (2 * math.tan(math.radians(fov_deg / 2)))

    bs = bbox_size * s + 1e-8
    tz = 2 * focal_length / bs
    cx = 2 * (bbox_center[0] - img_w / 2) / bs
    cy = 2 * (bbox_center[1] - img_h / 2) / bs

    cam_t = np.array([tx + cx, ty + cy, tz])

    # Translate 3D points to camera space
    j3d_cam = keypoints_3d + cam_t[None, :]

    # Perspective projection: K @ (p / p_z)
    j3d_norm = j3d_cam / j3d_cam[:, 2:3]
    kp_2d = np.zeros((keypoints_3d.shape[0], 2))
    kp_2d[:, 0] = focal_length * j3d_norm[:, 0] + img_w / 2
    kp_2d[:, 1] = focal_length * j3d_norm[:, 1] + img_h / 2

    return kp_2d


def draw_skeleton(frame, keypoints_2d, confidence_threshold=0.0):
    """Draw skeleton overlay on frame."""
    h, w = frame.shape[:2]

    # Draw connections
    for i, j in SKELETON_PAIRS:
        if i >= len(keypoints_2d) or j >= len(keypoints_2d):
            continue
        pt1 = tuple(keypoints_2d[i].astype(int))
        pt2 = tuple(keypoints_2d[j].astype(int))
        # Skip out-of-bounds points
        if not (0 <= pt1[0] < w and 0 <= pt1[1] < h):
            continue
        if not (0 <= pt2[0] < w and 0 <= pt2[1] < h):
            continue
        cv2.line(frame, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

    # Draw joints
    for i, pt in enumerate(keypoints_2d):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            color = JOINT_COLORS[min(i // 5, len(JOINT_COLORS) - 1)]
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_bbox(frame, bbox, color=(0, 200, 255), thickness=2):
    """Draw bounding box on frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def process_video(
    input_path: str,
    output_path: str,
    weights_dir: str = "/tmp/sam3d-mlx-weights/",
    bbox: list = None,
    target_region: list = None,
    max_frames: int = None,
    skip_frames: int = 0,
):
    """Process a video through SAM 3D Body MLX.

    Args:
        input_path: input video path
        output_path: output annotated video path
        weights_dir: path to MLX weights
        bbox: fixed [x1,y1,x2,y2] bbox (if None, runs person detection per frame)
        target_region: [x1,y1,x2,y2] hint for which person to track (e.g. mound area).
            On the first frame, picks the detection with highest IoU to this region,
            then tracks that person across subsequent frames.
        max_frames: process at most N frames
        skip_frames: skip every N frames (0 = process all)
    """
    from .estimator import SAM3DBodyEstimator

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{input_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    use_detection = bbox is None
    tracked_bbox = target_region  # seed the tracker with the hint region
    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames to process: {total_frames}")
    print(f"  Person detection: {'per-frame' if use_detection else 'fixed bbox'}")
    if target_region:
        print(f"  Target region: {target_region}")

    # Load model
    print(f"\nLoading model from {weights_dir}...")
    t0 = time.time()
    estimator = SAM3DBodyEstimator(weights_dir)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Warm up person detector if needed
    if use_detection:
        print("Loading person detector...")
        t0 = time.time()
        from .estimator import detect_persons_cached

        # Warm up with a small dummy image
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        detect_persons_cached(dummy)
        det_time = time.time() - t0
        print(f"Detector loaded in {det_time:.1f}s")

    # Output video
    out_fps = fps / (skip_frames + 1) if skip_frames > 0 else fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # Also save per-frame data
    all_keypoints_3d = []
    all_vertices = []
    all_cameras = []
    all_bboxes = []
    frame_times = []

    print(f"\nProcessing frames...")
    frame_idx = 0
    processed = 0
    detection_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= total_frames:
            break

        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Determine bbox for this frame
        if use_detection:
            from .estimator import detect_persons_cached

            detections = detect_persons_cached(rgb)
            frame_bbox = track_person(detections, tracked_bbox)
            if frame_bbox is not None:
                tracked_bbox = frame_bbox  # update tracker state
            else:
                frame_bbox = [0, 0, width, height]
                detection_failures += 1
        else:
            frame_bbox = bbox

        # Run inference with the chosen bbox
        t0 = time.perf_counter()
        result = estimator.predict(rgb, frame_bbox, auto_detect=False)
        inference_time = time.perf_counter() - t0
        frame_times.append(inference_time)

        used_bbox = result.get("bbox", frame_bbox)

        # Project 3D keypoints to 2D
        kp_3d = result["pred_keypoints_3d"]  # (70, 3)
        camera = result["pred_camera"]  # (3,)
        kp_2d = project_keypoints_perspective(kp_3d, camera, used_bbox, width, height)

        # Draw skeleton overlay
        annotated = frame.copy()

        # Draw person bbox if using detection
        if use_detection and used_bbox != [0, 0, width, height]:
            annotated = draw_bbox(annotated, used_bbox)

        annotated = draw_skeleton(annotated, kp_2d)

        # Add timing info
        fps_current = 1.0 / inference_time if inference_time > 0 else 0
        cv2.putText(
            annotated,
            f"MLX: {inference_time*1000:.0f}ms ({fps_current:.1f} fps)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Frame {frame_idx}/{total_frames}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        writer.write(annotated)

        # Store data
        all_keypoints_3d.append(kp_3d)
        all_cameras.append(camera)
        all_bboxes.append(used_bbox)

        processed += 1
        # Progress
        if processed % 10 == 0 or processed == 1:
            avg_ms = np.mean(frame_times[-10:]) * 1000
            eta = (total_frames - frame_idx) * avg_ms / 1000
            print(
                f"  Frame {frame_idx:4d}/{total_frames}  {avg_ms:.0f}ms/frame  ETA: {eta:.0f}s"
            )

        frame_idx += 1

    cap.release()
    writer.release()

    # Summary
    if frame_times:
        times = np.array(frame_times)
        median_ms = np.median(times) * 1000
        total_time = np.sum(times)
        throughput = processed / total_time

        print(f"\n{'='*60}")
        print(f"Video Processing Complete")
        print(f"{'='*60}")
        print(f"  Frames processed: {processed}")
        print(f"  Total inference:  {total_time:.1f}s")
        print(f"  Median per frame: {median_ms:.0f}ms")
        print(f"  Throughput:       {throughput:.1f} fps")
        if use_detection:
            print(f"  Detection fails:  {detection_failures}/{processed}")
        print(f"  Output:           {output_path}")
        print(
            f"  Output size:      {Path(output_path).stat().st_size / 1024**2:.1f} MB"
        )

        # Save keypoints
        kp_path = output_path.rsplit(".", 1)[0] + "_keypoints.npy"
        np.save(kp_path, np.array(all_keypoints_3d))
        print(f"  3D keypoints:     {kp_path} ({len(all_keypoints_3d)} frames)")

    return {
        "frames_processed": processed,
        "total_time": total_time if frame_times else 0,
        "median_ms": median_ms if frame_times else 0,
        "throughput_fps": throughput if frame_times else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="SAM 3D Body video processing (MLX)")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None, help="Output video path")
    parser.add_argument(
        "--weights", default="/tmp/sam3d-mlx-weights/", help="Weights directory"
    )
    parser.add_argument("--bbox", help="Fixed bbox as x1,y1,x2,y2")
    parser.add_argument("--max-frames", type=int, help="Process at most N frames")
    parser.add_argument("--skip", type=int, default=0, help="Skip every N frames")

    args = parser.parse_args()

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"output/{stem}_mlx.mp4"

    bbox = None
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    process_video(
        input_path=args.input,
        output_path=args.output,
        weights_dir=args.weights,
        bbox=bbox,
        max_frames=args.max_frames,
        skip_frames=args.skip,
    )


if __name__ == "__main__":
    main()
