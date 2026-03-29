"""SAM 3.1 Inference Pipeline — reuses SAM 3 generate.py.

Same API: Sam3Predictor, Sam3VideoPredictor, CLI.
Overrides predict_multi, track_video, and track_video_realtime to handle
TriViTDetNeck's 3-tuple output (detection, interactive, propagation FPNs).
"""

from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

# Re-export unchanged utilities from SAM 3
from ..sam3.generate import (
    COLORS_BGR,
    DetectionResult,
    Sam3Predictor,
    Sam3VideoPredictor,
    TrackingResult,
    _filter_by_regions,
    _resize_masks,
    _sigmoid,
    draw_frame,
    nms,
    run_image,
)


# ---------------------------------------------------------------------------
# SAM 3.1 predict_multi — handles TriViTDetNeck's 3-tuple output
# ---------------------------------------------------------------------------


def predict_multi(
    predictor: Sam3Predictor,
    image,
    prompts: List[str],
    boxes: Optional[np.ndarray] = None,
    score_threshold: Optional[float] = None,
) -> DetectionResult:
    """Run vision backbone ONCE, then text+DETR per prompt. Merge with labels.

    SAM 3.1 override: handles TriViTDetNeck's (det, interactive, propagation) output.
    For N prompts, cost = 1x ViT + Nx (text + DETR) instead of Nx full pipeline.
    """
    if len(prompts) == 1:
        result = predictor.predict(
            image,
            text_prompt=prompts[0],
            boxes=boxes,
            score_threshold=score_threshold,
        )
        if len(result.scores) > 0:
            result = nms(result)
            result.labels = [prompts[0]] * len(result.scores)
        else:
            result.labels = []
        return result

    # Run vision backbone once — TriViTDetNeck returns 3-tuple
    inputs = predictor.processor.preprocess_image(image)
    pixel_values = mx.array(inputs["pixel_values"])

    det = predictor.model.detector_model
    vision_out = det.vision_encoder(
        pixel_values, need_det=True, need_interactive=False, need_propagation=False
    )

    # TriViTDetNeck returns (det_features, interactive_features, propagation_features)
    if isinstance(vision_out, tuple):
        det_features = vision_out[0]
    else:
        det_features = vision_out

    fpn_pos = [det._pos_enc(feat) for feat in det_features]

    # SAM 3.1: 3 scales, no trimming needed (no 0.5x level)
    fpn_trimmed = det_features
    fpn_pos_trimmed = fpn_pos

    encoder_feat = fpn_trimmed[-1]  # 1x scale (72x72)
    B, H_f, W_f, D = encoder_feat.shape
    src = encoder_feat.reshape(B, H_f * W_f, D)
    pos_flat = fpn_pos_trimmed[-1].reshape(B, H_f * W_f, D)
    mx.eval(src, pos_flat)

    threshold = score_threshold or predictor.score_threshold
    image_size = image.size if hasattr(image, "size") else image.shape[:2]

    all_boxes, all_masks, all_scores, all_labels = [], [], [], []

    for prompt in prompts:
        inputs_embeds, attention_mask = predictor._get_input_embeddings(prompt)

        encoded = det.detr_encoder(src, pos_flat, inputs_embeds, attention_mask)
        mx.eval(encoded)

        hs, ref_boxes, presence_logits = det.detr_decoder(
            vision_features=encoded,
            inputs_embeds=inputs_embeds,
            vision_pos_encoding=pos_flat,
            text_mask=attention_mask,
            spatial_shape=(H_f, W_f),
        )

        pred_boxes_cxcywh = ref_boxes[-1]
        cx = pred_boxes_cxcywh[..., 0]
        cy = pred_boxes_cxcywh[..., 1]
        w = pred_boxes_cxcywh[..., 2]
        h = pred_boxes_cxcywh[..., 3]
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        all_logits = det.dot_product_scoring(hs, inputs_embeds, attention_mask)
        pred_logits = all_logits[-1].squeeze(-1)
        presence = presence_logits[-1]

        last_hs = hs[-1]
        seg_out = det.mask_decoder(
            last_hs,
            list(fpn_trimmed),
            encoder_hidden_states=encoded,
            prompt_features=inputs_embeds,
            prompt_mask=attention_mask,
        )
        mx.eval(pred_logits, pred_boxes_xyxy, seg_out, presence)

        outputs = {
            "pred_logits": pred_logits if pred_logits.ndim == 2 else pred_logits[None],
            "pred_boxes": (
                pred_boxes_xyxy if pred_boxes_xyxy.ndim == 3 else pred_boxes_xyxy[None]
            ),
            "pred_masks": seg_out["pred_masks"],
            "presence_logits": presence if presence.ndim == 2 else presence[None],
        }
        result = predictor._postprocess(outputs, image_size, threshold)
        if len(result.scores) > 0:
            result = nms(result)
            all_boxes.append(result.boxes)
            all_masks.append(result.masks)
            all_scores.append(result.scores)
            all_labels.extend([prompt] * len(result.scores))

    if not all_scores:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            masks=np.zeros((0, 1, 1), dtype=np.uint8),
            scores=np.zeros((0,)),
            labels=[],
        )

    return DetectionResult(
        boxes=np.concatenate(all_boxes),
        masks=np.concatenate(all_masks),
        scores=np.concatenate(all_scores),
        labels=all_labels,
    )


# ---------------------------------------------------------------------------
# track_video / track_video_realtime — use SAM 3.1's predict_multi
# ---------------------------------------------------------------------------


def track_video(
    video_path: str,
    prompts: List[str],
    output: Optional[str] = None,
    model_path: str = "facebook/sam3.1",
    threshold: float = 0.15,
    nms_thresh: float = 0.5,
    every: int = 2,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
    resolution: int = 1008,
):
    """Track objects in a video file using SAM 3.1.

    Uses SAM 3.1's predict_multi which handles TriViTDetNeck output.
    """
    import cv2

    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import get_model_path, load_model

    if output is None:
        p = Path(video_path)
        output = str(p.parent / f"{p.stem}_tracked{p.suffix}")

    box_array = None
    if boxes is not None:
        box_list = []
        for b in boxes.split(";"):
            coords = [float(x) for x in b.split(",")]
            if len(coords) == 4:
                box_list.append(coords)
        if box_list:
            box_array = np.array(box_list)
            print(f"Box prompts: {box_array.tolist()}")

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam31Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")
    prompt_str = " + ".join(prompts)
    print(f"Prompts: {prompts}, detect every {every} frames, threshold {threshold}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (W, H))

    latest_result = DetectionResult(
        boxes=np.array([]), masks=np.array([]),
        scores=np.array([]), labels=[],
    )

    for fi in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if fi % every == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = predict_multi(predictor, frame_pil, prompts)
            if box_array is not None and len(result.scores) > 0:
                result = _filter_by_regions(result, box_array)
            latest_result = result

            if fi % 40 == 0:
                print(f"  Frame {fi}/{total_frames}: {len(latest_result.scores)} detections")

        out = draw_frame(
            frame_bgr, latest_result.masks, latest_result.scores,
            latest_result.boxes, prompt_str, H, W,
            show_boxes=show_boxes, labels=latest_result.labels,
        )
        writer.write(out)

    writer.release()
    cap.release()
    print(f"\nSaved: {output}")


def track_video_realtime(
    video_path: str,
    prompts: List[str],
    model_path: str = "facebook/sam3.1",
    threshold: float = 0.15,
    nms_thresh: float = 0.5,
    boxes: Optional[str] = None,
    show_boxes: bool = True,
    resolution: int = 1008,
    bg_image: Optional[str] = None,
):
    """Track objects with a real-time preview using SAM 3.1.

    Uses SAM 3.1's predict_multi which handles TriViTDetNeck output.
    Delegates to SAM 3's realtime loop but with the correct predict_multi.
    """
    # Import the realtime implementation and monkey-patch predict_multi
    import mlx_vlm.models.sam3.generate as sam3_gen

    original_predict_multi = sam3_gen.predict_multi
    sam3_gen.predict_multi = predict_multi
    try:
        sam3_gen.track_video_realtime(
            video_path=video_path,
            prompts=prompts,
            model_path=model_path,
            threshold=threshold,
            nms_thresh=nms_thresh,
            boxes=boxes,
            show_boxes=show_boxes,
            resolution=resolution,
            bg_image=bg_image,
        )
    finally:
        sam3_gen.predict_multi = original_predict_multi


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """SAM 3.1 CLI — same interface as SAM 3."""
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3.1 inference")
    parser.add_argument("--task", default="segment",
                        choices=["detect", "segment", "track", "realtime"])
    parser.add_argument("--image", type=str)
    parser.add_argument("--video", type=str)
    parser.add_argument("--prompt", nargs="+", required=True)
    parser.add_argument("--model", default="facebook/sam3.1")
    parser.add_argument("--output", type=str)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--boxes", type=str)
    parser.add_argument("--show-boxes", action="store_true", default=True)
    parser.add_argument("--no-show-boxes", dest="show_boxes", action="store_false")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--bg-image", type=str)
    args = parser.parse_args()

    if args.task in ("track", "realtime"):
        video = args.video or "0"
        if args.task == "track":
            track_video(
                video, args.prompt, output=args.output, model_path=args.model,
                threshold=args.threshold, nms_thresh=args.nms_thresh,
                every=args.every, boxes=args.boxes, show_boxes=args.show_boxes,
                resolution=args.resolution,
            )
        else:
            track_video_realtime(
                video, args.prompt, model_path=args.model,
                threshold=args.threshold, nms_thresh=args.nms_thresh,
                boxes=args.boxes, show_boxes=args.show_boxes,
                resolution=args.resolution, bg_image=args.bg_image,
            )
    else:
        run_image(
            args.image, args.prompt, task=args.task, model_path=args.model,
            output=args.output, threshold=args.threshold, boxes=args.boxes,
            show_boxes=args.show_boxes,
        )


if __name__ == "__main__":
    main()
