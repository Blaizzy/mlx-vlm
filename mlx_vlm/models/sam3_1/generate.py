"""SAM 3.1 Inference Pipeline.

Overrides predict_multi, track_video, and track_video_realtime to handle
TriViTDetNeck's 3-tuple output (detection, interactive, propagation FPNs).
"""

from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from ..sam3.generate import (
    COLORS_BGR,
    DetectionResult,
    Sam3Predictor,
    _filter_by_regions,
    _resize_masks,
    draw_frame,
    nms,
    run_image,
)


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


def _get_backbone_features(model, pixel_values: mx.array) -> mx.array:
    """Run ViT backbone only (no FPN neck)."""
    features = model.detector_model.vision_encoder.backbone(pixel_values)
    mx.eval(features)
    return features


def _get_det_features(model, backbone_features: mx.array):
    """Run detection FPN neck + flatten for DETR. Returns (src, pos_flat, det_features, spatial)."""
    det = model.detector_model
    det_features, _, _ = det.vision_encoder.neck(
        backbone_features, need_det=True, need_interactive=False, need_propagation=False
    )
    fpn_pos = [det._pos_enc(feat) for feat in det_features]

    encoder_feat = det_features[-1]
    B, H_f, W_f, D = encoder_feat.shape
    src = encoder_feat.reshape(B, H_f * W_f, D)
    pos_flat = fpn_pos[-1].reshape(B, H_f * W_f, D)
    mx.eval(src, pos_flat)
    return src, pos_flat, det_features, (H_f, W_f)


def _run_detr_encoder(model, src, pos_flat, inputs_embeds, attention_mask):
    """Run DETR encoder. Cacheable when backbone + text are unchanged."""
    encoded = model.detector_model.detr_encoder(
        src, pos_flat, inputs_embeds, attention_mask
    )
    mx.eval(encoded)
    return encoded


def _postprocess_mlx(
    pred_logits: mx.array,
    pred_boxes: mx.array,
    pred_masks,
    presence: mx.array,
    image_size,
    threshold: float,
) -> DetectionResult:
    """Postprocess entirely in MLX, single numpy conversion at the end."""
    W, H = (
        image_size if isinstance(image_size, tuple) else (image_size[1], image_size[0])
    )

    # All scoring in MLX
    scores = mx.sigmoid(pred_logits[0].squeeze())
    if presence is not None:
        scores = scores * mx.sigmoid(presence[0])

    # Scale boxes in MLX before converting
    boxes = pred_boxes[0] * mx.array([W, H, W, H], dtype=pred_boxes.dtype)
    boxes = mx.clip(boxes, 0, max(H, W))

    # Single eval, single conversion
    if pred_masks is not None:
        mx.eval(scores, boxes, pred_masks)
        scores_np = np.array(scores)
        keep = scores_np > threshold
        if not keep.any():
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                masks=np.zeros((0, H, W), dtype=np.uint8),
                scores=np.zeros((0,)),
            )
        masks_np = np.array(pred_masks[0])[keep]
        masks_resized = _resize_masks(masks_np, (H, W))
        masks_binary = (masks_resized > 0).astype(np.uint8)
    else:
        mx.eval(scores, boxes)
        scores_np = np.array(scores)
        keep = scores_np > threshold
        if not keep.any():
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                masks=np.zeros((0, H, W), dtype=np.uint8),
                scores=np.zeros((0,)),
            )
        masks_binary = np.zeros((keep.sum(), H, W), dtype=np.uint8)

    return DetectionResult(
        boxes=np.array(boxes)[keep],
        masks=masks_binary,
        scores=scores_np[keep],
    )


def _detect_with_backbone(
    predictor: Sam3Predictor,
    backbone_features: mx.array,
    prompts: List[str],
    image_size,
    threshold: float,
    encoder_cache: Optional[Dict] = None,
) -> DetectionResult:
    """Run detection on pre-computed backbone features.

    With encoder_cache: skips DETR encoder when backbone + text are unchanged.
    All computation stays in MLX until final postprocessing.
    """
    det = predictor.model.detector_model

    # FPN neck (cheap ~3ms)
    src, pos_flat, det_features, spatial = _get_det_features(
        predictor.model, backbone_features
    )
    H_f, W_f = spatial

    all_boxes, all_masks, all_scores, all_labels = [], [], [], []

    for prompt in prompts:
        inputs_embeds, attention_mask = predictor._get_input_embeddings(prompt)

        # DETR encoder — use cache if available (caller controls invalidation)
        cached = encoder_cache.get(prompt) if encoder_cache is not None else None
        if cached is not None:
            encoded = cached["encoded"]
        else:
            encoded = _run_detr_encoder(
                predictor.model, src, pos_flat, inputs_embeds, attention_mask
            )
            if encoder_cache is not None:
                encoder_cache[prompt] = {"encoded": encoded}

        # DETR decoder
        hs, ref_boxes, presence_logits = det.detr_decoder(
            vision_features=encoded,
            inputs_embeds=inputs_embeds,
            vision_pos_encoding=pos_flat,
            text_mask=attention_mask,
            spatial_shape=(H_f, W_f),
        )

        # Box conversion in MLX (no numpy)
        pred_boxes_cxcywh = ref_boxes[-1]
        cx, cy, w, h = (
            pred_boxes_cxcywh[..., 0],
            pred_boxes_cxcywh[..., 1],
            pred_boxes_cxcywh[..., 2],
            pred_boxes_cxcywh[..., 3],
        )
        pred_boxes_xyxy = mx.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1
        )

        # Scoring in MLX
        all_logits = det.dot_product_scoring(hs, inputs_embeds, attention_mask)
        pred_logits = all_logits[-1].squeeze(-1)
        presence = presence_logits[-1]

        # Mask decoder
        last_hs = hs[-1]
        seg_out = det.mask_decoder(
            last_hs,
            list(det_features),
            encoder_hidden_states=encoded,
            prompt_features=inputs_embeds,
            prompt_mask=attention_mask,
        )

        # Single eval for all outputs
        mx.eval(pred_logits, pred_boxes_xyxy, seg_out, presence)

        # Postprocess — MLX until final conversion
        result = _postprocess_mlx(
            pred_logits if pred_logits.ndim == 2 else pred_logits[None],
            pred_boxes_xyxy if pred_boxes_xyxy.ndim == 3 else pred_boxes_xyxy[None],
            seg_out["pred_masks"],
            presence if presence.ndim == 2 else presence[None],
            image_size,
            threshold,
        )
        if len(result.scores) > 0:
            result = nms(result)
            all_boxes.append(result.boxes)
            all_masks.append(result.masks)
            all_scores.append(result.scores)
            all_labels.extend([prompt] * len(result.scores))

    if not all_scores:
        W, H = (
            image_size
            if isinstance(image_size, tuple)
            else (image_size[1], image_size[0])
        )
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            masks=np.zeros((0, H, W), dtype=np.uint8),
            scores=np.zeros((0,)),
            labels=[],
        )

    return DetectionResult(
        boxes=np.concatenate(all_boxes),
        masks=np.concatenate(all_masks),
        scores=np.concatenate(all_scores),
        labels=all_labels,
    )


def _init_tracker_memory(
    model,
    backbone_features: mx.array,
    detection_masks: List[np.ndarray],
) -> List[mx.array]:
    """Initialize tracker memory bank from detection masks."""
    prop_fpn = model._get_tracker_features(backbone_features)
    track_features = prop_fpn[2]  # 1x scale
    feat_H, feat_W = track_features.shape[1], track_features.shape[2]

    multiplex_count = model.config.tracker_config.multiplex_count
    n_ch = multiplex_count * 2
    # Mask downsampler has stride 16 total, so input must be feat_size * 16
    target = feat_H * 16

    # Build combined multiplex mask in MLX
    N = min(len(detection_masks), multiplex_count)
    # Stack masks → (N, H_mask, W_mask), resize once
    masks_mx = mx.array(
        np.stack(detection_masks[:N]).astype(np.float32)
    )  # single conversion
    mask_h, mask_w = masks_mx.shape[1], masks_mx.shape[2]
    if mask_h != target or mask_w != target:
        up = nn.Upsample(
            scale_factor=(target / mask_h, target / mask_w), mode="nearest"
        )
        masks_mx = up(masks_mx[:, :, :, None])[:, :, :, 0]  # (N, target, target)

    # Pack into multiplex channels in MLX
    channels = []
    for ch in range(n_ch):
        slot = ch // 2
        if slot < N:
            if ch % 2 == 0:
                channels.append(masks_mx[slot : slot + 1, :, :, None])  # mask
            else:
                channels.append(1.0 - masks_mx[slot : slot + 1, :, :, None])  # inverse
        else:
            channels.append(mx.zeros((1, target, target, 1)))
    # (multiplex*2, target, target, 1) → (1, target, target, n_ch)
    mask_mx = mx.concatenate(channels, axis=0)  # (n_ch, target, target, 1)
    mask_mx = mask_mx[:, :, :, 0].transpose(1, 2, 0)[None]  # (1, target, target, n_ch)
    memory = model.tracker_model.memory_encoder(track_features, mask_mx)
    mx.eval(memory)
    _, H_m, W_m, C = memory.shape
    return [memory.reshape(1, H_m * W_m, C)]


def _propagate_tracker(
    model,
    backbone_features: mx.array,
    memory_bank: List[mx.array],
    n_objects: int,
    image_size,
) -> tuple:
    """Run tracker propagation. Returns (DetectionResult, updated_memory_bank)."""
    prop_fpn = model._get_tracker_features(backbone_features)
    track_features = prop_fpn[2]
    high_res = [prop_fpn[0], prop_fpn[1]] if len(prop_fpn) > 1 else None

    result = model.tracker_model.track_step(
        current_features=track_features,
        memory_bank=memory_bank,
        multimask_output=False,
        high_res_features=high_res,
    )
    mx.eval(result)

    # Convert tracker output to DetectionResult — stay in MLX as long as possible
    pred_masks = result["pred_masks"]  # (B, M, num_masks, H, W) or (B, num_masks, H, W)
    iou_scores = result["iou_scores"]

    W, H = (
        image_size if isinstance(image_size, tuple) else (image_size[1], image_size[0])
    )
    N = min(n_objects, 16)

    # Extract per-object masks and scores in MLX (batched, no per-object np.array)
    if pred_masks.ndim == 5:
        # (B, M, num_masks, H_out, W_out) → take best mask per object
        obj_masks = pred_masks[0, :N, 0]  # (N, H_out, W_out) — still MLX
        obj_scores = iou_scores[0, :N, 0]  # (N,)
    else:
        obj_masks = mx.broadcast_to(pred_masks[0, 0:1], (N,) + pred_masks.shape[2:])
        obj_scores = mx.broadcast_to(iou_scores[0, 0:1], (N,))

    # Resize masks in MLX using Upsample (batched, no per-object PIL)
    mask_h, mask_w = obj_masks.shape[1], obj_masks.shape[2]
    if mask_h != H or mask_w != W:
        up = nn.Upsample(scale_factor=(H / mask_h, W / mask_w), mode="nearest")
        obj_masks_up = up(obj_masks[:, :, :, None])[:, :, :, 0]  # (N, H, W)
    else:
        obj_masks_up = obj_masks

    # Single eval + conversion
    mx.eval(obj_masks_up, obj_scores)
    masks_np = (np.array(obj_masks_up) > 0).astype(np.uint8)
    scores_np = np.array(obj_scores)

    # Derive boxes from masks (numpy — needed for contour-based boxes)
    boxes_list = []
    for i in range(N):
        ys, xs = np.where(masks_np[i])
        if len(ys) > 0:
            boxes_list.append([xs.min(), ys.min(), xs.max(), ys.max()])
        else:
            boxes_list.append([0, 0, 0, 0])

    det_result = DetectionResult(
        boxes=np.array(boxes_list, dtype=np.float32),
        masks=masks_np,
        scores=scores_np,
        labels=[],
    )

    # Update memory bank
    multiplex_count = model.config.tracker_config.multiplex_count
    n_ch = multiplex_count * 2
    feat_H = track_features.shape[1]
    target = feat_H * 16  # mask_downsampler has stride 16

    channels = []
    for ch in range(n_ch):
        slot = ch // 2
        is_inv = ch % 2 == 1
        if slot < n_objects and pred_masks.ndim == 5:
            slot_mask = pred_masks[:, slot, 0]  # (B, H_out, W_out)
            up = nn.Upsample(
                scale_factor=(target / slot_mask.shape[1], target / slot_mask.shape[2]),
                mode="nearest",
            )
            sig = mx.sigmoid(up(slot_mask[:, :, :, None])[:, :, :, 0] * 20.0 - 10.0)
            if is_inv:
                channels.append((1.0 - sig)[:, :, :, None])
            else:
                channels.append(sig[:, :, :, None])
        else:
            channels.append(mx.zeros((1, target, target, 1)))

    mask_for_mem = mx.concatenate(channels, axis=-1)
    memory = model.tracker_model.memory_encoder(track_features, mask_for_mem)
    mx.eval(memory)
    B_m, H_m, W_m, C_m = memory.shape
    new_mem = memory.reshape(1, H_m * W_m, C_m)

    max_mem = model.config.tracker_config.num_maskmem
    updated_bank = memory_bank + [new_mem]
    if len(updated_bank) > max_mem:
        updated_bank = updated_bank[-max_mem:]

    return det_result, updated_bank


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
        boxes=np.array([]),
        masks=np.array([]),
        scores=np.array([]),
        labels=[],
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
                print(
                    f"  Frame {fi}/{total_frames}: {len(latest_result.scores)} detections"
                )

        out = draw_frame(
            frame_bgr,
            latest_result.masks,
            latest_result.scores,
            latest_result.boxes,
            prompt_str,
            H,
            W,
            show_boxes=show_boxes,
            labels=latest_result.labels,
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
    detect_every: int = 15,
    recompute_backbone_every: int = 30,
    update_memory_every: int = 3,
):
    """Optimized realtime tracking with backbone caching + tracker propagation.

    Two key optimizations over the basic detect-per-frame approach:
    1. Backbone caching: skip ViT on intermediate frames (~67ms saved)
    2. Tracker propagation: use memory attention + mask decoder instead of DETR
       (only at native 1008px resolution; lower resolutions use cached DETR)

    Timeline (detect_every=15, recompute_backbone_every=5):
      Frame 0:  DETECT + ViT       ~92ms
      Frame 1-4: PROPAGATE cached   ~16ms each (1008px) or DETR cached ~25ms
      Frame 5:  PROPAGATE + ViT     ~83ms
      Frame 6-9: PROPAGATE cached   ~16ms each
      ...
      Frame 15: DETECT + ViT        ~92ms  (catch new objects)
    """
    import queue
    import threading
    import time

    import cv2

    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.utils import get_model_path, load_model

    # Parse box prompts
    box_array = None
    if boxes is not None:
        box_list = []
        for b in boxes.split(";"):
            coords = [float(x) for x in b.split(",")]
            if len(coords) == 4:
                box_list.append(coords)
        if box_list:
            box_array = np.array(box_list)

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam31Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    is_camera = str(video_path).isdigit()
    cap = cv2.VideoCapture(int(video_path) if is_camera else video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prompt_str = " + ".join(prompts)
    print(f"Video: {fps:.1f} fps, {W}x{H}")
    print(
        f"Prompts: {prompts}, threshold {threshold}, resolution {resolution}x{resolution}"
    )
    print(
        f"Optimized: detect every {detect_every}, ViT every {recompute_backbone_every}, "
        f"memory update every {update_memory_every}"
    )

    # Load background image for bg swap mode
    bg_frame = None
    if bg_image is not None:
        bg_pil = Image.open(bg_image).convert("RGB").resize((W, H))
        bg_frame = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

    print("Press 'q' to quit")

    frame_buffer = queue.Queue(maxsize=10)

    # Shared state between threads
    lock = threading.Lock()
    latest = {
        "overlay_scaled": np.zeros((H, W, 3), dtype=np.uint8),
        "fg_mask": None,
        "has_detections": False,
        "n_obj": 0,
        "fps": 0.0,
        "mode": "init",  # "detect" or "track"
    }
    pending_frame = {"pil": None}
    running = {"active": True}

    # --- Thread 1: Read frames ---
    frame_interval = 1.0 / fps if not is_camera else 0

    def reader_loop():
        next_frame_time = time.perf_counter()
        while running["active"]:
            if not is_camera:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(max(0, next_frame_time - now - 0.001))
                    continue

            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame_time = time.perf_counter()
                continue

            if frame_buffer.full():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            frame_buffer.put(frame)

            if not is_camera:
                next_frame_time += frame_interval

    # --- Thread 2: Inference (backbone caching + tracker propagation) ---
    def inference_loop():
        backbone_cache = {"features": None}
        encoder_cache = {}
        tracker_state = {"memory_bank": [], "n_objects": 0, "labels": []}
        inference_count = 0
        prop_count = 0

        while running["active"]:
            with lock:
                frame_pil = pending_frame["pil"]

            if frame_pil is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()
            image_size = frame_pil.size

            inputs = predictor.processor.preprocess_image(frame_pil)
            pixel_values = mx.array(inputs["pixel_values"])

            # Step 1: Backbone — always use pre-computed from previous frame
            if backbone_cache["features"] is None:
                backbone_cache["features"] = _get_backbone_features(model, pixel_values)
            backbone_features = backbone_cache["features"]

            # Step 2: Detect or propagate
            can_track = (
                resolution >= 1008
                and tracker_state["memory_bank"]
                and tracker_state["n_objects"] > 0
            )
            need_detect = (
                inference_count % detect_every == 0
                or not tracker_state["memory_bank"]
                or not can_track  # always detect at non-native resolutions
            )

            if need_detect:
                encoder_cache.clear()  # fresh detection needs fresh encoder
                result = _detect_with_backbone(
                    predictor,
                    backbone_features,
                    prompts,
                    image_size,
                    threshold,
                    encoder_cache=encoder_cache,
                )
                if box_array is not None and len(result.scores) > 0:
                    result = _filter_by_regions(result, box_array)

                mode = "detect"

                # Initialize tracker memory from detections (only at native res)
                if len(result.scores) > 0 and resolution >= 1008:
                    tracker_state["memory_bank"] = _init_tracker_memory(
                        model, backbone_features, list(result.masks)
                    )
                    tracker_state["n_objects"] = len(result.scores)
                    tracker_state["labels"] = (
                        result.labels
                        if result.labels
                        else [prompt_str] * len(result.scores)
                    )
                    prop_count = 0
                elif len(result.scores) > 0:
                    # At lower resolutions, just carry labels forward
                    tracker_state["labels"] = (
                        result.labels
                        if result.labels
                        else [prompt_str] * len(result.scores)
                    )
            else:
                # Tracker propagation (fast path — native resolution only)
                result, updated_bank = _propagate_tracker(
                    model,
                    backbone_features,
                    tracker_state["memory_bank"],
                    tracker_state["n_objects"],
                    image_size,
                )
                result.labels = tracker_state["labels"]

                # Update memory periodically
                prop_count += 1
                if prop_count % update_memory_every == 0:
                    tracker_state["memory_bank"] = updated_bank

                mode = "track"

            dt = time.perf_counter() - t0

            # Pre-render overlay — fast path: boolean indexing, no contours
            overlay = np.zeros((H, W, 3), dtype=np.uint8)
            fg_mask = None
            if len(result.scores) > 0:
                has_masks = result.masks.any()

                # Batch mask overlay: boolean index (11x faster than np.where + contours)
                if has_masks:
                    for i in range(len(result.scores)):
                        mask = result.masks[i]
                        if mask.shape[0] != H or mask.shape[1] != W:
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (W, H),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        overlay[mask > 0] = COLORS_BGR[i % len(COLORS_BGR)]

                    # Build foreground mask for bg swap
                    if bg_frame is not None:
                        fg_mask = np.any(result.masks > 0, axis=0).astype(np.uint8)
                        if fg_mask.shape[0] != H or fg_mask.shape[1] != W:
                            fg_mask = cv2.resize(
                                fg_mask, (W, H), interpolation=cv2.INTER_NEAREST
                            )

                # Boxes + labels
                for i in range(len(result.scores)):
                    color = COLORS_BGR[i % len(COLORS_BGR)]
                    lbl = (
                        result.labels[i]
                        if result.labels and i < len(result.labels)
                        else prompt_str
                    )
                    label = f"{lbl} {result.scores[i]:.2f}"

                    x1, y1, x2, y2 = result.boxes[i].astype(int)
                    if show_boxes or not has_masks:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    lx, ly = x1, max(y1 - 8, 12)

                    (tw, th_t), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        overlay,
                        (lx, max(ly - th_t - 6, 0)),
                        (lx + tw + 6, ly + 4),
                        color,
                        -1,
                    )
                    cv2.putText(
                        overlay,
                        label,
                        (lx + 3, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            scaled = (overlay.astype(np.uint16) * 115 >> 8).astype(np.uint8)

            # Pre-compute backbone for next refresh frame
            # This hides the ViT latency: detection always uses ready features
            next_count = inference_count + 1
            if (
                next_count % recompute_backbone_every == 0
                or next_count % detect_every == 0
            ):
                backbone_cache["features"] = _get_backbone_features(model, pixel_values)

            with lock:
                latest["overlay_scaled"] = scaled
                latest["fg_mask"] = fg_mask
                latest["has_detections"] = len(result.scores) > 0
                latest["n_obj"] = len(result.scores)
                latest["fps"] = 1.0 / max(dt, 1e-6)
                latest["mode"] = mode
                pending_frame["pil"] = None

            inference_count += 1

    # Start threads
    reader_thread = threading.Thread(target=reader_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    reader_thread.start()
    inference_thread.start()

    display_fps_counter = 0
    display_fps_t0 = time.perf_counter()
    display_fps_val = 0.0

    while True:
        try:
            frame_bgr = frame_buffer.get(timeout=0.05)
        except queue.Empty:
            continue

        with lock:
            if pending_frame["pil"] is None:
                pending_frame["pil"] = Image.fromarray(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                )

        with lock:
            overlay_scaled = latest["overlay_scaled"]
            fg_mask = latest["fg_mask"]
            has_det = latest["has_detections"]
            det_fps = latest["fps"]
            n_obj = latest["n_obj"]
            mode = latest["mode"]

        if bg_frame is not None and fg_mask is not None:
            fg_mask_3d = fg_mask[:, :, None]
            out = np.where(fg_mask_3d, frame_bgr, bg_frame)
        elif has_det:
            out = cv2.add(frame_bgr, overlay_scaled)
        else:
            out = frame_bgr

        display_fps_counter += 1
        now = time.perf_counter()
        if now - display_fps_t0 >= 0.5:
            display_fps_val = display_fps_counter / (now - display_fps_t0)
            display_fps_counter = 0
            display_fps_t0 = now

        cv2.putText(
            out,
            f"{mode.upper()}: {det_fps:.1f} FPS | Display: {display_fps_val:.0f} FPS | {n_obj} obj",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow(f"SAM3.1 Tracking - {prompt_str}", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    running["active"] = False
    inference_thread.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print("Done")


def main():
    """SAM 3.1 CLI — same interface as SAM 3 with optimization flags."""
    import argparse

    parser = argparse.ArgumentParser(description="SAM 3.1 inference")
    parser.add_argument(
        "--task", default="segment", choices=["detect", "segment", "track", "realtime"]
    )
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
    # Optimization parameters
    parser.add_argument(
        "--detect-every",
        type=int,
        default=15,
        help="Re-run DETR detection every N inference frames (default: 15)",
    )
    parser.add_argument(
        "--backbone-every",
        type=int,
        default=30,
        help="Re-run ViT backbone every N inference frames (default: 30)",
    )
    parser.add_argument(
        "--memory-every",
        type=int,
        default=3,
        help="Update tracker memory every N propagation frames (default: 3)",
    )
    args = parser.parse_args()

    if args.task in ("track", "realtime"):
        video = args.video or "0"
        if args.task == "track":
            track_video(
                video,
                args.prompt,
                output=args.output,
                model_path=args.model,
                threshold=args.threshold,
                nms_thresh=args.nms_thresh,
                every=args.every,
                boxes=args.boxes,
                show_boxes=args.show_boxes,
                resolution=args.resolution,
            )
        else:
            track_video_realtime(
                video,
                args.prompt,
                model_path=args.model,
                threshold=args.threshold,
                nms_thresh=args.nms_thresh,
                boxes=args.boxes,
                show_boxes=args.show_boxes,
                resolution=args.resolution,
                bg_image=args.bg_image,
                detect_every=args.detect_every,
                recompute_backbone_every=args.backbone_every,
                update_memory_every=args.memory_every,
            )
    else:
        run_image(
            args.image,
            args.prompt,
            task=args.task,
            model_path=args.model,
            output=args.output,
            threshold=args.threshold,
            boxes=args.boxes,
            show_boxes=args.show_boxes,
        )


if __name__ == "__main__":
    main()
