"""SAM 3.1 Inference Pipeline — reuses SAM 3 generate.py.

Same API: Sam3Predictor, Sam3VideoPredictor, CLI.
Overrides predict_multi to handle TriViTDetNeck's 3-tuple output.
"""

from typing import List, Optional

import mlx.core as mx
import numpy as np

# Re-export everything from SAM 3
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
    main,
    nms,
    run_image,
    track_video,
    track_video_realtime,
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
    vision_out = det.vision_encoder(pixel_values, need_det=True, need_interactive=False, need_propagation=False)

    # Handle both SAM 3 (flat list) and SAM 3.1 (3-tuple) vision encoder output
    if isinstance(vision_out, tuple):
        det_features = vision_out[0]  # Only need detection FPN
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
