"""Falcon Perception grounding tools for the grounded reasoning agent."""

import numpy as np
from PIL import Image


def _image_region(cx_norm, cy_norm):
    """Map a normalised centroid to a human-readable region string."""
    if cx_norm < 0.33:
        h = "left"
    elif cx_norm < 0.67:
        h = "center"
    else:
        h = "right"

    if cy_norm < 0.33:
        v = "top"
    elif cy_norm < 0.67:
        v = "middle"
    else:
        v = "bottom"

    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return h
    return f"{v}-{h}"


def _detection_to_metadata(det, orig_h, orig_w, mask_id):
    """Convert an FP detection dict to the agent metadata format.

    Returns a dict with id, area_fraction, centroid_norm, bbox_norm,
    image_region, and mask_np.  Returns None for empty detections.
    """
    cx = det["xy"]["x"]
    cy = det["xy"]["y"]
    bw = det["hw"]["w"]
    bh = det["hw"]["h"]

    x1 = max(0.0, cx - bw / 2)
    y1 = max(0.0, cy - bh / 2)
    x2 = min(1.0, cx + bw / 2)
    y2 = min(1.0, cy + bh / 2)

    if "mask" in det:
        mask_np = np.array(det["mask"]).astype(bool)
        if not mask_np.any():
            return None
        # True pixel centroid — more accurate than bbox midpoint
        yx = np.argwhere(mask_np)
        cy = float(yx[:, 0].mean()) / orig_h
        cx = float(yx[:, 1].mean()) / orig_w
        area_fraction = round(float(mask_np.sum()) / (orig_h * orig_w), 4)
    else:
        mask_np = None
        area_fraction = round(float(bw * bh), 4)

    return {
        "id": mask_id,
        "area_fraction": area_fraction,
        "centroid_norm": {"x": round(cx, 4), "y": round(cy, 4)},
        "bbox_norm": {
            "x1": round(x1, 4),
            "y1": round(y1, 4),
            "x2": round(x2, 4),
            "y2": round(y2, 4),
        },
        "image_region": _image_region(cx, cy),
        "mask_np": mask_np,
    }


def run_ground_expression(
    fp_model, fp_processor, image, expression, max_new_tokens=1024
):
    """Run Falcon Perception on image with expression.

    Returns a dict mapping 1-indexed mask IDs to metadata dicts.
    Empty or invalid detections are silently dropped.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    detections = fp_model.generate_perception(
        fp_processor,
        image=image,
        query=expression,
        max_new_tokens=max_new_tokens,
    )

    masks = {}
    for i, det in enumerate(detections, start=1):
        meta = _detection_to_metadata(det, orig_h, orig_w, mask_id=i)
        if meta is not None:
            masks[i] = meta

    return masks


def compute_relations(masks, mask_ids):
    """Compute pairwise spatial relationships between mask_ids.

    Returns a dict with a 'pairs' key containing per-pair IoU,
    relative positions, size ratio, and centroid distance.
    """
    valid_ids = [mid for mid in mask_ids if mid in masks]
    if len(valid_ids) < 2:
        return {
            "note": (
                f"Need at least 2 valid mask IDs for pairwise relations. "
                f"Requested: {mask_ids}, available: {sorted(masks.keys())}"
            )
        }

    pairs = {}
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            a_id, b_id = valid_ids[i], valid_ids[j]
            a, b = masks[a_id], masks[b_id]

            # IoU from binary masks — no pycocotools needed
            if a["mask_np"] is not None and b["mask_np"] is not None:
                ma, mb = a["mask_np"], b["mask_np"]
                intersection = float((ma & mb).sum())
                union = float((ma | mb).sum())
                iou = round(intersection / union, 4) if union > 0 else 0.0
            else:
                iou = 0.0

            cx_a = a["centroid_norm"]["x"]
            cy_a = a["centroid_norm"]["y"]
            cx_b = b["centroid_norm"]["x"]
            cy_b = b["centroid_norm"]["y"]
            dist = round(((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5, 4)

            area_a = a["area_fraction"]
            area_b = b["area_fraction"]
            size_ratio = round(area_a / area_b, 3) if area_b > 0 else None

            pairs[f"{a_id}_vs_{b_id}"] = {
                "iou": iou,
                f"{a_id}_left_of_{b_id}": cx_a < cx_b,
                f"{a_id}_above_{b_id}": cy_a < cy_b,
                f"{a_id}_larger_than_{b_id}": area_a > area_b,
                f"size_ratio_{a_id}_over_{b_id}": size_ratio,
                "centroid_distance_norm": dist,
            }

    return {"pairs": pairs}


def masks_to_json(masks):
    """Return a JSON-serialisable list of mask metadata, omitting mask_np."""
    result = []
    for mask_id in sorted(masks.keys()):
        m = masks[mask_id]
        entry = {
            "id": m["id"],
            "area_fraction": m["area_fraction"],
            "centroid_norm": m["centroid_norm"],
            "bbox_norm": m["bbox_norm"],
            "image_region": m["image_region"],
        }
        if "slot" in m:
            entry["slot"] = m["slot"]
        result.append(entry)
    return result
