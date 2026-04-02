from __future__ import annotations

import gc
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

_MIN_CROP_DIM = 16

LAYOUT_TO_OCR_CATEGORY: dict[str, str | None] = {
    "text": "text",
    "table": "table",
    "formula": "formula",
    "caption": "caption",
    "footnote": "footnote",
    "list-item": "list-item",
    "title": "title",
    "header": "text",
    "footer": "page-footer",
    "number": "text",
    "figure_title": "caption",
    "paragraph_title": "section-header",
    "doc_title": "title",
    "reference_content": "text",
    "reference": "text",
    "abstract": "text",
    "aside_text": "text",
    "content": "text",
    "formula_number": "text",
    "vision_footnote": "footnote",
    "algorithm": "text",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "section-header": "section-header",
    "image": None,
    "picture": None,
    "figure": None,
    "chart": None,
    "seal": None,
}


def _box_area(bbox: list[float]) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection_area(a: list[float], b: list[float]) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )


def _containment_ratio(small: list[float], large: list[float]) -> float:
    area = _box_area(small)
    if area <= 0:
        return 0.0
    return _intersection_area(small, large) / area


def filter_nested_detections(
    detections: list[dict],
    containment_threshold: float = 0.8,
) -> list[dict]:
    areas = [_box_area(d["bbox"]) for d in detections]
    keep = []
    for i, det in enumerate(detections):
        is_nested = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            if areas[j] <= areas[i]:
                continue
            if _containment_ratio(det["bbox"], other["bbox"]) > containment_threshold:
                is_nested = True
                break
        if not is_nested:
            keep.append(det)
    return keep


def crop_region(
    image: Image.Image,
    bbox: list[float],
    min_dim: int = _MIN_CROP_DIM,
    max_dim: int = 1024,
) -> Optional[Image.Image]:
    w, h = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cw, ch = x2 - x1, y2 - y1
    if cw < min_dim or ch < min_dim:
        return None
    short, long = sorted((cw, ch))
    if long > max_dim and short * (max_dim / long) < min_dim:
        return None
    return image.crop((x1, y1, x2, y2))


class LayoutDetector:
    DEFAULT_MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self._model = None
        self._processor = None
        self._id2label: dict[int, str] | None = None
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForObjectDetection

        logger.info("Loading layout model: %s", self.model_id)

        try:
            from transformers import PPDocLayoutV3ImageProcessorFast

            self._processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
                self.model_id
            )
        except ImportError:
            from transformers import AutoImageProcessor

            self._processor = AutoImageProcessor.from_pretrained(self.model_id)

        self._model = AutoModelForObjectDetection.from_pretrained(
            self.model_id, torch_dtype=torch.float16
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = self._model.to(device).eval()
        self._id2label = self._model.config.id2label
        self._device = device
        logger.info("Layout model loaded on %s", device)

    def unload(self):
        if self._model is None:
            return
        import torch

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("Layout model unloaded")

    def detect(
        self,
        images: list[Image.Image],
        threshold: float = 0.3,
        batch_size: int = 4,
        containment_threshold: float = 0.8,
    ) -> list[list[dict]]:
        """Run layout detection on a list of PIL images."""
        import torch

        self.load()
        assert self._model is not None
        assert self._processor is not None
        assert self._id2label is not None

        all_results: list[list[dict]] = []

        for batch_start in range(0, len(images), batch_size):
            batch_imgs = images[batch_start : batch_start + batch_size]
            target_sizes = torch.tensor([img.size[::-1] for img in batch_imgs])

            inputs = self._processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=self._device, dtype=torch.float16
            )

            with torch.inference_mode():
                outputs = self._model(pixel_values=pixel_values)

            batch_dets = self._post_process(outputs, target_sizes, threshold)

            for dets in batch_dets:
                dets = filter_nested_detections(dets, containment_threshold)
                all_results.append(dets)

            del pixel_values, outputs

        return all_results

    def _post_process(
        self,
        outputs,
        target_sizes,
        threshold: float,
    ) -> list[list[dict]]:
        import torch

        logits = outputs.logits
        boxes = outputs.pred_boxes

        box_centers, box_dims = boxes.split(2, dim=-1)
        boxes_xyxy = torch.cat(
            [box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims],
            dim=-1,
        )

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
            device=boxes_xyxy.device,
            dtype=boxes_xyxy.dtype,
        )
        boxes_xyxy = boxes_xyxy * scale[:, None, :]

        num_queries = logits.shape[1]
        num_classes = logits.shape[2]
        scores = logits.sigmoid()
        scores_flat, index = scores.flatten(1).topk(num_queries, dim=-1)
        labels = index % num_classes
        box_indices = index // num_classes
        boxes_xyxy = boxes_xyxy.gather(
            dim=1,
            index=box_indices.unsqueeze(-1).expand(-1, -1, 4),
        )

        order_logits = getattr(outputs, "order_logits", None)
        has_order = order_logits is not None
        order_seqs = None
        if has_order:
            get_order_fn = getattr(self._processor, "_get_order_seqs", None)
            if get_order_fn is not None:
                order_seqs = get_order_fn(order_logits)
            else:
                order_seqs = order_logits.argmax(dim=-1)
            order_seqs = order_seqs.gather(dim=1, index=box_indices)

        batch_results: list[list[dict]] = []
        for batch_idx in range(scores_flat.shape[0]):
            s = scores_flat[batch_idx]
            l = labels[batch_idx]
            b = boxes_xyxy[batch_idx]
            mask = s >= threshold

            if has_order and order_seqs is not None:
                o_valid = order_seqs[batch_idx][mask]
                _, indices_sorted = o_valid.sort()
            else:
                indices_sorted = torch.arange(mask.sum().item())

            detections = []
            for si, li, bi in zip(
                s[mask][indices_sorted],
                l[mask][indices_sorted],
                b[mask][indices_sorted],
            ):
                detections.append(
                    {
                        "category": self._id2label[li.item()],
                        "bbox": [round(x, 2) for x in bi.tolist()],
                        "score": round(si.item(), 4),
                    }
                )
            batch_results.append(detections)

        return batch_results
