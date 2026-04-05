"""OWLv2 detection inference pipeline."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from ..rfdetr.generate import _nms_per_class, box_cxcywh_to_xyxy


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N, 4) xyxy in pixel coordinates
    scores: np.ndarray  # (N,) confidence 0-1
    labels: np.ndarray  # (N,) class indices
    class_names: List[str]  # (N,) human-readable names
    objectness: Optional[np.ndarray] = None  # (N,) objectness scores


def postprocess(
    outputs: dict,
    original_size: Tuple[int, int],
    query_labels: List[str],
    score_threshold: float = 0.1,
    nms_threshold: float = 0.3,
    objectness_threshold: float = 0.0,
) -> DetectionResult:
    pred_logits = np.array(outputs["pred_logits"][0])  # (N, num_queries)
    pred_boxes = np.array(outputs["pred_boxes"][0])  # (N, 4)
    objectness = np.array(outputs["objectness_logits"][0, :, 0])  # (N,)

    # Sigmoid for scores (matches HF: max logit per token -> sigmoid)
    scores_all = 1.0 / (1.0 + np.exp(-pred_logits))  # (N, num_queries)
    objectness_scores = 1.0 / (1.0 + np.exp(-objectness))  # (N,)

    # Get max class score and label per detection
    max_scores = scores_all.max(axis=-1)  # (N,)
    max_labels = scores_all.argmax(axis=-1)  # (N,)

    # Filter by score threshold (matching HF behavior: no objectness multiplication)
    keep = max_scores > score_threshold
    if objectness_threshold > 0.0:
        keep = keep & (objectness_scores > objectness_threshold)
    scores = max_scores[keep]
    labels = max_labels[keep]
    boxes = pred_boxes[keep]
    obj_scores = objectness_scores[keep]

    if len(scores) == 0:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            scores=np.zeros((0,)),
            labels=np.zeros((0,), dtype=np.int64),
            class_names=[],
            objectness=np.zeros((0,)),
        )

    # Convert boxes from cxcywh to xyxy
    boxes = box_cxcywh_to_xyxy(boxes)

    # Denormalize to pixel coordinates
    # Boxes are normalized to the padded square image, so denormalize
    # by the padded size (max(h, w)), then clip to the original image
    h, w = original_size
    padded_size = max(h, w)
    boxes[:, [0, 2]] *= padded_size
    boxes[:, [1, 3]] *= padded_size
    boxes = np.clip(boxes, 0, [w, h, w, h])

    # Per-class NMS
    keep_idx = _nms_per_class(boxes, scores, labels, iou_threshold=nms_threshold)

    if len(keep_idx) == 0:
        return DetectionResult(
            boxes=np.zeros((0, 4)),
            scores=np.zeros((0,)),
            labels=np.zeros((0,), dtype=np.int64),
            class_names=[],
            objectness=np.zeros((0,)),
        )

    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    obj_scores = obj_scores[keep_idx]

    class_names = [query_labels[int(l)] for l in labels]

    return DetectionResult(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=class_names,
        objectness=obj_scores,
    )


class OWLv2Predictor:
    def __init__(
        self,
        model,
        processor,
        tokenizer,
        query_labels: List[str],
        score_threshold: float = 0.1,
        nms_threshold: float = 0.3,
        objectness_threshold: float = 0.0,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.query_labels = query_labels
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.objectness_threshold = objectness_threshold

        # Pre-encode text queries
        self._encode_queries()

    def _encode_queries(self):
        input_ids, attention_mask = self.tokenizer.tokenize(self.query_labels)
        self._input_ids = mx.array(input_ids)
        self._attention_mask = mx.array(attention_mask)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> DetectionResult:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        original_size = (image.height, image.width)
        inputs = self.processor.preprocess_image(image)
        pixel_values = mx.array(inputs["pixel_values"])

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=self._input_ids,
            attention_mask=self._attention_mask,
        )
        mx.eval(
            outputs["pred_logits"], outputs["pred_boxes"], outputs["objectness_logits"]
        )

        return postprocess(
            outputs,
            original_size,
            self.query_labels,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            objectness_threshold=self.objectness_threshold,
        )

    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
    ):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.predict(rgb)
                annotated = self._draw_detections(frame, result)

                if writer:
                    writer.write(annotated)
                if display:
                    cv2.imshow("OWLv2 Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    @staticmethod
    def _draw_detections(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        import cv2

        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for i in range(len(result.scores)):
            x1, y1, x2, y2 = result.boxes[i].astype(int)
            color = colors[int(result.labels[i]) % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{result.class_names[i]}: {result.scores[i]:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return frame
