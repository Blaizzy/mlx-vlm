"""Sapiens2 inference helper.

Unlike generative VLMs, Sapiens2 produces dense per-pixel or per-keypoint
predictions.  We expose a single `Sapiens2Predictor` that dispatches to
task-specific postprocessors and returns a typed `Sapiens2Result`.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from .processing_sapiens2 import Sapiens2Processor


@dataclass
class Sapiens2Result:
    task: str  # "pose" | "seg" | "normal" | "pointmap"
    original_size: Tuple[int, int]  # (orig_h, orig_w)

    # pose
    keypoints: Optional[np.ndarray] = None  # (N_kpt, 2) in (x, y) original pixel coords
    keypoint_scores: Optional[np.ndarray] = None  # (N_kpt,) heatmap peak values

    # seg
    mask: Optional[np.ndarray] = None  # (orig_h, orig_w) int class indices
    seg_logits: Optional[np.ndarray] = None  # (C, H_out, W_out) raw logits

    # normal
    normal: Optional[np.ndarray] = None  # (3, orig_h, orig_w) unit normals

    # pointmap
    pointmap: Optional[np.ndarray] = None  # (3, orig_h, orig_w) XYZ
    scale: Optional[float] = None  # focal-length ratio


def _bilinear_resize_np(x: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize (C, H, W) float array to (C, out_h, out_w) via PIL bilinear."""
    C, H, W = x.shape
    out_h, out_w = size
    out = np.empty((C, out_h, out_w), dtype=x.dtype)
    for c in range(C):
        im = Image.fromarray(x[c])
        im = im.resize((out_w, out_h), Image.BILINEAR)
        out[c] = np.asarray(im)
    return out


class Sapiens2Predictor:
    def __init__(self, model, processor: Sapiens2Processor):
        self.model = model
        self.processor = processor
        self.task = getattr(model.config, "task", "seg")

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Sapiens2Result:
        inputs = self.processor.preprocess_image(image)
        pixel_values = mx.array(inputs["pixel_values"])
        orig_h, orig_w = inputs["original_size"]

        outputs = self.model(pixel_values)
        mx.eval(outputs)

        if self.task == "pose":
            return self._postprocess_pose(outputs, (orig_h, orig_w))
        if self.task == "seg":
            return self._postprocess_seg(outputs, (orig_h, orig_w))
        if self.task == "normal":
            return self._postprocess_normal(outputs, (orig_h, orig_w))
        if self.task == "pointmap":
            return self._postprocess_pointmap(outputs, (orig_h, orig_w))
        raise ValueError(f"Unknown task: {self.task}")

    def _postprocess_pose(self, heatmaps: mx.array, orig) -> Sapiens2Result:
        # heatmaps: (1, H, W, K) (channels-last)
        hm = np.array(heatmaps)[0]  # (H, W, K)
        H, W, K = hm.shape
        flat = hm.reshape(-1, K)
        idx = np.argmax(flat, axis=0)  # (K,)
        ys = idx // W
        xs = idx % W
        scores = flat[idx, np.arange(K)]
        # Scale back to original image coordinates
        orig_h, orig_w = orig
        xs = xs.astype(np.float32) * (orig_w / W)
        ys = ys.astype(np.float32) * (orig_h / H)
        keypoints = np.stack([xs, ys], axis=-1)
        return Sapiens2Result(
            task="pose",
            original_size=orig,
            keypoints=keypoints,
            keypoint_scores=scores,
        )

    def _postprocess_seg(self, logits: mx.array, orig) -> Sapiens2Result:
        # logits: (1, H, W, C) channels-last
        arr = np.array(logits)[0]  # (H, W, C)
        arr = arr.transpose(2, 0, 1)  # (C, H, W)
        orig_h, orig_w = orig
        if arr.shape[1:] != orig:
            arr = _bilinear_resize_np(arr, (orig_h, orig_w))
        mask = np.argmax(arr, axis=0).astype(np.int32)  # (orig_h, orig_w)
        return Sapiens2Result(
            task="seg", original_size=orig, mask=mask, seg_logits=arr
        )

    def _postprocess_normal(self, normal: mx.array, orig) -> Sapiens2Result:
        arr = np.array(normal)[0].transpose(2, 0, 1)  # (3, H, W)
        orig_h, orig_w = orig
        if arr.shape[1:] != orig:
            arr = _bilinear_resize_np(arr, (orig_h, orig_w))
        norm = np.linalg.norm(arr, axis=0, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        arr = arr / norm
        return Sapiens2Result(task="normal", original_size=orig, normal=arr)

    def _postprocess_pointmap(self, outputs, orig) -> Sapiens2Result:
        pointmap, scale = outputs
        pm = np.array(pointmap)[0].transpose(2, 0, 1)  # (3, H, W)
        orig_h, orig_w = orig
        if pm.shape[1:] != orig:
            pm = _bilinear_resize_np(pm, (orig_h, orig_w))
        sc = float(np.array(scale).reshape(-1)[0])
        return Sapiens2Result(
            task="pointmap", original_size=orig, pointmap=pm, scale=sc
        )
