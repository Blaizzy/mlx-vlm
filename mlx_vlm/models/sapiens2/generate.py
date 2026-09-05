"""Single-image and streaming inference for Sapiens2 models (see README).

Everything from image ingest to the decoded outputs is one lazy MLX graph:
``infer`` returns unevaluated arrays in the original image resolution (dense
tasks) or in source-image pixel coordinates (pose), and ``stream`` pipelines
a sequence of frames with ``mx.async_eval``.
"""

import itertools
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, Optional, Tuple, Union

import mlx.core as mx

from ..interpolate import resize_bilinear_nhwc
from .image import ImageLike, normalize_image, preprocess, to_array, unpad_image
from .pose import (
    flip_indices_from_pairs,
    keypoints_to_image,
    udp_decode_batch,
    warp_to_input,
)


def read_image(image_source) -> mx.array:
    """Read an image (path, URL, data URI, BytesIO or PIL) as RGB uint8 (H, W, 3)."""
    from ...utils import load_image

    return to_array(load_image(image_source))


def read_video_frames(source, max_frames: Optional[int] = None) -> Iterator[mx.array]:
    """Yield the frames of a video file/URL or camera index as RGB uint8
    (H, W, 3) MLX arrays.

    Frames are decoded with OpenCV (imported on use, as elsewhere in
    mlx-vlm); the BGR -> RGB flip is a lazy device op.
    """
    import cv2

    is_camera = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_camera else str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source {source!r}")
    try:
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            yield to_array(frame)[..., ::-1]
            count += 1
    finally:
        cap.release()


def _resize_like(x: mx.array, size: Tuple[int, int]) -> mx.array:
    """Bilinear-resize a (H, W, C) map to ``size`` (h, w), matching the
    original vis tools."""
    return resize_bilinear_nhwc(x.astype(mx.float32)[None], size)[0]


# Flip-test average of two heatmap batches, in float32.
_average = mx.compile(lambda a, b: (a.astype(mx.float32) + b.astype(mx.float32)) * 0.5)


class Sapiens2Predictor:
    def __init__(self, model, processor=None):
        self.model = model
        self.processor = processor  # unused; preprocessing is task-specific
        self.config = model.config
        self.size = tuple(self.config.image_size)  # (H, W)
        self.flip_indices = None
        if self.config.task == "pose":
            self.flip_indices = flip_indices_from_pairs(
                self.config.num_labels, self.config.flip_pairs
            )

    def infer(
        self,
        image: ImageLike,
        boxes=None,
        flip_test: bool = True,
    ) -> Dict[str, mx.array]:
        """Build the inference graph for an (H, W, 3) RGB image (uint8 or float).

        Args:
            image: input image, as an MLX or numpy array or a PIL image.
            boxes: pose only - (N, 4) ``xyxy`` person boxes (array or nested
                list). Defaults to one box covering the full image.
            flip_test: pose only - also run the horizontally flipped crops
                and average the heatmaps. Doubles the model work; turn off
                for throughput.

        Returns a task-dependent dict of lazy MLX arrays, evaluated when
        read (``mx.eval``, ``np.array``, ``.tolist()``):
            backbone: ``last_hidden_state``, ``pooler_output`` (float32)
            seg:      ``segmentation`` (H, W) int32 class ids
            pose:     ``keypoints`` (N, K, 2), ``scores`` (N, K), ``boxes``
            normal:   ``normals`` (H, W, 3)
            pointmap: ``pointmaps`` (H, W, 3), ``scales`` (1,)
            matting:  ``alphas`` (H, W), ``foregrounds`` (H, W, 3)
        """
        image = to_array(image)
        task = self.config.task
        if task == "pose":
            return self._infer_pose(image, boxes, flip_test=flip_test)

        orig_size = image.shape[:2]  # (H, W)
        data = preprocess(image, task, self.size)
        output = self.model(data["pixel_values"])
        key = {
            "backbone": None,
            "seg": "logits",
            "normal": "normals",
            "pointmap": "pointmaps",
            "matting": None,
        }[task]

        if task == "backbone":
            return {k: v.astype(mx.float32) for k, v in output.items()}

        if task == "matting":
            rgba = mx.concatenate([output["foregrounds"][0], output["alphas"][0]], -1)
            # clip after resize, as bilinear interpolation can overshoot
            rgba = mx.clip(_resize_like(rgba, orig_size), 0, 1)
            return {"alphas": rgba[..., 3], "foregrounds": rgba[..., :3]}

        pred = output[key][0].astype(mx.float32)  # (h, w, C)
        if task == "normal":
            # unit-length normals, as in the original vis_normal
            norm = mx.linalg.norm(pred, axis=-1, keepdims=True)
            pred = pred / mx.maximum(norm, 1e-8)
        if data["padding"] is not None:
            pred = unpad_image(pred, data["padding"])
        pred = _resize_like(pred, orig_size)

        if task == "seg":
            return {"segmentation": pred.argmax(axis=-1).astype(mx.int32)}
        if task == "pointmap":
            return {
                "pointmaps": pred,
                "scales": output["scales"][0].astype(mx.float32),
            }
        return {"normals": pred}

    def stream(
        self,
        frames: Union[Iterable[ImageLike], str, Path, int],
        boxes: Optional[Iterable] = None,
        flip_test: bool = True,
        prefetch: int = 1,
    ) -> Iterator[Dict[str, mx.array]]:
        """Run a sequence of frames (e.g. a video) as a pipeline.

        Args:
            frames: any iterable of images ``infer`` accepts, or a video
                path / camera index (read with ``read_video_frames``).
            boxes: pose only - optional iterable of per-frame boxes, paired
                with the frames.
            flip_test: see ``infer``.
            prefetch: frames dispatched ahead of the one being yielded.

        Each frame's graph is dispatched with ``mx.async_eval``, and the
        previous frame's outputs are yielded while it runs: decoding,
        preprocessing and the caller's per-frame work overlap with the GPU.
        Reading a yielded output blocks only until that frame is done. Every
        yielded dict also carries ``frame``, the RGB uint8 input as a device
        array, for overlays.
        """
        if isinstance(frames, (str, Path, int)):
            frames = read_video_frames(frames)
        if boxes is None:
            pairs = zip(frames, itertools.repeat(None))
        else:
            pairs = zip(frames, boxes, strict=True)

        pending: Deque[Dict[str, mx.array]] = deque()
        for frame, frame_boxes in pairs:
            frame = to_array(frame)
            out = self.infer(frame, frame_boxes, flip_test=flip_test)
            out["frame"] = frame
            mx.async_eval(out)
            pending.append(out)
            if len(pending) > prefetch:
                yield pending.popleft()
        yield from pending

    def _infer_pose(
        self, image: mx.array, boxes, flip_test: bool = True
    ) -> Dict[str, mx.array]:
        h, w = image.shape[:2]
        if boxes is None or len(boxes) == 0:
            boxes = [[0, 0, w - 1, h - 1]]
        boxes = mx.array(boxes, dtype=mx.float32)

        input_wh = (self.size[1], self.size[0])  # codec uses (w, h)
        image = image.astype(mx.float32)
        crops, centers, scales = [], [], []
        for bbox in boxes:
            crop, center, scale = warp_to_input(image, bbox, input_wh)
            crops.append(normalize_image(crop))
            centers.append(center)
            scales.append(scale)

        batch = mx.stack(crops)
        if flip_test:
            # One forward for the crops and their mirror images.
            batch = mx.concatenate([batch, batch[:, :, ::-1, :]])
        heatmaps = self.model(batch)["heatmaps"]  # (.., h, w, K)
        if flip_test:
            heatmaps, flipped = mx.split(heatmaps, 2)
            heatmaps = _average(
                heatmaps, flipped[:, :, ::-1, :][..., self.flip_indices]
            )

        keypoints, scores = udp_decode_batch(heatmaps, input_wh)
        keypoints = keypoints_to_image(
            keypoints, input_wh, mx.stack(centers), mx.stack(scales)
        )
        return {"keypoints": keypoints, "scores": scores, "boxes": boxes}
