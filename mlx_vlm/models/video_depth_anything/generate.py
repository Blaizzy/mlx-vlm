"""Sliding-window video depth prediction for Video Depth Anything.

Ports the reference ``infer_video_depth`` loop: overlapping 32-frame windows,
keyframe conditioning, and scale/shift alignment between windows.

Usage:
    from mlx_vlm import load
    from mlx_vlm.models.video_depth_anything.generate import VideoDepthPredictor

    model, processor = load("mlx-community/Video-Depth-Anything-Small-MLX")
    predictor = VideoDepthPredictor(model, processor)
    depths = predictor.infer(frames)  # frames: (T, H, W, 3) uint8 RGB
"""

from typing import List, Optional

import mlx.core as mx
import numpy as np

# Inference settings from the reference implementation, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8


def compute_scale_and_shift(prediction, target, mask):
    """Least-squares scale/shift aligning prediction onto target."""
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)
    mask = mask.astype(np.float32)

    a_00 = np.sum(mask * prediction * prediction)
    a_01 = np.sum(mask * prediction)
    a_11 = np.sum(mask)

    b_0 = np.sum(mask * prediction * target)
    b_1 = np.sum(mask * target)

    x_0, x_1 = 1.0, 0.0
    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    return x_0, x_1


def get_interpolate_frames(frame_list_pre, frame_list_post):
    """Linear blend between the pre and post overlap frame lists."""
    assert len(frame_list_pre) == len(frame_list_post)
    n = len(frame_list_pre)
    step = 1.0 / (n - 1)
    post_w_list = [0.0] + [i * step for i in range(1, n - 1)] + [1.0]
    return [
        frame_list_pre[i] * (1 - w) + frame_list_post[i] * w
        for i, w in enumerate(post_w_list)
    ]


class VideoDepthPredictor:
    def __init__(self, model, processor=None, input_size: int = 518):
        self.model = model
        if processor is None:
            from .processing_video_depth_anything import VideoDepthProcessor

            processor = VideoDepthProcessor(input_size=input_size)
        self.processor = processor

    def infer(self, frames: np.ndarray, progress: bool = True) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, 3) uint8 RGB video frames.
        Returns:
            depths: (T, H, W) float32 depth maps at the input resolution.
        """
        frame_height, frame_width = frames.shape[1:3]

        frame_step = INFER_LEN - OVERLAP
        org_video_len = frames.shape[0]
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (
            INFER_LEN - frame_step
        )
        frame_list = [frames[i] for i in range(org_video_len)]
        frame_list += [frame_list[-1].copy()] * append_frame_len

        depth_list: List[np.ndarray] = []
        pre_input: Optional[mx.array] = None
        iterator = range(0, org_video_len, frame_step)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="inferring depth")
        for frame_id in iterator:
            cur = np.stack(
                [
                    self.processor.preprocess_frame(frame_list[frame_id + i])
                    for i in range(INFER_LEN)
                ]
            )
            cur_input = mx.array(cur)[None]  # (1, INFER_LEN, H, W, 3)
            if pre_input is not None:
                # Condition on the previous window's keyframes
                cur_input = mx.concatenate(
                    [pre_input[:, KEYFRAMES], cur_input[:, OVERLAP:]], axis=1
                )

            depth = self.model(cur_input)  # (1, INFER_LEN, H, W)

            # Resize each window back to the original frame resolution
            from .dpt import upsample_bilinear

            depth = upsample_bilinear(
                depth[0, :, :, :, None], size=(frame_height, frame_width)
            )[..., 0]
            depth_list += [np.array(depth[i]) for i in range(depth.shape[0])]

            pre_input = cur_input

        # Align consecutive windows on their shared keyframes
        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]
        metric = getattr(self.model.config, "metric", False)

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id + kf_id])
            else:
                curr_align = [
                    depth_list[frame_id + i] for i in range(len(kf_align_list))
                ]

                if metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(
                        np.concatenate(curr_align),
                        np.concatenate(ref_align),
                        np.concatenate([np.ones_like(r) for r in ref_align]),
                    )

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id + align_len : frame_id + OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = np.clip(
                        post_depth_list[i] * scale + shift, 0, None
                    )
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(
                    pre_depth_list, post_depth_list
                )

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id + i] * scale + shift
                    depth_list_aligned.append(np.clip(new_depth, 0, None))

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id + kf_id] * scale + shift
                    ref_align.append(np.clip(new_depth, 0, None))

        return np.stack(depth_list_aligned[:org_video_len], axis=0)


def read_video_frames(video_path: str, max_len: int = -1, target_fps: int = -1):
    """Read an RGB uint8 frame array from a video file (requires cv2)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if target_fps > 0 and original_fps > target_fps:
        stride = round(original_fps / target_fps)
    else:
        stride = 1
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (0 < max_len <= len(frames)):
            break
        if idx % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    fps = original_fps / stride
    return np.stack(frames), fps
