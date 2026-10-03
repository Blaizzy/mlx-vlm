"""Single-image geometry prediction for MoGe-3.

Usage:
    from mlx_vlm import load
    from mlx_vlm.models.moge3.generate import MoGe3Predictor, read_image

    model, processor = load("mlx-community/moge-3-vitl-mlx-fp32")
    predictor = MoGe3Predictor(model, processor)
    output = predictor.infer(read_image("image.jpg"))
    points, depth, mask = output["points"], output["depth"], output["mask"]
"""

from typing import Dict

import mlx.core as mx
import numpy as np


def read_image(image_source) -> np.ndarray:
    """Read an image (path, URL, data URI, BytesIO or PIL) as RGB uint8 (H, W, 3)."""
    from ...utils import load_image

    return np.asarray(load_image(image_source))


class MoGe3Predictor:
    def __init__(self, model, processor=None):
        self.model = model
        if processor is None:
            from .processing_moge3 import MogeProcessor

            processor = MogeProcessor()
        self.processor = processor

    def infer(self, image: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Predict geometry for an (H, W, 3) uint8/float RGB image or (B, H, W, 3) batch.

        Keyword arguments (``num_tokens``, ``resolution_level``,
        ``force_projection``, ``apply_mask``, ``fov_x``, ``refine_steps``,
        ``return_per_step``) are forwarded to :meth:`Model.infer`.

        Returns:
            Dict of numpy arrays with keys ``points`` (H, W, 3),
            ``intrinsics`` (3, 3), ``depth`` (H, W), ``mask`` (H, W),
            ``normal`` (H, W, 3) (batched forms carry a leading B).
        """
        image = self.processor.preprocess_image(image)
        output = self.model.infer(mx.array(image), **kwargs)
        return {
            key: (
                [np.array(v) for v in value]
                if isinstance(value, list)
                else np.array(value)
            )
            for key, value in output.items()
        }
