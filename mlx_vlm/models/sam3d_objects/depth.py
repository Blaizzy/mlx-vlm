"""Point-map estimation for SAM 3D with the shared MoGe-3 model."""

import mlx.core as mx

from .processing import float_rgb

# MoGe predicts OpenCV camera coordinates (+X right, +Y down, +Z forward);
# SAM 3D was trained on the PyTorch3D convention (+X left, +Y up, +Z forward).
CAMERA_FLIP = (-1.0, -1.0, 1.0)


def estimate_pointmap(model, image, num_tokens=None):
    output = model.infer(
        float_rgb(image),
        num_tokens=num_tokens,
        force_projection=False,
        apply_mask=False,
    )
    points = output["points"] * mx.array(CAMERA_FLIP)
    return mx.where(output["mask"][..., None], points, float("nan"))
