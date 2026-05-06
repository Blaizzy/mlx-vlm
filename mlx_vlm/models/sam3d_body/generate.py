"""SAM 3D Body predictor for mlx-vlm.

Usage:
    from mlx_vlm.models.sam3d_body.generate import SAM3DPredictor

    predictor = SAM3DPredictor.from_pretrained("path/to/weights")
    result = predictor.predict(image_rgb, bbox=[x1, y1, x2, y2])
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import SAM3DConfig
from .model import SAM3DBody


class SAM3DPredictor:
    """Single-image 3D body mesh prediction."""

    def __init__(self, model: SAM3DBody, config: SAM3DConfig):
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, weights_dir: str) -> "SAM3DPredictor":
        weights_dir = Path(weights_dir)
        config = SAM3DConfig.load(weights_dir / "config.json")
        model = SAM3DBody(config)
        model.load_all_weights(str(weights_dir))
        return cls(model, config)

    def predict(
        self,
        image: np.ndarray,
        bbox: list = None,
        cam_int: np.ndarray = None,
    ) -> dict:
        """Run 3D body estimation on a single image.

        Args:
            image: (H, W, 3) RGB uint8 numpy array
            bbox: [x1, y1, x2, y2] person bounding box. If None, uses full image.
            cam_int: (3, 3) camera intrinsic matrix. If None, estimated from image.

        Returns:
            dict with pred_vertices (V, 3), pred_keypoints_3d (70, 3),
            pred_camera (3,), bbox used.
        """
        from .batch_prep import get_cliff_condition, prepare_image

        h, w = image.shape[:2]
        if bbox is None:
            bbox = [0, 0, w, h]

        if cam_int is None:
            focal = math.sqrt(h**2 + w**2)  # image diagonal (PyTorch default)
            cam_int = np.array(
                [
                    [focal, 0, w / 2],
                    [0, focal, h / 2],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

        processed = prepare_image(
            image,
            bbox,
            image_size=self.config.image_size,
            mean=self.config.image_mean,
            std=self.config.image_std,
        )
        image_mx = mx.array(processed)

        focal_length = float(cam_int[0, 0])
        cliff = get_cliff_condition(bbox, (h, w), focal_length=focal_length)
        cliff_mx = mx.array(cliff[None])

        cam_int_mx = mx.array(cam_int)

        body_output, pred_cam = self.model(
            image_mx,
            cliff_condition=cliff_mx,
            bbox=bbox,
            img_size=(h, w),
            cam_int=cam_int_mx,
        )

        mx.eval(
            body_output["pred_vertices"],
            body_output["pred_keypoints_3d"],
            body_output["pred_joint_coords"],
            pred_cam,
        )

        return {
            "pred_vertices": np.array(body_output["pred_vertices"][0]),
            "pred_keypoints_3d": np.array(body_output["pred_keypoints_3d"][0]),
            "pred_joint_coords": np.array(body_output["pred_joint_coords"][0]),
            "pred_camera": np.array(pred_cam[0]),
            "bbox": bbox,
        }


def main():
    """CLI entry point for single-image prediction."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="SAM 3D Body MLX predictor")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--weights", required=True, help="Weights directory")
    parser.add_argument(
        "--bbox", type=str, default=None, help="Bounding box as 'x1,y1,x2,y2'"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    import cv2

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbox = None
    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]

    predictor = SAM3DPredictor.from_pretrained(args.weights)
    result = predictor.predict(image, bbox=bbox)

    v = result["pred_vertices"]
    span = np.max(v, axis=0) - np.min(v, axis=0)
    print(f"Vertices: {v.shape}, height: {span[1]:.3f}m")
    print(f"Camera: {result['pred_camera']}")

    if args.output:
        out = {
            "vertices_shape": list(v.shape),
            "mesh_span": span.tolist(),
            "camera": result["pred_camera"].tolist(),
            "bbox": result["bbox"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
