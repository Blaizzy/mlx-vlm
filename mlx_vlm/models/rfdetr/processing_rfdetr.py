"""RF-DETR image processor."""

from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch

# COCO 91-class names (index 0 = N/A background, actual categories at specific indices)
COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class RFDETRProcessor:
    """Image processor for RF-DETR detection model."""

    def __init__(
        self,
        resolution: int = 560,
        image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        image_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        num_select: int = 300,
    ):
        self.resolution = resolution
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.num_select = num_select

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        """Load processor from model directory."""
        import json
        from pathlib import Path

        path = Path(path)

        # Always read resolution from model config
        resolution = 560
        model_config_path = path / "config.json"
        if model_config_path.exists():
            with open(model_config_path) as f:
                mconfig = json.load(f)
            resolution = mconfig.get("resolution", 560)

        # Read image normalization from preprocessor config
        config_path = path / "preprocessor_config.json"
        if config_path.exists():
            with open(config_path) as f:
                pconfig = json.load(f)
            img_config = pconfig.get("config", {})
            return cls(
                resolution=resolution,
                image_mean=tuple(img_config.get("image_mean", (0.485, 0.456, 0.406))),
                image_std=tuple(img_config.get("image_std", (0.229, 0.224, 0.225))),
                num_select=pconfig.get("post_process_config", {}).get(
                    "num_select", 300
                ),
            )

        return cls(resolution=resolution)

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Preprocess an image for RF-DETR.

        Args:
            image: PIL Image or numpy array (H, W, 3) in RGB
        Returns:
            dict with 'pixel_values': (1, H, W, 3) normalized array
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_w, orig_h = image.size
        image = image.convert("RGB")
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)

        pixel_values = np.array(image, dtype=np.float32) / 255.0
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        pixel_values = pixel_values[np.newaxis, ...]

        return {
            "pixel_values": pixel_values,
            "original_size": (orig_h, orig_w),
        }

    def preprocess_bgr(self, bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocessing from BGR numpy array (cv2 frame).

        Args:
            bgr: (H, W, 3) BGR uint8 array (from cv2)
        Returns:
            dict with 'pixel_values': (1, H, W, 3) normalized array
        """
        orig_h, orig_w = bgr.shape[:2]
        # Use PIL resize to match predict() path exactly
        rgb = Image.fromarray(bgr[..., ::-1])  # BGR->RGB via array flip + PIL
        rgb = rgb.resize((self.resolution, self.resolution), Image.BILINEAR)
        pixel_values = np.array(rgb, dtype=np.float32) / 255.0
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        pixel_values = pixel_values[np.newaxis, ...]

        return {
            "pixel_values": pixel_values,
            "original_size": (orig_h, orig_w),
        }


# Install auto processor patch for model loading
install_auto_processor_patch(["rf-detr"], RFDETRProcessor)
