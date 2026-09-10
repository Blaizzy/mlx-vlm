"""RTMDet preprocessor: letterbox resize to image_size, pad with 114.

The original mmdetection pipeline is:
  1. keep-ratio resize so that max(H, W) == image_size  (PIL bilinear)
  2. pad to (image_size, image_size) at bottom/right with grey 114

No mean/std normalization (RTMDet runs on [0, 255] inputs directly).
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch


class RTMDetProcessor:
    def __init__(self, image_size: Tuple[int, int] = (640, 640),
                 pad_value: Tuple[int, int, int] = (114, 114, 114),
                 # mmdet DetDataPreprocessor defaults for RTMDet / Sapiens pose-bbox-detector:
                 #   mean = [103.53, 116.28, 123.675] (BGR order; bgr_to_rgb=False)
                 #   std  = [ 57.375, 57.12,  58.395]
                 # Since we load images as RGB via PIL, the channel-order flip means the
                 # effective RGB mean/std is the reverse (ImageNet-standard on [0, 255]).
                 image_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
                 image_std: Tuple[float, float, float] = (58.395, 57.12, 57.375)):
        self.image_size = tuple(image_size)
        self.pad_value = np.array(pad_value, dtype=np.float32)
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)

    @classmethod
    def from_pretrained(cls, path, **_kwargs):
        import json
        path = Path(path)
        image_size = (640, 640)
        cfg = path / "config.json"
        if cfg.exists():
            obj = json.loads(cfg.read_text())
            if "image_size" in obj:
                image_size = tuple(obj["image_size"])
        return cls(image_size=image_size)

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Returns:
          pixel_values : (1, H, W, 3) float32 in [0, 255]
          original_size: (orig_h, orig_w)
          scale        : scalar — ratio used by the resize
          pad_top_left : (pad_h, pad_w) offsets applied during padding (here 0, 0)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        H, W = self.image_size
        scale = min(H / orig_h, W / orig_w)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32)  # (new_h, new_w, 3)

        # Pad bottom/right to (H, W) with 114 (same grey the detector was trained on)
        canvas = np.full((H, W, 3), self.pad_value, dtype=np.float32)
        canvas[:new_h, :new_w] = arr
        # Normalize (DetDataPreprocessor step): (x - mean) / std — runs on the
        # padded image so that pad pixels also get normalized (mmdet does the
        # same, the grey 114 maps to near-zero post-normalization).
        canvas = (canvas - self.image_mean) / self.image_std

        return {
            "pixel_values": canvas[np.newaxis, ...],
            "original_size": (orig_h, orig_w),
            "scale": scale,
            "pad_top_left": (0, 0),
        }

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            items = [self.preprocess_image(im) for im in images]
            return {
                "pixel_values": np.concatenate([it["pixel_values"] for it in items], axis=0),
                "original_size": [it["original_size"] for it in items],
                "scale": [it["scale"] for it in items],
                "pad_top_left": [it["pad_top_left"] for it in items],
            }
        return self.preprocess_image(images)

    def save_pretrained(self, save_directory, **_kwargs):
        import json
        d = Path(save_directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "preprocessor_config.json").write_text(json.dumps({
            "image_size": list(self.image_size),
            "pad_value": self.pad_value.tolist(),
        }, indent=2))


install_auto_processor_patch(["rtmdet"], RTMDetProcessor)
