"""OWLv2 image and text processing."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ..base import install_auto_processor_patch


class OWLv2Tokenizer:
    """Simple CLIP-style tokenizer for OWLv2 text queries."""

    def __init__(self, max_length: int = 16):
        self.max_length = max_length
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer

            self._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch16"
            )
        return self._tokenizer

    def tokenize(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        tokenizer = self._get_tokenizer()
        encoded = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
        )
        return encoded["input_ids"], encoded["attention_mask"]


class OWLv2Processor:
    def __init__(
        self,
        image_size: int = 960,
        image_mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        image_std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
        max_length: int = 16,
    ):
        self.image_size = image_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.tokenizer = OWLv2Tokenizer(max_length=max_length)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        model_path = Path(model_path)

        image_size = 960
        image_mean = (0.48145466, 0.4578275, 0.40821073)
        image_std = (0.26862954, 0.26130258, 0.27577711)
        max_length = 16

        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            vision_cfg = config.get("vision_config", {})
            image_size = vision_cfg.get("image_size", image_size)

        preproc_path = model_path / "preprocessor_config.json"
        if preproc_path.exists():
            with open(preproc_path) as f:
                preproc = json.load(f)
            image_mean = tuple(preproc.get("image_mean", image_mean))
            image_std = tuple(preproc.get("image_std", image_std))
            image_size = preproc.get("size", {}).get("height", image_size)

        text_cfg = {}
        if config_path.exists():
            text_cfg = config.get("text_config", {})
        max_length = text_cfg.get("max_position_embeddings", max_length)

        return cls(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
            max_length=max_length,
        )

    def preprocess_image(self, image: Image.Image) -> Dict[str, np.ndarray]:
        original_size = (image.height, image.width)
        image = image.convert("RGB")

        # Match HF pipeline: rescale -> pad to square -> resize -> normalize
        arr = np.array(image, dtype=np.float32) / 255.0  # (H, W, 3)

        # Pad to square (bottom/right with zeros)
        h, w = arr.shape[:2]
        size = max(h, w)
        if h != size or w != size:
            padded = np.zeros((size, size, 3), dtype=np.float32)
            padded[:h, :w, :] = arr
            arr = padded

        # Resize to target using scipy (matches HF's anti-aliased resize)
        if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
            try:
                from scipy.ndimage import gaussian_filter, zoom

                # HF resizes in channels-last format (H, W, 3)
                target = self.image_size
                factors = np.array([arr.shape[0] / target, arr.shape[1] / target, 1.0])
                sigma = np.maximum(0, (factors - 1) / 2)
                arr = gaussian_filter(arr, sigma=sigma, mode="mirror")
                zoom_factors = [1.0 / f for f in factors]
                arr = zoom(arr, zoom_factors, order=1, mode="mirror", grid_mode=True)
                arr = np.clip(arr, 0.0, 1.0)
            except ImportError:
                # Fallback to PIL resize if scipy not available
                arr_pil = Image.fromarray((arr * 255).astype(np.uint8))
                arr_pil = arr_pil.resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
                arr = np.array(arr_pil, dtype=np.float32) / 255.0

        arr = (arr - self.image_mean) / self.image_std
        return {
            "pixel_values": arr[np.newaxis].astype(np.float32),  # (1, H, W, 3)
            "original_size": original_size,
        }


install_auto_processor_patch(["owlv2"], OWLv2Processor)
