"""SAM3 Processor: Image and text preprocessing for SAM3 model.

Handles image resizing/normalization and CLIP tokenization.
"""

from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image


class Sam3Processor:
    """Processor for SAM3 model."""

    def __init__(
        self,
        image_size: int = 1008,
        image_mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, ...] = (0.5, 0.5, 0.5),
        max_text_length: int = 32,
    ):
        self.image_size = image_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_text_length = max_text_length
        self._tokenizer = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model."""
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download

        model_path = Path(pretrained_model_name_or_path)
        if not model_path.exists():
            try:
                proc_path = hf_hub_download(
                    pretrained_model_name_or_path, "processor_config.json"
                )
                with open(proc_path) as f:
                    proc_config = json.load(f)
            except Exception:
                proc_config = {}
        else:
            proc_file = model_path / "processor_config.json"
            if proc_file.exists():
                with open(proc_file) as f:
                    proc_config = json.load(f)
            else:
                proc_config = {}

        img_proc = proc_config.get("image_processor", {})
        size = img_proc.get("size", {})
        image_size = size.get("height", 1008)
        image_mean = tuple(img_proc.get("image_mean", [0.5, 0.5, 0.5]))
        image_std = tuple(img_proc.get("image_std", [0.5, 0.5, 0.5]))

        return cls(
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save processor config and tokenizer to directory."""
        import json
        from pathlib import Path

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save processor_config.json
        proc_config = {
            "processor_class": "Sam3Processor",
            "image_processor": {
                "image_processor_type": "Sam3ImageProcessor",
                "size": {"height": self.image_size, "width": self.image_size},
                "image_mean": [float(x) for x in self.image_mean],
                "image_std": [float(x) for x in self.image_std],
                "do_resize": True,
                "do_normalize": True,
                "do_rescale": True,
                "rescale_factor": 1 / 255.0,
            },
            "target_size": self.image_size,
        }
        with open(save_dir / "processor_config.json", "w") as f:
            json.dump(proc_config, f, indent=2)

        # Copy tokenizer files if available
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(save_dir))
        else:
            # Save tokenizer from the CLIP model
            try:
                tok = self.tokenizer  # triggers lazy load
                tok.save_pretrained(str(save_dir))
            except Exception:
                pass

    @property
    def tokenizer(self):
        """Lazy-load CLIP tokenizer."""
        if self._tokenizer is None:
            from transformers import CLIPTokenizer

            # SAM3 uses CLIP tokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        return self._tokenizer

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, List],
    ) -> Dict[str, np.ndarray]:
        """Preprocess image(s) for the model.

        Args:
            image: PIL Image, numpy array (H,W,3), or list of images
        Returns:
            dict with 'pixel_values': (B, H, W, 3) normalized array
        """
        if isinstance(image, list):
            images = [self._process_single_image(img) for img in image]
            pixel_values = np.stack(images)
        else:
            pixel_values = self._process_single_image(image)[None]

        return {"pixel_values": pixel_values}

    def _process_single_image(
        self, image: Union[Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Process a single image."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Resize to target size
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy and normalize
        pixel_values = np.array(image).astype(np.float32) / 255.0
        pixel_values = (pixel_values - self.image_mean) / self.image_std

        return pixel_values  # (H, W, 3) - MLX channel-last

    def preprocess_text(
        self,
        text: Union[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """Tokenize text prompt(s).

        Args:
            text: string or list of strings
        Returns:
            dict with 'input_ids' (B, seq_len) and 'attention_mask' (B, seq_len)
        """
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="np",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def preprocess_video(
        self,
        frames: List[Union[Image.Image, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Preprocess video frames.

        Args:
            frames: list of images
        Returns:
            dict with 'pixel_values': (T, H, W, 3)
        """
        processed = [self._process_single_image(f) for f in frames]
        return {"pixel_values": np.stack(processed)}


# Install auto processor patch
from ..base import install_auto_processor_patch

install_auto_processor_patch(["sam3_video", "sam3"], Sam3Processor)
