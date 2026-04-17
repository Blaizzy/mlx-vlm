"""Custom processor for Moondream3.

Handles image cropping and text tokenization for the Moondream3 model.
"""

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..base import install_auto_processor_patch, load_chat_template
from .image_crops import create_crops

# Moondream3 uses a custom tokenizer from this repo
TOKENIZER_REPO = "moondream/starmie-v1"

# Number of vision tokens produced by the vision encoder (27x27 grid)
NUM_VISION_TOKENS = 729

# Special token IDs for moondream3
BOS_ID = 0
EOS_ID = 0
ANSWER_ID = 3
THINKING_ID = 4


class Moondream3Processor:
    """Processor for Moondream3 that handles image cropping and tokenization."""

    def __init__(self, tokenizer, crop_size=378, max_crops=12, overlap_margin=4):
        self.tokenizer = tokenizer
        self.crop_size = crop_size
        self.max_crops = max_crops
        self.overlap_margin = overlap_margin

        # Ensure special tokens are set (starmie-v1 tokenizer lacks these)
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        # Filter out kwargs that aren't relevant to tokenizer loading
        tokenizer_kwargs = {}
        for key in ("trust_remote_code", "revision", "token"):
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]
        tokenizer_kwargs.setdefault("trust_remote_code", True)

        # Try loading tokenizer from the model path first, fall back to starmie-v1
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                **tokenizer_kwargs,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_REPO,
                **tokenizer_kwargs,
            )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        # Read config for crop parameters
        config_path = Path(pretrained_model_name_or_path) / "config.json"
        crop_size = 378
        max_crops = 12
        overlap_margin = 4

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                vc = config.get("vision_config", {})
                crop_size = vc.get("crop_size", crop_size)
                max_crops = vc.get("max_crops", max_crops)
                overlap_margin = vc.get("overlap_margin", overlap_margin)

        # Read processor_config.json for correct init kwargs
        proc_cfg_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        proc_kwargs = {}
        if proc_cfg_path.exists():
            with open(proc_cfg_path) as f:
                proc_cfg = json.load(f)
            for k in ("crop_size", "max_crops", "overlap_margin"):
                if k in proc_cfg:
                    proc_kwargs[k] = proc_cfg[k]

        # proc_kwargs override config.json values if present
        crop_size = proc_kwargs.pop("crop_size", crop_size)
        max_crops = proc_kwargs.pop("max_crops", max_crops)
        overlap_margin = proc_kwargs.pop("overlap_margin", overlap_margin)

        return cls(
            tokenizer=tokenizer,
            crop_size=crop_size,
            max_crops=max_crops,
            overlap_margin=overlap_margin,
        )

    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        images=None,
        return_tensors: str = "np",
        padding: bool = True,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        result = {}

        # Process images
        has_images = images is not None and (
            not hasattr(images, "__len__") or len(images) > 0
        )
        if has_images:
            if not isinstance(images, list):
                images = [images]

            all_crops = []
            num_crops = []
            crop_layouts = []

            for image in images:
                crops, layout = create_crops(
                    image,
                    self.crop_size,
                    self.max_crops,
                    self.overlap_margin,
                )
                all_crops.extend(crops)
                num_crops.append(len(crops))
                crop_layouts.append(layout)

            # Stack all crops: (total_crops, H, W, C) - already in channel-last
            pixel_values = np.stack(all_crops, axis=0)
            result["pixel_values"] = pixel_values
            result["num_crops"] = num_crops
            result["crop_layouts"] = crop_layouts

        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]

            # Build input sequences: [BOS] [image_placeholders] [text_tokens]
            all_input_ids = []
            bos_id = self.tokenizer.bos_token_id or 0

            for t in text:
                if has_images:
                    # Format: BOS + 729 placeholder + "\n\nQuestion: {q}\n\nAnswer: " + ANSWER_ID
                    formatted = f"\n\nQuestion: {t}\n\nAnswer: "
                    text_tokens = self.tokenizer.encode(
                        formatted, add_special_tokens=False
                    )
                    input_ids = (
                        [bos_id] + [0] * NUM_VISION_TOKENS + text_tokens + [ANSWER_ID]
                    )
                else:
                    text_tokens = self.tokenizer.encode(t, add_special_tokens=False)
                    if add_special_tokens:
                        input_ids = [bos_id] + text_tokens
                    else:
                        input_ids = text_tokens

                all_input_ids.append(input_ids)

            # Pad if needed
            if padding and len(all_input_ids) > 1:
                max_len = max(len(ids) for ids in all_input_ids)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = []
                masks = []
                for ids in all_input_ids:
                    pad_len = max_len - len(ids)
                    padded.append([pad_id] * pad_len + ids)  # left padding
                    masks.append([0] * pad_len + [1] * len(ids))
                result["input_ids"] = np.array(padded, dtype=np.int32)
                result["attention_mask"] = np.array(masks, dtype=np.int32)
            else:
                result["input_ids"] = np.array(all_input_ids, dtype=np.int32)
                result["attention_mask"] = np.ones_like(
                    result["input_ids"], dtype=np.int32
                )

        return result


# Register the processor
install_auto_processor_patch(["moondream3"], Moondream3Processor)
