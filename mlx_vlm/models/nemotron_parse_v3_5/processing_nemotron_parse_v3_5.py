# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, BatchFeature

from ..base import install_auto_processor_patch, load_chat_template
from ..nemotron_parse.processing_nemotron_parse import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    _resize_with_aspect_ratio,
)


class ImageProcessor:
    def __init__(
        self,
        final_size=(2048, 1664),
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
        **kwargs,
    ):
        self.final_size = final_size
        self.image_mean = np.array(image_mean, dtype=np.float32)[:, None, None]
        self.image_std = np.array(image_std, dtype=np.float32)[:, None, None]

    def __call__(self, images, **kwargs):
        images = images if isinstance(images, list) else [images]
        if len(images) != 1:
            raise ValueError("Nemotron 3.5 MLX inference supports one page at a time")
        array = _resize_with_aspect_ratio(
            np.asarray(images[0].convert("RGB")), self.final_size
        )
        h, w = array.shape[:2]
        ph, pw = ((h + 15) // 16) * 16, ((w + 63) // 64) * 64
        array = np.pad(array, ((0, ph - h), (0, pw - w), (0, 0)), constant_values=255)
        array = array.transpose(2, 0, 1).astype(np.float32) / 255
        array = (array - self.image_mean) / self.image_std
        patches = (
            array.reshape(3, ph // 16, 16, pw // 16, 16)
            .transpose(1, 3, 0, 2, 4)
            .reshape(-1, 768)
        )
        return {
            "pixel_values": patches,
            "image_sizes": np.array([[ph, pw]], dtype=np.int32),
            "content_sizes": np.array([[h, w]], dtype=np.int32),
        }


class Processor:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        data = self.image_processor(images) if images is not None else {}
        if text is not None:
            # The control-token prompt already includes BOS/EOS.
            data.update(
                self.tokenizer(text, add_special_tokens=False, return_tensors="np")
            )
        return BatchFeature(data=data)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config_path = Path(path) / "preprocessor_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        tokenizer = load_chat_template(
            AutoTokenizer.from_pretrained(path, **kwargs), path
        )
        return cls(ImageProcessor(**config), tokenizer)


install_auto_processor_patch("nemotron_parse_v3_5", Processor)
