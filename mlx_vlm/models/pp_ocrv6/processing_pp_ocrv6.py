from __future__ import annotations

import json
from pathlib import Path

from ..base import install_auto_processor_patch
from .pp_ocrv6 import ImageProcessor


class PPOCRV6Processor(ImageProcessor):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.image_processor = self
        self.tokenizer = self
        self.eos_token_id = 0
        self.eos_token_ids = [self.eos_token_id]

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        path = Path(path)
        data = {}

        config_path = path / "config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                data["config"] = json.load(f)

        preprocessor_path = path / "preprocessor_config.json"
        if preprocessor_path.is_file():
            with open(preprocessor_path, encoding="utf-8") as f:
                data.update(json.load(f))

        data.update(kwargs)
        return cls(**data)

    def __call__(
        self,
        images=None,
        text=None,
        padding=True,
        return_tensors="mlx",
        **kwargs,
    ):
        del text, padding, kwargs
        if images is None:
            raise ValueError("PP-OCRv6 processor requires images.")
        return self.preprocess(images, return_tensors=return_tensors)

    process = __call__

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        del kwargs
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        text = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.eos_token_ids:
                continue
            if 0 <= token_id < len(self.character_list):
                token = self.character_list[token_id]
                if skip_special_tokens and token == "blank":
                    continue
                text.append(token)
        return "".join(text)

    def encode(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        char_to_id = {char: i for i, char in enumerate(self.character_list)}
        return [char_to_id.get(char, self.eos_token_id) for char in text]


install_auto_processor_patch(
    [
        "pp_ocrv6",
        "pp_ocrv6_small_det",
        "pp_ocrv6_medium_det",
        "pp_ocrv6_tiny_rec",
        "pp_ocrv6_small_rec",
    ],
    PPOCRV6Processor,
)
