from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from ..pixtral.language import LanguageModel
from .config import ModelConfig


class VisionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def sanitize(self, weights: Dict[str, mx.array]):
        return weights


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights: Dict[str, mx.array]):
        def transform_key(key: str) -> str:
            if key.startswith("model.language_model.lm_head."):
                return key.replace(
                    "model.language_model.lm_head.", "language_model.lm_head."
                )
            if key.startswith("model.lm_head."):
                return key.replace("model.lm_head.", "language_model.lm_head.")
            if key.startswith("lm_head."):
                return key.replace("lm_head.", "language_model.lm_head.")
            if key.startswith("model.language_model.model."):
                return key.replace(
                    "model.language_model.model.", "language_model.model."
                )
            if key.startswith("model.language_model."):
                return key.replace("model.language_model.", "language_model.model.")
            if key.startswith("model."):
                suffix = key[len("model.") :]
                if suffix.startswith(("embed_tokens", "layers", "norm")):
                    return "language_model.model." + suffix
            return key

        remapped = {transform_key(k): v for k, v in weights.items()}
        return {k: v for k, v in remapped.items() if k.startswith("language_model.")}
