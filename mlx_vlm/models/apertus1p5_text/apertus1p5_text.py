"""Apertus 1.5 text backbone.

The language tower of ``swiss-ai/Apertus-v1.5-8B`` on its own, for text-only
conversions. The implementation lives in ``models/apertus1p5``; this package
just wraps it, the way ``gemma3_text`` wraps ``gemma3``.

``sanitize`` accepts the multimodal checkpoint layout directly: text weights
under ``model.language_model.*`` with a top-level ``lm_head.weight``, and the
image / audio code tokenizers dropped.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..apertus1p5.apertus1p5 import AUDIO_PREFIX, LANGUAGE_PREFIX, VISION_PREFIX
from ..apertus1p5.language import LanguageModel
from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig

CODE_TOKENIZER_PREFIXES = (VISION_PREFIX, AUDIO_PREFIX)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        if any(k.startswith("language_model.") for k in weights):
            return weights
        weights = {
            # Strip the multimodal wrapper so keys line up with the trunk.
            (
                "model." + k[len(LANGUAGE_PREFIX) :]
                if k.startswith(LANGUAGE_PREFIX)
                else k
            ): v
            for k, v in weights.items()
            if not k.startswith(CODE_TOKENIZER_PREFIXES)
        }
        weights = self.language_model.sanitize(weights)
        return {f"language_model.{k}": v for k, v in weights.items()}

    @property
    def layers(self):
        return self.language_model.layers
