"""Apertus 1.5 text backbone.

Apertus 1.5 (``swiss-ai/Apertus-v1.5-8B``) is an early-fusion, discrete-token
multimodal model: images and audio are encoded to code tokens that live in the
same vocabulary as text. Its language tower is a plain Apertus decoder (xIELU
MLP, qk-norm attention, llama3 RoPE), so the trunk is reused verbatim from
``..apertus``. Two things differ:

1. Split vocabulary. The input embedding spans the *extended* vocabulary
   (``vocab_size`` = 266752: text + visual + audio codes) while the LM head is
   physically pruned to the text-only prefix (``output_vocab_size`` = 131072).
   Code tokens are input-only and never generated, so the pruned head covers
   exactly the generatable range.

2. Checkpoint layout. Text weights live under ``model.language_model.*`` (the
   multimodal wrapper's submodule), the head is a top-level ``lm_head.weight``,
   and rope config is a single ``rope_parameters`` dict rather than the legacy
   ``rope_theta`` + ``rope_scaling`` pair.

This module covers the text tower only; the vision/audio code tokenizers
(``model.vision_tokenizer.*``, ``model.audio_tokenizer.*``) are dropped when
loading and image/audio input is not supported yet.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel

# Submodules of the multimodal wrapper that this text-only model does not use.
CODE_TOKENIZER_PREFIXES = ("model.vision_tokenizer.", "model.audio_tokenizer.")

WRAPPER_PREFIX = "model.language_model."


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
                "model." + k[len(WRAPPER_PREFIX) :]
                if k.startswith(WRAPPER_PREFIX)
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
