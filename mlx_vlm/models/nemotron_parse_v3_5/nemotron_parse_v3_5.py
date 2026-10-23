# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..nemotron_parse.nemotron_parse import Model as EncoderDecoderModel
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(
            config.text_config, config.max_sequence_length
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if input_ids is None or pixel_values is None:
            raise ValueError("Nemotron 3.5 requires prompt input_ids and pixel_values")
        vision = self.vision_tower(
            pixel_values, kwargs.get("image_sizes"), kwargs.get("content_sizes")
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=mx.concatenate(
                [vision, self.language_model.embed(input_ids)], axis=1
            )
        )

    def __call__(self, input_ids=None, pixel_values=None, cache=None, **kwargs):
        if pixel_values is not None:
            embeddings = self.get_input_embeddings(
                input_ids, pixel_values, **kwargs
            ).inputs_embeds
            return self.language_model(
                inputs_embeds=embeddings, cache=cache, logits_to_keep=input_ids.shape[1]
            )
        return self.language_model(input_ids, cache=cache)

    @staticmethod
    def sanitize(weights):
        # Reuse only the checkpoint's vision name/layout mapping from the 2.0 port.
        vision = EncoderDecoderModel.sanitize(
            {k: v for k, v in weights.items() if k.startswith("encoder.")}
        )
        result = dict(vision)
        for name, value in weights.items():
            if name.startswith("encoder.") or name.startswith("lm_head."):
                continue
            if name.startswith("decoder.extra_heads.") or name.startswith(
                "decoder.extra_proj."
            ):
                # ponytail: ordinary autoregressive decoding; MTP acceleration can follow.
                continue
            result[
                (
                    name.replace("decoder.", "language_model.", 1)
                    if name.startswith("decoder.")
                    else name
                )
            ] = value
        return result
