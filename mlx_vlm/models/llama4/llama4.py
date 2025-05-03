import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .language import LanguageModel, TextConfig
from .vision import Llama4MultiModalProjector, VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 200092
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(
        self,
        pixel_values: mx.array,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
        **kwargs,
    ):
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                f"Unexpected select feature strategy: {self.vision_feature_select_strategy}"
            )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        hidden_state = self.vision_model(
            pixel_values, output_hidden_states=False, **kwargs
        )
        return hidden_state

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=kwargs.get("vision_feature_layer", -1),
            vision_feature_select_strategy=kwargs.get(
                "vision_feature_select_strategy", "default"
            ),
        )

        vision_flat = image_features.reshape(-1, image_features.shape[-1])
        projected_vision_flat = self.multi_modal_projector(vision_flat)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            projected_vision_flat, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        inputs_embeds[:, image_positions, :] = image_features

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        cache=None,
        **kwargs,
    ):

        input_embeddings = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        logits = self.language_model(
            input_ids=input_ids, cache=cache, input_embeds=input_embeddings
        )
        return logits
