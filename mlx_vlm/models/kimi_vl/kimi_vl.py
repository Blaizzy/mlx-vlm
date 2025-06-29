import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    vocab_size: int = 128259
    scale_factor: int = 2
    media_placeholder_token_id: int = 163606
    image_token_index: Optional[int] = None
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.media_placeholder_token_id

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class KimiVLMultiModalProjector(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = (
            config.vision_config.hidden_size
            * config.vision_config.merge_kernel_size[0]
            * config.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(
            self.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, image_features: list[mx.array]) -> mx.array:
        image_features = mx.concatenate(image_features, axis=0)
        h = self.pre_norm(image_features).reshape(-1, self.hidden_size)
        h = self.linear_1(h)
        h = self.act(h)
        h = self.linear_2(h)
        return h


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
            grid_thw=grid_thw,
        )

        image_features = self.multi_modal_projector(hidden_state)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
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
        image_grid_thw = kwargs.pop("image_grid_hws", None)
        video_grid_thw = kwargs.pop("video_grid_hws", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        input_embeddings = self.get_input_embeddings(
            input_ids, pixel_values, grid_thw=grid_thw
        )
        logits = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits

    def sanitize(self, weights):
        return {
            k.replace("encoder.", "") if "vision_tower" in k else k: v
            for k, v in weights.items()
        }
