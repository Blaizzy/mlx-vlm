import glob
import inspect
import json
import re
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import to_numpy_array

from ..base import BaseImageProcessor
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    auto_map: dict
    hidden_size: int
    mm_hidden_size: int
    mm_projector_type: str = "mlp2x_gelu"
    ignore_index: int = -100
    image_token_index: int = -200
    vocab_size: int = 151936
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        if not params.get("text_config", {}):
            # Copy text config parameters from root level
            excluded_keys = {"vision_config"}
            params["text_config"] = dict(
                filter(lambda x: x[0] not in excluded_keys, params.items())
            )
        if not params.get("vision_config", {}).get("model_type", {}):
            # Set default model type
            params["vision_config"]["model_type"] = "siglip_vision_model"

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class ImageProcessor(BaseImageProcessor):
    def preprocess(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)

        return images


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class SigLipVisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(
        self, x: mx.array, output_hidden_states: Optional[bool] = None
    ) -> mx.array:
        return self.vision_tower(x, output_hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = SigLipVisionTower(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = LlavaMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        *_, hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = hidden_state[-1].astype(pixel_values.dtype)
        assert image_features.shape[-2] == 729

        image_features = self.mm_projector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        batch_size, seq_length, embed_dim = inputs_embeds.shape
        num_images, num_image_patches, _ = image_features.shape

        # Positions of <image> tokens in input_ids for each batch
        image_positions = mx.argmax(input_ids == image_token_index, axis=1)

        final_embeddings = []
        for b in range(batch_size):
            text_segments = []
            start_idx = 0
            position = int(image_positions[b].item())

            text_segments.append(inputs_embeds[b : b + 1, start_idx:position])
            text_segments.append(image_features[b : b + 1])
            text_segments.append(inputs_embeds[b : b + 1, position + 1 :])

            batch_embeddings = mx.concatenate(text_segments, axis=1)
            final_embeddings.append(batch_embeddings)

        # Create a final embedding of shape
        # (batch_size, num_image_patches + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=0)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        **kwargs,
    ):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
            mask=None,  # TODO: add mask
        )
        return logits

    def sanitize(self, weights):
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.vision_tower", k)
                else (
                    f"mm_projector.linear_1.{k.split('.')[-1]}"
                    if re.match(r"^model\.mm_projector\.0", k)
                    else (
                        f"mm_projector.linear_2.{k.split('.')[-1]}"
                        if re.match(r"^model\.mm_projector\.2", k)
                        else (
                            f"language_model.model.{k}"
                            if re.match(r"^lm_head", k)
                            else (
                                f"language_model.{k}"
                                if re.match(r"^model\.(embed_tokens|norm|layers)", k)
                                else k
                            )
                        )
                    )
                )
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"vision_tower.vision_tower.vision_model.head.attention.in_proj.bias"
                if re.match(
                    r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_bias",
                    k,
                )
                else (
                    f"vision_tower.vision_tower.vision_model.head.attention.in_proj.weight"
                    if re.match(
                        r"^vision_tower\.vision_tower\.vision_model\.head\.attention\.in_proj_weight",
                        k,
                    )
                    else k
                )
            ): v
            for k, v in weights.items()
        }

        return weights
