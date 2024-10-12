import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from ..base import KVCache
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 128256
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

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
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = input_ids == image_token_index
        inputs_embeds = np.array(inputs_embeds.astype(mx.float32))
        print(inputs_embeds.shape, image_positions.shape, image_features.shape)
        inputs_embeds[image_positions] = image_features

        # TODO: Add video features

        return mx.array(inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[mx.array, Optional[mx.array]]:

        aspect_ratio_ids = kwargs.pop("aspect_ratio_ids", None)
        aspect_ratio_mask = kwargs.pop("aspect_ratio_mask", None)
        cross_attention_mask = kwargs.pop("cross_attention_mask", None)

        # inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        inputs_embeds = None
        # Process vision input if provided
        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError(
                    "`aspect_ratio_ids` must be provided if `pixel_values` is provided"
                )

            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs[0]

            cross_attention_states = self.multi_modal_projector(
                cross_attention_states
            ).reshape(
                -1,
                cross_attention_states.shape[-2],
                self.config.text_config.hidden_size,
            )
            # inputs_embeds = self._merge_input_ids_with_image_features(
            #     cross_attention_states, inputs_embeds, input_ids
            # )
        else:
            cross_attention_states = None

        # Prepare cross attention mask
        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = (
                self._prepare_cross_attention_mask(
                    cross_attention_mask,
                    num_vision_tokens=(
                        self.config.vision_config.image_size
                        // self.config.vision_config.patch_size
                    )
                    ** 2
                    + 1,
                )
            )
        else:
            full_text_row_masked_out_mask = None

        # cache = None
        if cross_attention_mask is not None:
            cache_position = mx.arange(input_ids.shape[1], dtype=mx.int32)
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[
                :, :, cache_position
            ]

        # Process language input
        outputs = self.language_model(
            input_ids=input_ids,
            mask=mask,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            inputs_embeds=inputs_embeds,
            cache=cache,
        )

        return outputs

    def _prepare_cross_attention_mask(
        self,
        cross_attention_mask: mx.array,
        num_vision_tokens: int,
    ) -> Tuple[mx.array, mx.array]:
        batch_size, text_total_length, *_ = cross_attention_mask.shape
        cross_attention_mask = np.repeat(
            cross_attention_mask, num_vision_tokens, axis=3
        )
        cross_attention_mask = cross_attention_mask.reshape(
            batch_size, text_total_length, -1
        )
        cross_attention_mask = np.expand_dims(cross_attention_mask, 1)

        # Invert the mask
        inverted_cross_attn_mask = 1.0 - cross_attention_mask
        cross_attention_mask = np.where(
            inverted_cross_attn_mask,
            np.full_like(
                inverted_cross_attn_mask, np.finfo(inverted_cross_attn_mask.dtype).min
            ),
            cross_attention_mask,
        )

        # Apply full-row bias
        full_text_row_masked_out_mask = np.any(
            cross_attention_mask != np.finfo(cross_attention_mask.dtype).min,
            axis=-1,
            keepdims=True,
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        return mx.array(cross_attention_mask), mx.array(full_text_row_masked_out_mask)

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" not in key:
                key = key.replace("vision_model", "vision_tower")
            return key

        return {transform_key(k): v for k, v in weights.items()}
