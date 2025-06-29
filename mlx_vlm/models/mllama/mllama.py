import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from ..cache import KVCache
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
    eos_token_id: Optional[List[int]] = None

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
        cross_attention_mask = mx.repeat(
            cross_attention_mask, num_vision_tokens, axis=3
        )
        cross_attention_mask = cross_attention_mask.reshape(
            batch_size, text_total_length, -1
        )
        cross_attention_mask = mx.expand_dims(cross_attention_mask, 1)

        # Invert the mask
        inverted_cross_attn_mask = 1.0 - cross_attention_mask
        fill_array = mx.array(-1e9)
        fill_array = mx.broadcast_to(fill_array, inverted_cross_attn_mask.shape)
        cross_attention_mask = mx.where(
            inverted_cross_attn_mask,
            fill_array,
            cross_attention_mask,
        )

        # Apply full-row bias
        full_text_row_masked_out_mask = mx.any(
            cross_attention_mask != -1e9,
            axis=-1,
            keepdims=True,
        )
        cross_attention_mask *= full_text_row_masked_out_mask

        return cross_attention_mask, full_text_row_masked_out_mask

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" not in key:
                key = key.replace("vision_model", "vision_tower")
            return key

        return {transform_key(k): v for k, v in weights.items()}
