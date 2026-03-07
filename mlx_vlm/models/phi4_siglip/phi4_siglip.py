import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_phi4_siglip  # noqa: F401
from .config import ModelConfig, VisionConfig
from .language import LanguageModel
from .vision import VisionModel

IMAGE_TOKEN_INDEX = -200


class MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        self.linear_1 = nn.Linear(config.mm_hidden_size, hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_tower = VisionModel(config)

    def __call__(self, *args, **kwargs):
        return self.vision_tower(*args, **kwargs)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionTower(config.vision_config)
        self.mm_projector = MultiModalProjector(config)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        pixel_values=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_embeddings_features = self.get_input_embeddings(
                inputs, pixel_values, **kwargs
            )
            inputs_embeds = input_embeddings_features.inputs_embeds

        return self.language_model(
            inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache
        )

    def get_input_embeddings(
        self,
        inputs: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        spatial_shapes = kwargs.get("spatial_shapes", None)
        pixel_attention_mask = kwargs.get("pixel_attention_mask", None)

        inputs_embeds = self.language_model.model.embed_tokens(inputs)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        encoder_outputs, _, _ = self.vision_tower(
            pixel_values, output_hidden_states=True, spatial_shapes=spatial_shapes
        )
        hidden_states = encoder_outputs[self.config.mm_vision_select_layer]

        image_features_list = []
        if pixel_attention_mask is not None:
            for img_idx in range(hidden_states.shape[0]):
                valid_len = int(pixel_attention_mask[img_idx].sum().item())
                feature = hidden_states[img_idx, :valid_len, :]
                projected = self.mm_projector(feature)
                image_features_list.append(projected)
        else:
            for img_idx in range(hidden_states.shape[0]):
                feature = hidden_states[img_idx]
                projected = self.mm_projector(feature)
                image_features_list.append(projected)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features_list, inputs_embeds, inputs
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def _prepare_inputs_for_multimodal(image_features_list, inputs_embeds, input_ids):
        batch_size = input_ids.shape[0]
        new_embeds_list = []
        cur_image_idx = 0

        for b in range(batch_size):
            cur_input_ids = input_ids[b]
            cur_embeds = inputs_embeds[b]

            image_positions = np.where(np.array(cur_input_ids) == IMAGE_TOKEN_INDEX)[0]
            num_images = len(image_positions)

            if num_images == 0:
                new_embeds_list.append(cur_embeds)
                continue

            segments = []
            prev_pos = 0
            for i, pos in enumerate(image_positions):
                pos = int(pos)
                if pos > prev_pos:
                    segments.append(cur_embeds[prev_pos:pos])
                segments.append(image_features_list[cur_image_idx])
                cur_image_idx += 1
                prev_pos = pos + 1

            seq_len = int(cur_input_ids.shape[0])
            if prev_pos < seq_len:
                segments.append(cur_embeds[prev_pos:])

            new_embeds_list.append(mx.concatenate(segments, axis=0))

        if batch_size == 1:
            return new_embeds_list[0][None, :]
        else:
            max_len = max(e.shape[0] for e in new_embeds_list)
            embed_dim = new_embeds_list[0].shape[-1]
            padded = mx.zeros((batch_size, max_len, embed_dim))
            for i, emb in enumerate(new_embeds_list):
                padded[i, : emb.shape[0]] = emb
            return padded

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def head_dim(self):
        return (
            self.config.text_config.hidden_size
            // self.config.text_config.num_attention_heads
        )

    @property
    def n_kv_heads(self):
        return self.config.text_config.num_key_value_heads

    @property
    def vision_model(self):
        return self.vision_tower

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "position_ids" in k:
                continue
            if "vision_model.head." in k:
                continue

            new_key = k
            new_key = re.sub(r"mm_projector\.0\.", "mm_projector.linear_1.", new_key)
            new_key = re.sub(r"mm_projector\.2\.", "mm_projector.linear_2.", new_key)

            if new_key.startswith("model.vision_tower."):
                new_key = new_key[len("model.") :]
            elif new_key.startswith("model.mm_projector."):
                new_key = new_key[len("model.") :]
            elif new_key.startswith("model."):
                new_key = "language_model." + new_key
            elif new_key.startswith("lm_head."):
                new_key = "language_model." + new_key

            sanitized[new_key] = v

        return sanitized
