from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class CallableModuleList(list):
    def __call__(self, x: mx.array):
        for item in self:
            x = item(x)
        return x


class KimiK25MultiModalProjector(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = (
            config.vision_config.hidden_size
            * config.vision_config.merge_kernel_size[0]
            * config.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)

        self.proj = CallableModuleList()
        self.proj.append(nn.Linear(self.hidden_size, self.hidden_size, bias=True))
        self.proj.append(nn.GELU())
        self.proj.append(
            nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=True)
        )

    def __call__(self, image_features: list[mx.array]) -> mx.array:
        outputs = []
        for item in image_features:
            h = self.pre_norm(item).reshape(item.shape[0], -1)
            h = self.proj(h)
            outputs.append(h)
        return outputs


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = KimiK25MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids), None, None, None

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
            grid_thw=grid_thw,
        )

        image_features = self.mm_projector(hidden_state)

        final_inputs_embeds, expanded_mask, position_ids, expanded_ids = (
            self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask
            )
        )
        return final_inputs_embeds, expanded_mask, position_ids, expanded_ids

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        inputs_embeds[:, image_positions, :] = image_features

        return inputs_embeds

    # This logic was co-authored with codex based on the custom modeling code
    def _merge_input_ids_with_image_features(
        self,
        image_features: list[mx.array],
        inputs_embeds: mx.array,
        input_ids: mx.array,
        attention_mask: Optional[mx.array],
    ):
        feature_lengths = [f.shape[0] for f in image_features]
        image_features = mx.concatenate(image_features, axis=0)
        embed_dim = image_features.shape[-1]

        image_token_index = self.config.media_placeholder_token_id
        pad_token_id = getattr(self.config, "pad_token_id", 0)
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)

        batch_size, _ = input_ids.shape
        left_padding = mx.sum(attention_mask[:, -1] == 0) == 0

        flat_ids = input_ids.reshape(-1)
        token_occupation = mx.ones_like(flat_ids).astype(mx.int32)
        image_pos = np.where(np.array(flat_ids) == image_token_index)[0]
        image_pos = mx.array(image_pos, dtype=mx.int32)
        token_occupation[image_pos] = mx.array(
            feature_lengths, dtype=token_occupation.dtype
        )
        token_occupation = token_occupation.reshape(input_ids.shape)

        max_embed_dim = int(mx.max(mx.sum(token_occupation, axis=1)).item())
        new_token_positions = mx.cumsum(token_occupation, axis=1) - 1
        nb_image_pad = (max_embed_dim - 1) - new_token_positions[:, -1]
        if left_padding:
            new_token_positions = new_token_positions + nb_image_pad[:, None]

        non_image = np.where(np.array(input_ids) != image_token_index)
        batch_indices = mx.array(non_image[0], dtype=mx.int32)
        non_image_indices = mx.array(non_image[1], dtype=mx.int32)
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        final_embedding = mx.zeros(
            (batch_size, max_embed_dim, embed_dim),
            dtype=inputs_embeds.dtype,
        )
        final_attention_mask = mx.zeros(
            (batch_size, max_embed_dim),
            dtype=mx.int32,
        )
        final_input_ids = mx.full(
            (batch_size, max_embed_dim),
            pad_token_id,
            dtype=mx.int32,
        )

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[
            batch_indices, non_image_indices
        ]

        image_to_overwrite = mx.ones((batch_size, max_embed_dim), dtype=mx.bool_)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= (
            mx.cumsum(image_to_overwrite, axis=1) - 1
        ) >= nb_image_pad[:, None]

        if int(mx.sum(image_to_overwrite).item()) != image_features.shape[0]:
            raise ValueError(
                "The input provided to the model are wrong. The number of image tokens is "
                f"{int(mx.sum(image_to_overwrite).item())} while the number of image features "
                f"given to the model is {image_features.shape[0]}."
            )

        image_indices = np.where(np.array(image_to_overwrite))
        image_batch = mx.array(image_indices[0], dtype=mx.int32)
        image_pos = mx.array(image_indices[1], dtype=mx.int32)
        final_embedding[image_batch, image_pos] = image_features.reshape(-1, embed_dim)
        final_attention_mask = mx.logical_or(final_attention_mask, image_to_overwrite)
        final_input_ids[image_batch, image_pos] = image_token_index

        position_ids = mx.cumsum(final_attention_mask, axis=1) - 1
        position_ids = mx.where(
            final_attention_mask, position_ids, mx.ones_like(position_ids)
        )

        pad_positions = np.where(np.array(attention_mask) == 0)
        if pad_positions[0].size > 0:
            pad_batch_indices = mx.array(pad_positions[0], dtype=mx.int32)
            pad_indices = mx.array(pad_positions[1], dtype=mx.int32)
            indices_to_mask = new_token_positions[pad_batch_indices, pad_indices]
            final_embedding[pad_batch_indices, indices_to_mask] = 0

        return final_embedding, final_attention_mask, position_ids, final_input_ids

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        cache=None,
        **kwargs,
    ):
        print("__call__")
        image_grid_thw = kwargs.pop("image_grid_hws", None)
        video_grid_thw = kwargs.pop("video_grid_hws", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        input_embeddings, _, _, _ = self.get_input_embeddings(
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
