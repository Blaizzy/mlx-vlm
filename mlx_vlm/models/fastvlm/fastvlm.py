from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import re

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel, CallableModuleList


def build_vision_projector(config):
    hidden_size = config.text_config.hidden_size
    projector_type = getattr(config, "mm_projector_type", "mlp2x_gelu")
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = CallableModuleList()
        modules.append(nn.Linear(config.mm_hidden_size, hidden_size))
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return modules
    raise ValueError(f'Unknown projector type: {projector_type}')


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = build_vision_projector(config)

    # def get_input_embeddings(
    #     self,
    #     input_ids: Optional[mx.array] = None,
    #     pixel_values: Optional[mx.array] = None,
    # ):
    #     if pixel_values is None:
    #         return self.language_model.model.embed_tokens(input_ids)

    #     # Get the input embeddings from the language model
    #     inputs_embeds = self.language_model.model.embed_tokens(input_ids)

    #     # Get the ouptut hidden states from the vision model
    #     *_, hidden_states = self.vision_tower(
    #         pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
    #     )

    #     # Select the hidden states from the desired layer
    #     selected_image_feature = hidden_states[self.vision_feature_layer]

    #     if isinstance(self.vision_feature_layer, int):
    #         if self.vision_feature_select_strategy == "default":
    #             selected_image_feature = selected_image_feature[:, 1:]

    #     else:
    #         hs_pool = [
    #             hidden_states[layer_idx] for layer_idx in self.vision_feature_layer
    #         ]
    #         # For default; crop CLS from each hidden state in the hidden state pool
    #         if self.vision_feature_select_strategy == "default":
    #             hs_pool = [hs[:, 1:] for hs in hs_pool]
    #         selected_image_feature = mx.concatenate(hs_pool, axis=-1)

    #     # Pass image features through the multi-modal projector
    #     image_features = self.multi_modal_projector(selected_image_feature)

    #     # Insert special image tokens in the input_ids
    #     final_inputs_embeds = self._merge_input_ids_with_image_features(
    #         image_features, inputs_embeds, input_ids
    #     )
    #     return final_inputs_embeds

    # def _merge_input_ids_with_image_features(
    #     self, image_features, inputs_embeds, input_ids
    # ):
    #     image_token_index = self.config.image_token_index

    #     # Positions of <image> tokens in input_ids, assuming batch size is 1
    #     image_positions = np.where(input_ids == image_token_index)[1].tolist()
    #     num_images, _, vision_hidden_size = image_features.shape

    #     reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

    #     # cast to the dtype of the input_embeds to support quantized models
    #     reshaped_image_hidden_states = reshaped_image_hidden_states.astype(
    #         inputs_embeds.dtype
    #     )

    #     # Pad image_positions to match the length of reshaped_image_hidden_states
    #     num_positions_needed = len(image_positions)

    #     if reshaped_image_hidden_states.shape[0] > num_positions_needed:
    #         # TODO: Think about how to handle this case
    #         raise ValueError(
    #             "Llava model supports only one image per input. Please check your input_ids and pixel_values."
    #         )

    #     inputs_embeds[:, image_positions, :] = reshaped_image_hidden_states
    #     return inputs_embeds

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids, mask=mask, cache=cache, inputs_embeds=input_embddings
        )
        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "vision_tower" in key:
                key = key.replace("model.vision_tower.vision_tower.model", "vision_tower.vision_model")
                key = key.replace("patch_embed", "patch_embed.blocks")
                return key
            if "lm_head" in key:
                return key
            if "mm_projector" in key:
                return key.replace("model.", "")
            return "language_model." + key

        return {transform_key(k): v for k, v in weights.items()}
