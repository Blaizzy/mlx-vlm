from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoProcessor

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .processing_kimi_vl import KimiVLImageProcessor, KimiVLProcessor
from .vision import VisionModel

# Register custom processor classes for kimi_vl model type
try:
    MODEL_TYPE = "kimi_vl"
    AutoImageProcessor.register(
        MODEL_TYPE, slow_image_processor_class=KimiVLImageProcessor
    )
    AutoProcessor.register(MODEL_TYPE, KimiVLProcessor)
except Exception:
    raise Exception("Failed to register kimi_vl processor")


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
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_hws", None)
        video_grid_thw = kwargs.pop("video_grid_hws", None)
        image_token_id = kwargs.pop("image_token_id", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        hidden_state = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
            grid_thw=grid_thw,
        )

        image_features = self.multi_modal_projector(hidden_state)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features,
            inputs_embeds,
            input_ids,
            image_token_id=image_token_id,
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _prepare_inputs_for_multimodal(
        self,
        image_features,
        inputs_embeds,
        input_ids,
        image_token_id=None,
    ):
        candidate_token_ids = []
        for token_id in [
            image_token_id,
            self.config.image_token_index,
            getattr(self.config, "media_placeholder_token_id", None),
        ]:
            if token_id is None:
                continue
            if isinstance(token_id, mx.array):
                if token_id.size == 0:
                    continue
                token_id = token_id.item()
            token_id = int(token_id)
            if token_id not in candidate_token_ids:
                candidate_token_ids.append(token_id)

        image_mask = mx.zeros(input_ids.shape, dtype=mx.bool_)
        for token_id in candidate_token_ids:
            image_mask = mx.logical_or(image_mask, input_ids == token_id)

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(np.array(image_mask))[1].tolist()
        num_image_tokens = len(image_positions)
        num_image_features = image_features.shape[0]
        if num_image_tokens != num_image_features:
            raise ValueError(
                "Number of image placeholder tokens does not match extracted image features: "
                f"{num_image_tokens} tokens for {num_image_features} features. "
                f"Candidate token IDs: {candidate_token_ids}."
            )

        inputs_embeds[:, image_positions, :] = image_features

        return inputs_embeds

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

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits

    def sanitize(self, weights):
        return {
            k.replace("encoder.", "") if "vision_tower" in k else k: v
            for k, v in weights.items()
        }
