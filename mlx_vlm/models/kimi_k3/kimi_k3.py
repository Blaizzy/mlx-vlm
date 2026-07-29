from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class CallableModuleList(list):
    def __call__(self, x: mx.array):
        for item in self:
            x = item(x)
        return x


class KimiK3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        vc = config.vision_config
        self.hidden_size = (
            vc.mm_hidden_size * vc.merge_kernel_size[0] * vc.merge_kernel_size[1]
        )

        self.proj = CallableModuleList()
        self.proj.append(nn.Linear(self.hidden_size, self.hidden_size, bias=False))
        self.proj.append(nn.GELU())
        self.proj.append(
            nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=False)
        )
        self.post_norm = nn.RMSNorm(
            config.text_config.hidden_size, eps=vc.projector_ln_eps
        )

    def __call__(self, image_features: mx.array) -> mx.array:
        h = image_features.reshape(image_features.shape[0], -1)
        return self.post_norm(self.proj(h))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.mm_projector = KimiK3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        grid_thws = kwargs.pop("grid_thws", None)
        if grid_thws is None:
            grid_thws = kwargs.pop("image_grid_hws", None)
        if grid_thws is None:
            grid_thws = kwargs.pop("video_grid_hws", None)
        image_token_id = kwargs.pop("image_token_id", None)

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            hidden_state = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                grid_thws=grid_thws,
            )
            image_features = self.mm_projector(hidden_state)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids, image_token_id=image_token_id
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

        batch_positions, token_positions = np.where(np.array(image_mask))
        num_image_tokens = len(token_positions)
        num_image_features = image_features.shape[0]
        if num_image_tokens != num_image_features:
            raise ValueError(
                "Number of image placeholder tokens does not match extracted image "
                f"features: {num_image_tokens} tokens for {num_image_features} "
                f"features. Candidate token IDs: {candidate_token_ids}."
            )

        inputs_embeds[mx.array(batch_positions), mx.array(token_positions)] = (
            image_features.astype(inputs_embeds.dtype)
        )
        return inputs_embeds

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)

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
        embedding_output = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=embedding_output.inputs_embeds,
        )
        return logits

    def sanitize(self, weights):
        return weights
