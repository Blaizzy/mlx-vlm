from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


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
        grid_shapes = kwargs.pop("_grid_shapes", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            hidden_state = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                output_hidden_states=True,
                grid_thw=grid_thw,
                grid_shapes=grid_shapes,
            )

            image_features = self.multi_modal_projector(hidden_state)

        image_mask = mx.zeros(input_ids.shape, dtype=mx.bool_)
        for tid in [
            image_token_id,
            self.config.image_token_index,
            getattr(self.config, "media_placeholder_token_id", None),
        ]:
            if tid is not None:
                if isinstance(tid, mx.array) and tid.size == 0:
                    continue
                image_mask = image_mask | (input_ids == tid)

        mask_flat = image_mask.reshape(-1)
        cumsum = mx.cumsum(mask_flat.astype(mx.int32)) - 1
        feat_idx = mx.where(mask_flat, cumsum, 0).reshape(input_ids.shape)
        gathered = image_features[feat_idx]
        inputs_embeds = mx.where(image_mask[..., None], gathered, inputs_embeds)

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

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
