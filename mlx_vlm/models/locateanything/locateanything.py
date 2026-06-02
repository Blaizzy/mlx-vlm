from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..cache import KVCache
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class LocateAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        vit_hidden = config.vision_config.hidden_size
        merge = config.vision_config.merge_kernel_size
        self.input_dim = vit_hidden * merge[0] * merge[1]
        llm_hidden = config.text_config.hidden_size

        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.linear_1 = nn.Linear(self.input_dim, llm_hidden)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden, llm_hidden)

    def __call__(self, image_features: list[mx.array]) -> mx.array:
        h = mx.concatenate(image_features, axis=0).reshape(-1, self.input_dim)
        h = self.layer_norm(h)
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
        self.multi_modal_projector = LocateAnythingMultiModalProjector(config)
        self.image_token_index = config.image_token_index

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_hws = kwargs.pop("image_grid_hws", None)
        grid_shapes = kwargs.pop("_grid_shapes", None)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            image_features = cached
        else:
            hidden_state = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1),
                grid_thw=image_grid_hws,
                grid_shapes=grid_shapes,
            )
            image_features = self.multi_modal_projector(hidden_state)

        image_token_id = kwargs.pop("image_token_id", None) or self.image_token_index
        image_mask = input_ids == image_token_id

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
        pixel_values: mx.array = None,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        return self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )

    def make_cache(self):
        return [KVCache() for _ in self.language_model.model.layers]

    def pbd_generate(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        generation_mode: str = "hybrid",
        max_tokens: int = 2048,
        cache=None,
        **kwargs,
    ) -> List[int]:
        from .pbd import PBDDecoder

        embeds = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        if cache is None:
            cache = self.make_cache()
        decoder = PBDDecoder(self, generation_mode=generation_mode)
        return decoder.generate(
            input_ids, embeds.inputs_embeds, cache, max_tokens=max_tokens
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if k == "language_model.lm_head.weight":
                continue

            if k.startswith("vision_model."):
                k = k.replace("vision_model.encoder.", "vision_tower.").replace(
                    "vision_model.", "vision_tower."
                )
            elif k.startswith("mlp1."):
                k = (
                    k.replace("mlp1.0.", "multi_modal_projector.layer_norm.")
                    .replace("mlp1.1.", "multi_modal_projector.linear_1.")
                    .replace("mlp1.3.", "multi_modal_projector.linear_2.")
                )

            sanitized[k] = v
        return sanitized
