from __future__ import annotations

from dataclasses import replace
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


# -----------------------------
# Utils
# -----------------------------
def l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
    denom = mx.sqrt(mx.maximum((x * x).sum(axis=-1, keepdims=True), eps))
    return x / denom


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
) -> mx.array:

    final_embedding_shape = final_embedding.shape

    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    return mx.reshape(final_embedding_flattened, final_embedding_shape)


# -----------------------------
# Backbone (Qwen3-VL)
# -----------------------------
class VLMBackbone(nn.Module):


    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            return {
                "inputs_embeds": self.language_model.model.embed_tokens(input_ids),
                "visual_pos_masks": None,
                "deepstack_visual_embeds": None,
            }

        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        hidden_states, deepstack_image_embeds = self.vision_tower(pixel_values, grid_thw)

        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )

        image_mask = image_mask[..., 0]
        return {
            "inputs_embeds": inputs_embeds,
            "visual_pos_masks": image_mask,
            "deepstack_visual_embeds": deepstack_image_embeds,
        }

    @staticmethod
    def merge_input_ids_with_image_features(
        image_features, inputs_embeds, input_ids, image_token_index, video_token_index
    ):
        special_image_mask = (input_ids == image_token_index) | (input_ids == video_token_index)
        n_image_tokens = special_image_mask.sum()

        special_image_mask = special_image_mask[..., None]
        special_image_mask = mx.broadcast_to(special_image_mask, inputs_embeds.shape)

        n_image_features = image_features.shape[0]
        n_image_mask_elements = special_image_mask.sum()
        if n_image_mask_elements != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: "
                f"tokens: {n_image_tokens}, features {n_image_features}"
            )

        inputs_embeds = masked_scatter(inputs_embeds, special_image_mask, image_features)
        return inputs_embeds, special_image_mask

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        inputs = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs.update({"pixel_values": pixel_values, **inputs})
        logits = self.language_model(input_ids, mask=mask, cache=cache, **kwargs)
        return logits


# -----------------------------
# ColQwen3 Wrapper (Tomoro)
# -----------------------------
class Model(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        backbone_cfg = replace(config, model_type="qwen3_vl")
        self.vlm = VLMBackbone(backbone_cfg)

        hidden = config.text_config.hidden_size
        self.embedding_proj_layer = nn.Linear(hidden, config.embed_dim, bias=True)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        return self.vlm(input_ids, pixel_values=pixel_values, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights: dict) -> dict:
        out = {}
        for k, v in weights.items():
            # ---- ColQwen3/Tomoro özel fixler ----
            # HF: vlm.model.language_model.layers...  -> MLX: vlm.language_model.model.layers...
            if k.startswith("vlm.model.language_model."):
                k = "vlm.language_model.model." + k[len("vlm.model.language_model."):]
                out[k] = v
                continue

            # HF: vlm.lm_head.weight -> MLX: vlm.language_model.lm_head.weight
            if k.startswith("vlm.lm_head."):
                k = "vlm.language_model.lm_head." + k[len("vlm.lm_head."):]
                out[k] = v
                continue

            # HF: vlm.model.visual.*  -> MLX: vlm.vision_tower.*
            if k.startswith("vlm.model.visual."):
                k = "vlm.vision_tower." + k[len("vlm.model.visual."):]
                out[k] = v
                continue

            # HF: vlm.model.vision_tower.* -> MLX: vlm.vision_tower.*
            if k.startswith("vlm.model.vision_tower."):
                k = "vlm.vision_tower." + k[len("vlm.model.vision_tower."):]
                out[k] = v
                continue

            if k.startswith("embedding_proj_layer."):
                out[k] = v
                continue

            if k.startswith("model.language_model."):
                k = "vlm.language_model.model." + k[len("model.language_model."):]
                out[k] = v
                continue
            if k.startswith("model.visual."):
                k = "vlm.vision_tower." + k[len("model.visual."):]
                out[k] = v
                continue

            out[k] = v

        return out