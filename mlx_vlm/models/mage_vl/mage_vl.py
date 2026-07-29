"""Mage-VL (microsoft/Mage-VL) for MLX / mlx-vlm.

Mage-ViT vision tower (codec-native, 3D-RoPE, 2x2 merger) + Qwen3-4B text
backbone. The LLM uses plain 1D position ids; image features are spliced at
`image_token_id` via masked_scatter (Qwen2-VL style)."""
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def masked_scatter(final_embedding, mask_expanded, features):
    shape = final_embedding.shape
    feats = mx.flatten(features)
    flat = mx.flatten(final_embedding)
    pos = mx.array(np.where(mx.flatten(mask_expanded))[0], mx.uint32)
    flat[pos] = feats.astype(flat.dtype)
    return mx.reshape(flat, shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        grid_thw = kwargs.get("image_grid_thw", kwargs.get("video_grid_thw", None))
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        if hasattr(pixel_values, "astype"):
            pixel_values = pixel_values.astype(dtype)
        image_features = self.vision_tower(pixel_values, grid_thw)

        tok = self.config.image_token_id
        vtok = self.config.video_token_id
        mask = (input_ids == tok) | (input_ids == vtok)
        mask = mx.broadcast_to(mask[..., None], inputs_embeds.shape)
        if int(mask.sum()) != image_features.size:
            raise ValueError(
                f"image tokens != features: mask={int(mask.sum())} feat={image_features.size}"
            )
        inputs_embeds = masked_scatter(inputs_embeds, mask, image_features)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, inputs_embeds=feats.inputs_embeds, mask=mask, cache=cache
        )

    def sanitize(self, weights):
        w = {}
        for k, v in weights.items():
            if k.startswith("model.language_model"):
                k = k.replace("model.language_model", "language_model.model")
            elif k.startswith("model.visual"):
                k = k.replace("model.visual", "vision_tower")
            elif k.startswith("lm_head"):
                k = k.replace("lm_head", "language_model.lm_head")
            w[k] = v
        # vision-specific fixups (patch_embedding reshape, merger rename)
        vt = {k[len("vision_tower.") :]: v for k, v in w.items() if k.startswith("vision_tower.")}
        vt = self.vision_tower.sanitize(vt)
        w = {k: v for k, v in w.items() if not k.startswith("vision_tower.")}
        for k, v in vt.items():
            w["vision_tower." + k] = v
        return w
