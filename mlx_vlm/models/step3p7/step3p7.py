from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def _masked_scatter(inputs_embeds, mask, image_features):
    shape = inputs_embeds.shape
    flat = inputs_embeds.reshape(-1)
    idx = mx.array(np.where(mask.reshape(-1))[0], dtype=mx.uint32)
    flat[idx] = image_features.reshape(-1)
    return flat.reshape(shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        self.vit_large_projector = nn.Linear(
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    @property
    def layers(self):
        return self.language_model.layers

    def _process_image_features(self, image_features):
        b, p, c = image_features.shape
        hw = int(p**0.5)
        x = image_features.reshape(b, hw, hw, c)
        x = self.vision_model.vit_downsampler1(x)
        x = self.vision_model.vit_downsampler2(x)
        b, h, w, c = x.shape
        return self.vit_large_projector(x.reshape(b, h * w, c))

    def _vision_embeddings(
        self,
        pixel_values=None,
        patch_pixel_values=None,
        num_patches=None,
        image_embeds=None,
        **kwargs,
    ):
        if image_embeds is not None:
            return [image_embeds.reshape(-1, image_embeds.shape[-1])]
        if pixel_values is None:
            return None
        if pixel_values.ndim >= 5:
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
        image_features = self._process_image_features(self.vision_model(pixel_values))
        patch_features = None
        if patch_pixel_values is not None and patch_pixel_values.size > 0:
            if patch_pixel_values.ndim >= 5:
                patch_pixel_values = patch_pixel_values.reshape(
                    -1, *patch_pixel_values.shape[-3:]
                )
            patch_features = self._process_image_features(
                self.vision_model(patch_pixel_values)
            )
        if num_patches is None:
            return [image_features.reshape(-1, image_features.shape[-1])]
        merged = []
        cursor = 0
        for i, count in enumerate(num_patches):
            count = int(count)
            parts = []
            if count > 0 and patch_features is not None:
                parts.append(
                    patch_features[cursor : cursor + count].reshape(
                        -1, patch_features.shape[-1]
                    )
                )
                cursor += count
            parts.append(image_features[i].reshape(-1, image_features.shape[-1]))
            merged.append(mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
        return merged

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        vision_embeddings = self._vision_embeddings(pixel_values=pixel_values, **kwargs)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if vision_embeddings is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)
        image_features = mx.concatenate(vision_embeddings, axis=0)
        image_mask = (input_ids == self.config.image_token_index)[..., None]
        image_mask = mx.broadcast_to(image_mask, inputs_embeds.shape)
        if image_mask.sum() != image_features.size:
            raise ValueError(
                f"Image features and image tokens do not match: tokens={int(image_mask.sum().item())}, features={image_features.shape[0]}"
            )
        inputs_embeds = _masked_scatter(inputs_embeds, image_mask, image_features)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith("model.layers."):
                parts = key.split(".")
                if (
                    len(parts) > 2
                    and parts[2].isdigit()
                    and int(parts[2]) >= self.config.text_config.num_hidden_layers
                ):
                    continue
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
                key = key.replace("model.", "language_model.model.", 1)
            elif key.startswith("model.vision_model."):
                key = key.replace("model.vision_model.", "vision_model.", 1)
            elif key.startswith("model.vit_large_projector."):
                key = key.replace(
                    "model.vit_large_projector.", "vit_large_projector.", 1
                )
            elif key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.", 1)
            elif (
                key.startswith("conv1.")
                or key.startswith("ln_pre.")
                or key.startswith("ln_post.")
                or key.startswith("transformer.")
                or key.startswith("vit_downsampler")
                or key == "positional_embedding"
                or key == "class_embedding"
            ):
                key = f"vision_model.{key}"
            if "vision_model." in key:
                sub = self.vision_model.sanitize(
                    {key.replace("vision_model.", "", 1): value}
                )
                for sk, sv in sub.items():
                    sanitized[f"vision_model.{sk}"] = sv
                continue
            sanitized[key] = value
        return self.language_model.sanitize(sanitized)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
