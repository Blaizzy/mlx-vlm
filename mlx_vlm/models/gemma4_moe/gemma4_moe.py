import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..gemma4.vision import VisionModel
from .config import ModelConfig
from .language import LanguageModel, RMSNormNoScale


def masked_scatter(input_tensor, mask, source):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(
        input_tensor.shape
    )


class MultimodalEmbedder(nn.Module):
    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(
            embedding_dim, text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = RMSNormNoScale(text_hidden_size, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        proj = self.embedding_projection(inputs_embeds)
        return self.embedding_post_projection_norm(proj)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.embed_vision(image_features)
            image_features = image_features.astype(inputs_embeds.dtype)

            image_mask = input_ids == self.config.image_token_id
            image_mask_expanded = mx.expand_dims(image_mask, -1)
            image_mask_expanded = mx.broadcast_to(
                image_mask_expanded, inputs_embeds.shape
            )
            inputs_embeds = masked_scatter(
                inputs_embeds, image_mask_expanded, image_features
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )
        return self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )

    def sanitize(self, weights):
        sanitized = {}
        # Check if vision uses clipped linears
        use_clipped = getattr(self.config.vision_config, "use_clipped_linears", True)
        for k, v in weights.items():
            if any(
                s in k
                for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                if not use_clipped or "vision_tower" not in k:
                    continue
            if "rotary_emb.inv_freq" in k or "rotary_emb" in k:
                continue
            # No audio tower
            if "audio_tower" in k or "embed_audio" in k:
                continue

            if k.startswith("model."):
                new_key = k[len("model."):]
            else:
                new_key = k

            # language_model.xxx -> language_model.model.xxx
            if new_key.startswith("language_model."):
                rest = new_key[len("language_model."):]
                new_key = "language_model.model." + rest

            # Drop layer_scalar for sliding attention layers
            if "layer_scalar" in new_key and "language_model" in new_key:
                m = re.search(r"layers\.(\d+)\.layer_scalar", new_key)
                if m:
                    layer_idx = int(m.group(1))
                    layer_types = self.config.text_config.layer_types
                    if (
                        layer_idx < len(layer_types)
                        and layer_types[layer_idx] != "full_attention"
                    ):
                        continue

            sanitized[new_key] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
