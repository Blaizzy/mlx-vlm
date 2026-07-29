from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, pixel_shuffle
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel, check_array_shape


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Attribute names match the HF checkpoint keys (vision_backbone / language_model / mlp1)
        self.vision_backbone = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

        self.downsample_ratio = config.downsample_ratio
        if config.select_layer != -1:
            raise ValueError(
                f"select_layer={config.select_layer} is not supported (only -1)"
            )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = [
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        ]

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = (
            self.vision_backbone.vision_model.embeddings.patch_embedding.weight.dtype
        )
        pixel_values = pixel_values.astype(dtype)

        if pixel_values.ndim == 5:
            pixel_values = pixel_values[0]

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
        else:
            # SigLIP: NCHW -> NHWC. No CLS token, so nothing is stripped from the front
            hidden_states, _, _ = self.vision_backbone(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=False
            )

            hidden_states = pixel_shuffle(
                hidden_states, shuffle_ratio=self.downsample_ratio
            )

            for layer in self.mlp1:
                hidden_states = layer(hidden_states)

        image_token_index = kwargs.get(
            "image_token_index", self.config.image_token_index
        )

        final_inputs_embeds = self._merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids, image_token_index
        )
        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, image_token_index=None
    ):
        B, N, C = inputs_embeds.shape
        if image_token_index is None:
            image_token_index = self.config.image_token_index

        image_positions = input_ids == image_token_index
        n_img = int(mx.sum(image_positions))
        image_features = image_features.reshape(-1, image_features.shape[-1])
        if n_img != image_features.shape[0]:
            raise ValueError(
                f"image token count ({n_img}) != vision feature count "
                f"({image_features.shape[0]})"
            )

        image_indices = np.where(image_positions)[1].tolist()
        inputs_embeds[:, image_indices, :] = image_features
        return inputs_embeds.reshape(B, N, C)

    @property
    def layers(self):
        return self.language_model.model.layers

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            # SigLIP's attention pooling head is unused when only last_hidden_state is consumed
            if ".vision_model.head." in k:
                continue
            if "patch_embedding.weight" in k:
                sanitized[k] = v if check_array_shape(v) else v.transpose(0, 2, 3, 1)
            else:
                sanitized[k] = v
        return sanitized

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            None, cache=cache, inputs_embeds=input_embeddings_features.inputs_embeds
        )
        return logits
