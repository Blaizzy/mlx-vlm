import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..pixtral import LanguageModel
from ..pixtral import Model as PixtralModel
from ..pixtral import TextConfig, VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 10
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    spatial_merge_size: int = 2

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def __call__(self, image_features: mx.array, image_sizes: mx.array) -> mx.array:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]
        image_features = image_features.astype(mx.bfloat16)
        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            mx.split(image_features, tokens_per_image)
        ):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            h = int(h)
            w = int(w)

            image_grid = image_tokens.reshape(h, w, d).transpose(2, 0, 1)[None, ...]
            # Implement unfold manually since mlx doesn't have unfold
            b, c, h, w = image_grid.shape
            patches = []
            for i in range(0, h - self.spatial_merge_size + 1, self.spatial_merge_size):
                for j in range(
                    0, w - self.spatial_merge_size + 1, self.spatial_merge_size
                ):
                    patch = image_grid[
                        :,
                        :,
                        i : i + self.spatial_merge_size,
                        j : j + self.spatial_merge_size,
                    ]
                    patches.append(patch.reshape(b, -1))
            grid = mx.concatenate(patches, axis=1).transpose()
            grid = grid.reshape(d * self.spatial_merge_size**2, -1).T
            permuted_tensor.append(grid)

        image_features = mx.concatenate(permuted_tensor, axis=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm = nn.RMSNorm(config.vision_config.hidden_size)
        self.patch_merger = Mistral3PatchMerger(config)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=False
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array, image_sizes: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.patch_merger(x, image_sizes)
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(PixtralModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        self.multi_modal_projector = Mistral3MultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        image_sizes = kwargs.get("image_sizes", None)

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the output hidden states from the vision model
        if isinstance(pixel_values, list):
            pixel_values = mx.concatenate(
                [mx.array(pv)[None, ...] for pv in pixel_values], axis=0
            )
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]

        # Pass pixel_values as list of images, as each image is individually run through conv2d and position encoding
        # Reference code from transformers: https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py#L479C9-L479C21
        # and mistral_inference: https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/vision_encoder.py#L85
        *_, hidden_states = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[self.vision_feature_layer]

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(selected_image_feature, image_sizes)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index
        num_images, num_image_patches, embed_dim = image_features.shape

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        # Split image features into separate embeddings for each image
        image_embeddings = mx.split(image_features, num_image_patches, axis=1)
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)
