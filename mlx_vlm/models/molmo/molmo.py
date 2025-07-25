import glob
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ) -> Dict[str, Union[mx.array, List[Tuple[mx.array, mx.array]]]]:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        batch_size, seq_len = input_ids.shape

        image_input_idx = kwargs.get("image_input_idx", None)
        image_masks = kwargs.get("image_masks", None)

        if pixel_values is not None:
            assert (
                image_masks is not None and image_input_idx is not None
            ), "image_masks and image_input_idx must be provided when images are given"

            dtype = self.vision_tower.image_vit.patch_embedding.weight.dtype
            pixel_values = pixel_values.astype(dtype)

            # Process images
            if pixel_values.ndim == 3:
                pixel_values = mx.expand_dims(pixel_values, 0)
                image_masks = (
                    mx.expand_dims(image_masks, 0) if image_masks is not None else None
                )
                image_input_idx = (
                    mx.expand_dims(image_input_idx, 0)
                    if image_input_idx is not None
                    else None
                )

            image_features, cls_embed = self.vision_tower(pixel_values, image_masks)

            # Insert image features into the input embeddings
            num_image, num_patch = image_features.shape[1:3]

            assert image_input_idx.shape == (
                batch_size,
                num_image,
                num_patch,
            ), f"image_input_idx.shape: {image_input_idx.shape}, expected: {(batch_size, num_image, num_patch)}"

            # Insert image features into the input embeddings
            image_features = image_features.reshape(
                batch_size, num_image * num_patch, -1
            )
            image_input_idx = image_input_idx.reshape(batch_size, num_image * num_patch)

            valid = np.where(image_input_idx >= 0)[0].tolist()
            batch_idx = mx.arange(batch_size)
            batch_idx = mx.tile(batch_idx[:, None], [1, image_features.shape[1]])

            input_embeddings = self.language_model.model.wte(input_ids)
            input_embeddings[
                batch_idx[valid], image_input_idx[valid]
            ] += image_features[valid]
        else:
            input_embeddings = None

        # Forward pass through the language model
        logits = self.language_model(
            input_ids,
            inputs_embeds=input_embeddings,
            mask=mask,
            cache=cache,
        )

        return logits

    def sanitize(self, weights):
        def transform_key(key):
            if "model.transformer" in key:
                key = key.replace("model.transformer", "language_model.model")
            if "model.vision_backbone" in key:
                key = key.replace("model.vision_backbone", "vision_tower")
            return key

        return {transform_key(k): v for k, v in weights.items()}
