from itertools import accumulate
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.visual = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_image_features(
        self, pixel_values: mx.array, image_grid_thw: Optional[mx.array] = None
    ):
        pixel_values = pixel_values.astype(self.visual.patch_embed.proj.weight.dtype)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, image_grid_thw, output_hidden_states=False
        )
        split_sizes = [
            int((thw[0] * thw[1] * thw[2]).item() // self.visual.spatial_merge_size**2)
            for thw in image_grid_thw
        ]
        if len(split_sizes) > 1:
            split_indices = list(accumulate(split_sizes[:-1]))
            image_embeds = mx.split(image_embeds, split_indices, axis=0)
        else:
            image_embeds = [image_embeds]
        return image_embeds, deepstack_image_embeds

    def get_video_features(
        self, pixel_values_videos: mx.array, video_grid_thw: Optional[mx.array] = None
    ):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size = input_ids.shape[0]
        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]
                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]
                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )
                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        batch_size, _ = input_ids.shape
        
        image_features = None
        if pixel_values is not None:
            image_embeds, _ = self.get_image_features(pixel_values, image_grid_thw)
            image_features = mx.concatenate(image_embeds, axis=0).astype(inputs_embeds.dtype)
        
        video_features = None
        if pixel_values_videos is not None:
            video_embeds, _ = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_features = mx.concatenate(video_embeds, axis=0).astype(inputs_embeds.dtype)
        
        if image_features is None and video_features is None:
            return inputs_embeds
        
        batch_outputs = []
        feature_start_idx = 0
        
        for batch_idx in range(batch_size):
            batch_output = inputs_embeds[batch_idx]
            
            if image_features is not None:
                image_mask = input_ids[batch_idx] == self.config.image_token_id
                num_image_positions = mx.sum(image_mask).item()
                
                if num_image_positions > 0:
                    batch_features = image_features[
                        feature_start_idx : feature_start_idx + num_image_positions
                    ]
                    cumsum = mx.cumsum(image_mask.astype(mx.int32))
                    feature_indices = mx.where(image_mask, cumsum - 1, 0)
                    gathered_features = batch_features[feature_indices]
                    image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                    batch_output = mx.where(
                        image_mask_expanded, gathered_features, batch_output
                    )
                    feature_start_idx += num_image_positions
            
            if video_features is not None:
                video_mask = input_ids[batch_idx] == self.config.video_token_id
                num_video_positions = mx.sum(video_mask).item()
                
                if num_video_positions > 0:
                    batch_features = video_features[
                        feature_start_idx : feature_start_idx + num_video_positions
                    ]
                    cumsum = mx.cumsum(video_mask.astype(mx.int32))
                    feature_indices = mx.where(video_mask, cumsum - 1, 0)
                    gathered_features = batch_features[feature_indices]
                    video_mask_expanded = mx.expand_dims(video_mask, axis=-1)
                    batch_output = mx.where(
                        video_mask_expanded, gathered_features, batch_output
                    )
                    feature_start_idx += num_video_positions
            
            batch_outputs.append(batch_output)
        
        return mx.stack(batch_outputs, axis=0)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        inputs_embeds = self.get_input_embeddings(
            input_ids, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
        )

        return self.language_model(
            input_ids,
            inputs_embeds,
            mask=mask,
            cache=cache,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_tower."):
                key = key.replace("vision_tower.", "visual.")
            elif "visual" in key and key.startswith("model.visual"):
                key = key.replace("model.visual", "visual")
            sanitized[key] = value
        return sanitized