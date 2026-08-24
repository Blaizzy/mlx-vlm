from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def _replace_features(inputs_embeds, positions, features, label):
    expected = int(positions.sum().item())
    if expected != features.shape[0]:
        raise ValueError(
            f"{label} features and placeholder tokens do not match: "
            f"{features.shape[0]} features for {expected} tokens."
        )

    outputs = []
    offset = 0
    for batch_idx in range(inputs_embeds.shape[0]):
        row_mask = positions[batch_idx]
        count = int(row_mask.sum().item())
        row_features = features[offset : offset + count]
        if count:
            feature_index = mx.where(
                row_mask, mx.cumsum(row_mask.astype(mx.int32)) - 1, 0
            )
            gathered = row_features[feature_index]
            row = mx.where(row_mask[:, None], gathered, inputs_embeds[batch_idx])
        else:
            row = inputs_embeds[batch_idx]
        outputs.append(row)
        offset += count
    return mx.stack(outputs)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    def _video_grid(self, video_grid_thw):
        flattened = []
        for t, h, w in video_grid_thw.tolist():
            flattened.extend([[1, h, w]] * t)
        return mx.array(flattened, dtype=video_grid_thw.dtype)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        attention_mask = kwargs.get("mask")

        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        cached = kwargs.get("cached_image_features")

        in_video = mx.zeros(input_ids.shape, dtype=mx.bool_)
        if self.config.video_start_token_id is not None:
            starts = input_ids == self.config.video_start_token_id
            ends = input_ids == self.config.video_end_token_id
            in_video = mx.cumsum(starts.astype(mx.int32), axis=-1) > mx.cumsum(
                ends.astype(mx.int32), axis=-1
            )

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("`image_grid_thw` is required with image pixels.")
            if cached is None:
                dtype = self.vision_tower.patch_embed.proj.weight.dtype
                features = self.vision_tower(pixel_values.astype(dtype), image_grid_thw)
            else:
                features = cached
            image_positions = (input_ids == self.config.image_token_id) & ~in_video
            inputs_embeds = _replace_features(
                inputs_embeds,
                image_positions,
                features.astype(inputs_embeds.dtype),
                "Image",
            )

        if pixel_values_videos is not None:
            if video_grid_thw is None:
                raise ValueError("`video_grid_thw` is required with video pixels.")
            dtype = self.vision_tower.patch_embed.proj.weight.dtype
            features = self.vision_tower(
                pixel_values_videos.astype(dtype), self._video_grid(video_grid_thw)
            )
            video_positions = (input_ids == self.config.image_token_id) & in_video
            inputs_embeds = _replace_features(
                inputs_embeds,
                video_positions,
                features.astype(inputs_embeds.dtype),
                "Video",
            )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        features = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            attention_mask=features.attention_mask,
            cache=cache,
        )

    def sanitize(self, weights):
        remapped = {}
        for key, value in weights.items():
            if key.startswith("model.visual."):
                key = "vision_tower." + key[len("model.visual.") :]
            elif key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key == "lm_head.weight":
                key = "language_model.lm_head.weight"
            remapped[key] = value

        remapped = self.language_model.sanitize(remapped)
        return self.vision_tower.sanitize(remapped)

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    def make_cache(self):
        return self.language_model.make_cache()
