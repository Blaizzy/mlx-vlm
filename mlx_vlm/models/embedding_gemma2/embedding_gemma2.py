from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..gemma4.audio import AudioEncoder
from ..gemma4.gemma4 import MultimodalEmbedder
from ..gemma4.vision import VisionModel
from ..pooling import EmbeddingOutput, mean_pooling, normalize_embeddings
from .config import ModelConfig
from .language import TextModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = TextModel(config.text_config)
        self.vision_tower = self.embed_vision = None
        self.audio_tower = self.embed_audio = None
        if config.vision_config is not None:
            self.vision_tower = VisionModel(config.vision_config)
            self.embed_vision = MultimodalEmbedder(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
                config.vision_config.rms_norm_eps,
            )
        if config.audio_config is not None:
            self.audio_tower = AudioEncoder(config.audio_config)
            self.embed_audio = MultimodalEmbedder(
                config.audio_config.output_proj_dims or config.audio_config.hidden_size,
                config.text_config.hidden_size,
                config.audio_config.rms_norm_eps,
            )

    def get_image_features(self, pixel_values, image_position_ids):
        if self.vision_tower is None:
            raise ValueError("Image and video inputs require vision_config")
        return self.embed_vision(self.vision_tower(pixel_values, image_position_ids))[0]

    def get_video_features(self, pixel_values_videos, video_position_ids):
        return self.get_image_features(pixel_values_videos, video_position_ids)

    def get_audio_features(self, input_features, input_features_mask=None):
        if self.audio_tower is None:
            raise ValueError("Audio inputs require audio_config")
        invalid = (
            mx.zeros(input_features.shape[:2], dtype=mx.bool_)
            if input_features_mask is None
            else ~input_features_mask.astype(mx.bool_)
        )
        features, invalid = self.audio_tower(input_features, invalid)
        features = self.embed_audio(features)
        return mx.concatenate(
            [
                features[i, : int((~mask).sum().item())]
                for i, mask in enumerate(invalid)
            ],
            axis=0,
        )

    @staticmethod
    def _scatter(embeddings, input_ids, token_id, features):
        mask = input_ids == token_id
        count = int(mask.sum().item())
        if count != features.shape[0]:
            raise ValueError(
                f"Media token count ({count}) does not match feature count ({features.shape[0]})"
            )
        if count == 0:
            return embeddings
        indices = mx.maximum(mx.cumsum(mask.reshape(-1)) - 1, 0)
        aligned = features[indices].reshape(embeddings.shape)
        return mx.where(mask[..., None], aligned.astype(embeddings.dtype), embeddings)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_position_ids: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        video_position_ids: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        if attention_mask is None:
            attention_mask = kwargs.get("mask")
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        media_mask = (
            (input_ids == self.config.image_token_id)
            | (input_ids == self.config.video_token_id)
            | (input_ids == self.config.audio_token_id)
        )
        text_ids = mx.where(media_mask, self.config.text_config.pad_token_id, input_ids)
        embeddings = self.language_model.embed_tokens(text_ids)
        embeddings = embeddings * mx.array(
            self.config.text_config.hidden_size**0.5, dtype=embeddings.dtype
        )
        if pixel_values is not None:
            embeddings = self._scatter(
                embeddings,
                input_ids,
                self.config.image_token_id,
                self.get_image_features(pixel_values, image_position_ids),
            )
        if pixel_values_videos is not None:
            embeddings = self._scatter(
                embeddings,
                input_ids,
                self.config.video_token_id,
                self.get_video_features(pixel_values_videos, video_position_ids),
            )
        if input_features is not None:
            embeddings = self._scatter(
                embeddings,
                input_ids,
                self.config.audio_token_id,
                self.get_audio_features(input_features, input_features_mask),
            )
        hidden_states = self.language_model(embeddings, attention_mask, position_ids)
        pooled = mean_pooling(hidden_states, attention_mask)
        return EmbeddingOutput(
            last_hidden_state=hidden_states,
            text_embeds=normalize_embeddings(pooled),
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if self.vision_tower is None and key.startswith(
                ("vision_tower.", "embed_vision.")
            ):
                continue
            if self.audio_tower is None and key.startswith(
                ("audio_tower.", "embed_audio.")
            ):
                continue
            if "rotary_emb.inv_freq" in key:
                continue
            if "subsample_conv_projection" in key and key.endswith("conv.weight"):
                channels = (
                    1
                    if ".layer0." in key
                    else self.config.audio_config.subsampling_conv_channels[0]
                )
                if value.shape[-1] != channels:
                    value = value.transpose(0, 2, 3, 1)
            if key.endswith("depthwise_conv1d.weight") and value.shape[-1] != 1:
                value = value.transpose(0, 2, 1)
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
