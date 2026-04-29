from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .audio import (
    SoundEncoder,
    SoundFeatureExtractor,
    SoundProjection,
    SquaredReLU,
    sanitize_audio_weights,
)
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class VisionProjection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        scale = int(1 / config.downsample_ratio)
        in_features = config.vit_hidden_size * (scale**2)
        self.layers = [
            nn.RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, config.projector_hidden_size, bias=False),
            SquaredReLU(),
            nn.Linear(
                config.projector_hidden_size,
                config.text_config.hidden_size,
                bias=False,
            ),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def masked_scatter(
    final_embedding: mx.array,
    mask: mx.array,
    source: mx.array,
):
    final_shape = final_embedding.shape
    final_flat = mx.flatten(final_embedding)
    source_flat = mx.flatten(source)
    mask_flat = mx.flatten(mask)
    positions = mx.array(np.where(mask_flat)[0], mx.uint32)
    final_flat[positions] = source_flat
    return final_flat.reshape(final_shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_model = VisionModel(config.vision_config)
        self.mlp1 = VisionProjection(config)

        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id
        self.sound_context_token_id = config.sound_context_token_id
        self.video_temporal_patch_dim = config.video_temporal_patch_size
        self.video_pruning_rate = config.video_pruning_rate

        self.sound_encoder = None
        self.sound_projection = None
        self.sound_feature_extractor = None
        if config.sound_config is not None:
            self.sound_encoder = SoundEncoder(config.sound_config)
            self.sound_projection = SoundProjection(
                config.sound_config, config.text_config.hidden_size
            )
            self.sound_feature_extractor = SoundFeatureExtractor(config.sound_config)

    @property
    def layers(self):
        return self.language_model.layers

    def _merge_features(self, inputs_embeds, input_ids, token_id, features, name):
        if token_id is None:
            raise ValueError(f"{name} context token id is not configured.")

        features = features.reshape(-1, inputs_embeds.shape[-1]).astype(
            inputs_embeds.dtype
        )
        token_mask = input_ids == token_id
        n_tokens = int(mx.sum(token_mask).item())
        if n_tokens != features.shape[0]:
            raise ValueError(
                f"{name} token count ({n_tokens}) does not match feature count "
                f"({features.shape[0]})."
            )

        token_mask = mx.broadcast_to(token_mask[..., None], inputs_embeds.shape)
        return masked_scatter(inputs_embeds, token_mask, features)

    def _extract_sound_features(
        self,
        sound_clips=None,
        input_features=None,
        feature_attention_mask=None,
        feature_lengths=None,
    ):
        if sound_clips is None and input_features is None:
            return None

        if self.sound_encoder is None or self.sound_projection is None:
            raise RuntimeError("Sound encoder is not initialized for this model.")

        if sound_clips is not None:
            (
                input_features,
                feature_attention_mask,
                feature_lengths,
            ) = self.sound_feature_extractor(sound_clips)
        if not isinstance(input_features, mx.array):
            input_features = mx.array(input_features)
        input_features = input_features.astype(self.language_model.lm_head.weight.dtype)

        if feature_attention_mask is not None and not isinstance(
            feature_attention_mask, mx.array
        ):
            feature_attention_mask = mx.array(feature_attention_mask)

        sound_embeds = self.sound_encoder(input_features, feature_attention_mask)
        sound_embeds = self.sound_projection(sound_embeds)

        if feature_lengths is None and feature_attention_mask is not None:
            feature_lengths = mx.sum(feature_attention_mask, axis=-1).astype(mx.int32)

        if feature_lengths is None:
            return sound_embeds

        if not isinstance(feature_lengths, mx.array):
            feature_lengths = mx.array(feature_lengths, dtype=mx.int32)

        output_lengths = self.sound_encoder.encoder._get_subsampling_output_length(
            feature_lengths
        )
        pieces = [
            sound_embeds[i, : int(length)]
            for i, length in enumerate(output_lengths.tolist())
        ]
        return mx.concatenate(pieces, axis=0) if pieces else None

    def pixel_shuffle(self, x, scale_factor=0.5):
        batch, width, height, channels = x.shape
        x = x.reshape(
            batch,
            width,
            int(height * scale_factor),
            int(channels / scale_factor),
        )
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            batch,
            int(height * scale_factor),
            int(width * scale_factor),
            int(channels / (scale_factor * scale_factor)),
        )
        if self.config.ps_version != "v1":
            x = x.transpose(0, 2, 1, 3)
        return x

    def _ensure_4d_pixels(self, pixel_values):
        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(pixel_values)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values[None, ...]
        return pixel_values

    def _extract_feature_single(self, pixel_values):
        pixel_values = self._ensure_4d_pixels(pixel_values)
        vit_embeds = self.vision_model(pixel_values).features
        patch_size = self.vision_model.radio_model.model.patch_generator.patch_size
        _, _, height, width = pixel_values.shape
        patch_h = height // patch_size
        patch_w = width // patch_size
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], patch_h, patch_w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.config.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def extract_feature(self, pixel_values):
        if isinstance(pixel_values, (list, tuple)):
            return mx.concatenate(
                [self._extract_feature_single(pv) for pv in pixel_values], axis=0
            )
        return self._extract_feature_single(pixel_values)

    def extract_video_feature(self, pixel_values_videos):
        if isinstance(pixel_values_videos, (list, tuple)):
            pixel_values_videos = mx.concatenate(
                [self._ensure_4d_pixels(pv) for pv in pixel_values_videos], axis=0
            )
        else:
            pixel_values_videos = self._ensure_4d_pixels(pixel_values_videos)

        temporal = self.video_temporal_patch_dim
        num_frames, channels, height, width = pixel_values_videos.shape
        if num_frames % temporal != 0:
            pad = temporal - (num_frames % temporal)
            padding = mx.broadcast_to(
                pixel_values_videos[-1:],
                (pad, channels, height, width),
            )
            pixel_values_videos = mx.concatenate([pixel_values_videos, padding], axis=0)
            num_frames = pixel_values_videos.shape[0]

        num_groups = num_frames // temporal
        x = pixel_values_videos.reshape(num_groups, temporal * channels, height, width)
        vit_embeds = self.vision_model(x, use_video_embedder=True).features

        patch_size = self.vision_model.radio_model.model.patch_generator.patch_size
        patch_h = height // patch_size
        patch_w = width // patch_size
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], patch_h, patch_w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.config.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return self.mlp1(vit_embeds)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if pixel_values is not None:
            image_features = self.extract_feature(pixel_values)
            inputs_embeds = self._merge_features(
                inputs_embeds,
                input_ids,
                self.img_context_token_id,
                image_features,
                "Image",
            )

        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            if self.video_pruning_rate > 0:
                raise NotImplementedError(
                    "Efficient video sampling is not implemented for Nemotron Omni yet."
                )
            video_features = self.extract_video_feature(pixel_values_videos)
            inputs_embeds = self._merge_features(
                inputs_embeds,
                input_ids,
                self.img_context_token_id,
                video_features,
                "Video",
            )

        feature_attention_mask = kwargs.get("feature_attention_mask", None)
        if feature_attention_mask is None:
            feature_attention_mask = kwargs.get("sound_attention_mask", None)

        feature_lengths = kwargs.get("audio_feature_lengths", None)
        if feature_lengths is None:
            feature_lengths = kwargs.get("sound_feature_lengths", None)

        sound_features = self._extract_sound_features(
            sound_clips=kwargs.get("sound_clips", None),
            input_features=kwargs.get("input_features", None),
            feature_attention_mask=feature_attention_mask,
            feature_lengths=feature_lengths,
        )
        if sound_features is not None:
            inputs_embeds = self._merge_features(
                inputs_embeds,
                input_ids,
                self.sound_context_token_id,
                sound_features,
                "Sound",
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        embedding_output = self.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        return self.language_model(
            input_ids,
            inputs_embeds=embedding_output.inputs_embeds,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in sanitize_audio_weights(weights).items():
            if key.startswith("mlp1."):
                key = key.replace("mlp1.0.", "mlp1.layers.0.")
                key = key.replace("mlp1.1.", "mlp1.layers.1.")
                key = key.replace("mlp1.3.", "mlp1.layers.3.")
            sanitized[key] = value
        return sanitized


__all__ = ["Model", "VisionModel", "LanguageModel", "LanguageModelOutput"]
