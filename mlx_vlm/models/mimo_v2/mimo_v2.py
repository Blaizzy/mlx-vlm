from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .audio import AudioEncoder, build_speech_embeddings
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = (
            None if config.skip_vision else VisionModel(config.vision_config)
        )
        self.speech_embeddings = build_speech_embeddings(config.audio_config)
        self.audio_encoder = AudioEncoder(config.audio_config)

    def _encode_vision(self, pixel_values, grid_thw, modality):
        if self.vision_tower is None:
            raise ValueError(f"Cannot encode {modality} when vision is disabled")
        if grid_thw is None:
            raise ValueError(f"{modality}_grid_thw is required with pixel values")
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        return self.vision_tower(pixel_values.astype(dtype), grid_thw)

    def encode_image(self, pixel_values, image_grid_thw=None):
        return self._encode_vision(pixel_values, image_grid_thw, "image")

    def encode_images(self, pixel_values, **kwargs):
        return [self.encode_image(pixel_values, kwargs.get("image_grid_thw"))]

    def encode_video(self, pixel_values, video_grid_thw=None):
        return self._encode_vision(pixel_values, video_grid_thw, "video")

    def encode_audio(self, audio_codes):
        return self.audio_encoder(audio_codes, self.speech_embeddings)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        modalities = (
            (
                self.config.image_token_id,
                pixel_values,
                kwargs.get("image_grid_thw"),
                kwargs.get("cached_image_features"),
                self.encode_image,
            ),
            (
                self.config.video_token_id,
                kwargs.get("pixel_values_videos", kwargs.get("video_pixel_values")),
                kwargs.get("video_grid_thw"),
                kwargs.get("cached_video_features"),
                self.encode_video,
            ),
        )
        for token_id, pixels, grid, cached, encode in modalities:
            if pixels is None and cached is None:
                continue
            if cached is None:
                cached = encode(pixels, grid)
            if not isinstance(cached, mx.array):
                cached = mx.concatenate(list(cached), axis=0)
            inputs_embeds = self._replace_modal_embeddings(
                input_ids, inputs_embeds, token_id, cached
            )
        audio_codes = kwargs.get("audio_codes")
        audio_features = kwargs.get("cached_audio_features", kwargs.get("audio_embeds"))
        if audio_codes is not None or audio_features is not None:
            if audio_features is None:
                audio_features = self.encode_audio(audio_codes)
            inputs_embeds = self._replace_modal_embeddings(
                input_ids,
                inputs_embeds,
                self.config.audio_token_id,
                audio_features,
            )
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def _replace_modal_embeddings(input_ids, inputs_embeds, token_id, features):
        if features.ndim != 2:
            raise ValueError(
                f"Modal features must be 2D, received shape {features.shape}"
            )
        if features.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(
                f"Modal feature width {features.shape[-1]} does not match "
                f"text embedding width {inputs_embeds.shape[-1]}"
            )
        mask = input_ids == token_id
        positions = [i for i, value in enumerate(mask.flatten().tolist()) if value]
        if len(positions) != features.shape[0]:
            raise ValueError(
                f"Found {len(positions)} placeholder tokens for {features.shape[0]} features"
            )
        if not positions:
            return inputs_embeds
        flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        flat[mx.array(positions, mx.uint32)] = features.astype(inputs_embeds.dtype)
        return flat.reshape(inputs_embeds.shape)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        inputs_embeds = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        ).inputs_embeds
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=inputs_embeds, mask=mask, **kwargs
        )

    def sanitize(self, weights):
        # re-prefixing a converted checkpoint yields language_model.language_model.*
        if any(k.startswith("language_model.") for k in weights):
            return weights

        vision, audio, speech, language = {}, {}, {}, {}
        for key, value in weights.items():
            if key.startswith("audio_encoder."):
                audio[key[len("audio_encoder.") :]] = value
            elif key.startswith("speech_embeddings."):
                speech[key] = value
            elif key.startswith("visual."):
                vision[key[len("visual.") :]] = value
            else:
                language[key] = value

        sanitized = {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(language).items()
        }
        if self.vision_tower is not None:
            sanitized.update(
                {
                    f"vision_tower.{k}": v
                    for k, v in self.vision_tower.sanitize(vision).items()
                }
            )
        sanitized.update({f"audio_encoder.{k}": v for k, v in audio.items()})
        sanitized.update(speech)
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        """Quantize each half of the checkpoint on the grid it came from.

        The routed experts ship as MXFP4 with block 32, so re-using that grid
        is bit-exact; re-quantizing them affine costs ~12% per tensor. The
        dense projections come from block-FP8 instead, where 4 bits cost ~13%
        and 8-bit affine costs ~0.8%.
        """

        def predicate(path, module):
            if "switch_mlp" in path:
                return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            return {"group_size": 64, "bits": 8, "mode": "affine"}

        return predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()
