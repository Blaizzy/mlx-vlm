import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .audio import AudioConfig, AudioModel, Gemma3nAudioEmbedder
from .language import LanguageModel, TextConfig
from .vision import Gemma3p5VisionEmbedder, VisionConfig, VisionModel
from .config import ModelConfig



class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config


        # Text
        self.language_model = LanguageModel(config.text_config)

        # Vision
        # self.vision_tower = VisionModel(config.vision_config)
        # self.embed_vision = Gemma3p5VisionEmbedder(config.vision_config)

        # # Audio
        self.audio_tower = AudioModel(config.audio_config)
        self.embed_audio = Gemma3nAudioEmbedder(config)

    def embed(self, input_ids):
        text_input_ids = mx.where(input_ids < self.config.vocab_size, input_ids, 0)
        inputs_embeds = self.language_model.model.embed_tokens(text_input_ids)

        # vision_embeds = self.embed_vision(input_ids)
        # inputs_embeds = mx.where(
        #     input_ids[..., None] < self.embed_vision.vocab_offset,
        #     inputs_embeds,
        #     vision_embeds,
        # )

        audio_embeds = self.embed_audio(input_ids)
        inputs_embeds = mx.where(
            input_ids[..., None] < self.embed_audio.vocab_offset,
            inputs_embeds,
            audio_embeds,
        )
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None and input_features is None:
            return self.embed(input_ids)

        inputs_embeds = self.embed(input_ids)

        # if pixel_values is not None:
        #     image_features = self.get_image_features(pixel_values)
        #     return self.merge_multimodal_and_text(
        #         input_ids, inputs_embeds, image_features, self.config.image_token_id
        #     )

        if input_features is not None:
            audio_outputs = self.get_audio_features(input_features, input_features_mask)
            return self.merge_multimodal_and_text(
                input_ids, inputs_embeds, audio_outputs, self.config.audio_token_id
            )

    def get_audio_features(self, input_features, input_features_mask):
        audio_outputs, _ = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(audio_outputs, is_soft_embedding=True)

    def get_image_features(self, pixel_values):
        vision_outputs, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1),
            output_hidden_states=True,
        )
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).transpose(0, 2, 1)

        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(vision_outputs, is_soft_embedding=True)

    def merge_multimodal_and_text(self, input_ids, inputs_embeds, features, token_id):
        if input_ids is None:
            special_image_mask = (
                inputs_embeds
                == self.language_model.model.embed_tokens(
                    mx.array([token_id])
                )
            )
        else:
            special_image_mask = mx.expand_dims(input_ids == token_id, -1)
            special_image_mask = mx.broadcast_to(
                special_image_mask, inputs_embeds.shape
            )

        # Count special tokens by summing the mask
        special_token_count = mx.sum(special_image_mask[..., 0] if special_image_mask.ndim > 2 else special_image_mask)
        expected_feature_count = features.shape[0] * features.shape[1] if features.ndim > 1 else features.shape[0]

        if special_token_count != expected_feature_count:
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {special_token_count} (image/audio) tokens in the text and "
                f"{expected_feature_count} tokens from (image/audio) embeddings."
            )
        features = features.astype(inputs_embeds.dtype)
        inputs_embeds = mx.where(special_image_mask, features.flatten(), inputs_embeds)
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):

        # Audio features
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        input_embeddings = self.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, input_features=input_features, input_features_mask=input_features_mask, **kwargs
        )

        logits = self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=input_embeddings,
        )
        return logits

    def sanitize(self, weights):
        sanitized_weights = {".".join(k.split(".")[1:]): v for k, v in weights.items() if "vision_tower" not in k and "embed_vision" not in k}
        return sanitized_weights

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        with open(path / "config.json", "r") as f:
            config = json.load(f)

        model_config = ModelConfig.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights=weights)

        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model
