import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    vocab_size: int = 128259
    scale_factor: int = 2
    image_token_id: int = 49153
    image_token_index: Optional[int] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def __call__(self, x):
        return self.proj(x)


class Idefics3Connector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = MLP(config)

    def pixel_shuffle(self, x, scale_factor=2):
        bsz, seq, embed_dim = x.shape
        height = width = int(seq**0.5)
        x = x.reshape(bsz, height, width, embed_dim)
        x = x.reshape(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def __call__(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.connector = Idefics3Connector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_attention_mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.embed_tokens(input_ids)

        inputs_embeds = self.language_model.embed_tokens(input_ids)

        pooler_output, embeddings, hidden_state = self.vision_model(
            pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
        )

        image_features = pooler_output.astype(pixel_values.dtype)
        image_features = self.connector(image_features)

        final_inputs_embeds = self._prepare_inputs_for_multimodal(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _prepare_inputs_for_multimodal(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids == image_token_index)[1].tolist()

        num_images, _, vision_hidden_size = image_features.shape

        reshaped_image_hidden_states = image_features.reshape(-1, vision_hidden_size)

        # cast to the dtype of the input_embeds to support quantized models
        reshaped_image_hidden_states = reshaped_image_hidden_states.astype(
            inputs_embeds.dtype
        )
        inputs_embeds[:, image_positions, :] = reshaped_image_hidden_states

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            inputs=input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits

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

        text_config = AutoConfig.from_pretrained(config["text_config"]["model_type"])
        text_config = text_config.to_dict()
        config["text_config"] = text_config
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
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        model.load_weights(list(weights.items()))
        return model

    def sanitize(self, weights):
        weights = {
            (
                f"{k.split('.', 1)[1]}"
                if re.match(r"^model\.", k)
                else (f"language_model.{k}" if re.match(r"^lm_head\.", k) else k)
            ): v
            for k, v in weights.items()
        }

        weights = {
            (
                f"language_model.{k.split('.', 1)[1]}"
                if re.match(
                    r"^text_model\.",
                    k,
                )
                else k
            ): v
            for k, v in weights.items()
        }

        return weights
