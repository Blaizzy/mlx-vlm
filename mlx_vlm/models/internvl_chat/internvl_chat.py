import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from ..base import pixel_shuffle
from ..qwen2_vl.language import LanguageModel
from .vision import VisionConfig, VisionModel


@dataclass
class TextConfig:
    model_type: str
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    use_sliding_window: bool
    max_window_layers: int
    num_key_value_heads: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict
    rope_traditional: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    sliding_window: Optional[int] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 151667
    video_token_index: int = 151656
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -1
    vocab_size: int = 32000
    downsample_ratio: float = 0.5
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

        self.downsample_ratio = config.downsample_ratio

        vit_hidden_size = self.config.vision_config.hidden_size
        llm_hidden_size = self.config.text_config.hidden_size

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
    ):

        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # TODO: Remove this after transformers implementation is merged
        if pixel_values.ndim == 5:
            pixel_values = pixel_values[0]

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, _, _ = self.vision_model(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        # Extract vision embeddings, removing the class token (first token)
        hidden_states = hidden_states[:, 1:, :]

        # Apply pixel shuffle with downsampling
        hidden_states = pixel_shuffle(
            hidden_states, shuffle_ratio=self.downsample_ratio
        )

        # Apply MLP transformation
        for layer in self.mlp1:
            hidden_states = layer(hidden_states)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            hidden_states, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        B, N, C = inputs_embeds.shape
        image_token_index = self.config.image_token_index
        video_token_index = self.config.video_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = input_ids == image_token_index
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_index

        image_indices = np.where(image_positions)[1].tolist()

        image_features = image_features.reshape(-1, image_features.shape[-1])

        inputs_embeds[:, image_indices, :] = image_features

        return inputs_embeds.reshape(B, N, C)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(None, cache=cache, inputs_embeds=input_embddings)
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
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = VisionModel.sanitize(weights)
        weights = LanguageModel.sanitize(weights)

        model.load_weights(list(weights.items()))
        return model
