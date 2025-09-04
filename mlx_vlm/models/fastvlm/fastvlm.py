import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import coremltools
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download

from .language import LanguageModel, TextConfig


@dataclass
class VisionConfig:
    mm_hidden_size: int
    mm_vision_tower: str

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
    eos_token_id: int
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 151936

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
        params["text_config"] = dict(filter(lambda x: "mm" not in x[0], params.items()))
        # Copy vision config parameters from root level
        params["vision_config"] = dict(filter(lambda x: "mm" in x[0], params.items()))

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class FastVLMMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_0 = nn.Linear(
            config.vision_config.mm_hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_0(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = None
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = FastVLMMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get image features from CoreML model
        coreml_out_dict = self.vision_tower.predict(
            {"images": np.array(pixel_values, copy=False)}
        )

        # Pass image features through the multi-modal projector
        image_features = self.multi_modal_projector(
            mx.array(coreml_out_dict["image_features"])
        )

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
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()
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
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
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
            model_config = json.load(f)

        model_config = ModelConfig.from_dict(model_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)

        model = Model(model_config)
        weight_files = glob.glob(str(path / "*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = LanguageModel.sanitize(weights)

        # Load CoreML vision tower
        coreml_file = glob.glob(str(path / "*.mlpackage"))
        assert len(coreml_file) == 1, "Found multiple vision model files"
        model.vision_tower = coremltools.models.MLModel(coreml_file[0])

        model.load_weights(list(weights.items()))
        return model
