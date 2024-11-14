import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ..base import KVCache
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    image_feature_dropout: float = 0.0
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    image_pooling_2d: str = "attention"
    image_projector: str = "mlp"

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
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

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

        if pixel_values is not None:
            assert (
                mask is not None and image_input_idx is not None
            ), "mask and image_input_idx must be provided when images are given"

            # Process images
            image_features, cls_embed = self.vision_tower(pixel_values, mask)

            # Insert image features into the input embeddings
            input_embeddings = self.language_model.model.wte(input_ids)
            num_image, num_patch = image_features.shape[1:3]
            image_features = image_features.reshape(
                batch_size, num_image * num_patch, -1
            )
            image_input_idx = image_input_idx.reshape(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = (
                mx.arange(batch_size)
                .reshape(-1, 1)
                .repeat(image_features.shape[1], axis=1)
            )
            input_embeddings = mx.where(
                (batch_idx[None, :] == mx.arange(batch_size)[:, None, None])
                & (image_input_idx[None, :] == mx.arange(seq_len)[:, None, None]),
                image_features,
                input_embeddings,
            )
        else:
            input_embeddings = None

        # Forward pass through the language model
        logits = self.language_model(
            input_ids,
            mask=None,
            cache=cache,
            inputs_embeds=input_embeddings,
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

        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)

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

    def sanitize(self, weights):
        def transform_key(key):
            if "model.transformer" in key:
                key = key.replace("model.transformer", "language_model.model")
            if "model.vision_backbone" in key:
                key = key.replace("model.vision_backbone", "vision_tower")
            return key

        return {transform_key(k): v for k, v in weights.items()}