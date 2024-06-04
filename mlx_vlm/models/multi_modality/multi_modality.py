import glob
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array

from ..base import expand2square
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class AlignerConfig:
    cls: str
    model_type: str
    params: dict


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    select_layer: int = -1
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class ImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 384,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])

    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        width, height = pil_img.size
        max_size = max(width, height)

        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        pil_img = self.resize_image(
            pil_img,
            size,
            interpolation=Image.BICUBIC,
            antialias=True,
        )

        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def resize_image(
        self, pil_img, size, interpolation=Image.BILINEAR, antialias=False
    ):
        """Resize the input PIL Image to the given size."""
        return pil_img.resize(size, resample=interpolation)

    def preprocess(self, images, return_tensors: str = "np", **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        images: List[np.ndarray] = [self.resize(image) for image in images]

        # resacle from [0, 255] -> [0, 1]
        images = [
            self.rescale(
                image=image,
                scale=self.rescale_factor,
                input_data_format="channels_first",
            )
            for image in images
        ]

        # normalize
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                )
                for image in images
            ]

        return images

    @property
    def default_shape(self):
        return [self.image_size, self.image_size, 3]


class MlpProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.aligner_config["params"]["projector_type"] == "mlp_gelu":
            self.layers = [
                nn.Linear(
                    config.vision_config.hidden_size,
                    config.text_config.hidden_size,
                    bias=True,
                )
            ]
            mlp_depth = config.aligner_config["params"]["depth"]
            for _ in range(1, mlp_depth):
                self.layers.append(nn.GELU())
                self.layers.append(
                    nn.Linear(
                        config.text_config.hidden_size,
                        config.text_config.hidden_size,
                        bias=True,
                    )
                )

        else:
            projector_type = config.aligner_config["params"]["projector_type"]
            raise ValueError(f"Unknown projector type: {projector_type}")

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.aligner = MlpProjector(config)
        self.vision_feature_layer = config.select_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, _, _ = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        # Select the hidden states from the desired layer
        selected_image_feature = hidden_states[None, :]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                "Unexpected feature selection strategy: "
                f"{self.vision_feature_select_strategy}"
            )

        # Pass image features through the multi-modal projector
        image_features = self.aligner(selected_image_feature)

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

        if len(image_positions) != num_images:
            raise ValueError(
                f"The number of image tokens ({len(image_positions)}) does not "
                f" match the number of image inputs ({num_images})."
            )

        text_segments = []
        start_idx = 0

        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = mx.split(image_features, image_features.shape[0])
        final_embeddings = [v for p in zip(text_segments, image_embeddings) for v in p]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        # Create a final embedding of shape
        # (1, num_image_patches*num_images + sequence_len, embed_dim)
        return mx.concatenate(final_embeddings, axis=1)

    def __call__(
        self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None
    ):
        input_embddings = self.get_input_embeddings(input_ids, pixel_values)
        logits, cache = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embddings
        )
        return logits, cache

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
        weights = {
            (
                f"vision_tower.vision_model.{re.sub(r'^vision_model.vision_tower.', '', k)}"
                if re.match(r"^vision_model.vision_tower.", k)
                else k
            ): v
            for k, v in weights.items()
        }

        return weights
