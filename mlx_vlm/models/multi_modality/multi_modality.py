import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import to_numpy_array

from ..base import BaseImageProcessor, expand2square
from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ProjectorConfig:
    cls: str
    model_type: str
    params: dict

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
    projector_config: ProjectorConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 100015
    vision_feature_select_strategy: str = "default"
    select_layer: int = -1
    pad_id: int = 100001
    num_image_tokens: int = 576
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        if "aligner_config" in params:
            params["projector_config"] = params["aligner_config"]
            del params["aligner_config"]

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
        config,
        image_size: int = 384,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.5,
            0.5,
            0.5,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.5,
            0.5,
            0.5,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "high_res_cfg" in config["vision_config"]["params"]:
            self.image_size = config["vision_config"]["params"]["high_res_cfg"][
                "image_size"
            ]
            self.image_mean = config["vision_config"]["params"]["high_res_cfg"][
                "pixel_mean"
            ]
            self.image_std = config["vision_config"]["params"]["high_res_cfg"][
                "pixel_std"
            ]
            self.do_normalize = False
        else:
            self.image_size = image_size
            self.image_mean = image_mean
            self.image_std = image_std
            self.do_normalize = do_normalize

        self.rescale_factor = rescale_factor
        self.min_size = min_size

        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in self.image_mean])

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

        pil_img = pil_img.resize(size=tuple(size[::-1]), resample=Image.BICUBIC)

        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    def preprocess(self, images, **kwargs) -> BatchFeature:
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


class MlpProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        if config.projector_config.params["projector_type"] == "mlp_gelu":
            self.layers = [
                nn.Linear(
                    config.vision_config.hidden_size,
                    config.text_config.hidden_size,
                    bias=True,
                )
            ]
            mlp_depth = config.projector_config.params["depth"]
            for _ in range(1, mlp_depth):
                self.layers.append(nn.GELU())
                self.layers.append(
                    nn.Linear(
                        config.text_config.hidden_size,
                        config.text_config.hidden_size,
                        bias=True,
                    )
                )
        elif (
            config.projector_config.params["projector_type"]
            == "low_high_hybrid_split_mlp_gelu"
        ):
            mlp_depth = config.projector_config.params["depth"]
            self.high_up_proj = nn.Linear(
                config.vision_config.hidden_size, config.text_config.hidden_size // 2
            )
            self.low_up_proj = nn.Linear(
                config.vision_config.hidden_size, config.text_config.hidden_size // 2
            )

            self.layers = []
            for _ in range(1, mlp_depth):
                self.layers.append(nn.GELU())
                self.layers.append(
                    nn.Linear(
                        config.text_config.hidden_size, config.text_config.hidden_size
                    )
                )

        else:
            projector_type = config.projector_config.params["projector_type"]
            raise ValueError(f"Unknown projector type: {projector_type}")

    def __call__(self, x: Union[mx.array, Tuple]) -> mx.array:

        if isinstance(x, tuple):
            high_x, low_x = x

            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)

            B, D = high_x.shape[0], high_x.shape[-1]
            high_x = high_x.reshape(B, -1, D)

            x = mx.concatenate([high_x, low_x], axis=-1)

        for layer in self.layers:
            x = layer(x)

        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.aligner = MlpProjector(config)
        self.vision_feature_layer = config.select_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def add_image_token(
        self,
        image_indices: list,
        input_ids: np.ndarray,
        image_token_index: int,
        num_image_tokens: int,
        add_special_token: bool = False,
    ):
        """
        Inserts image tokens into an array of input IDs at specified indices.

        Args:
            image_indices (List[int]): Indices where image tokens should be inserted.
            input_ids (np.ndarray): Original array of input IDs, expected to be two-dimensional.
            image_token_index (int): The ID used to represent an image token.
            num_image_tokens (int): Number of image tokens to insert at each index.
            add_special_token (bool): If True, adjusts the indices to include a special token.

        Returns:
            Tuple of (np.ndarray, np.ndarray):
                - Updated array of input IDs with image tokens inserted.
                - Array indicating the number of image tokens added at each position.
        """
        input_slices = []

        start = 0
        flat_input_ids = input_ids.flatten()

        for index in image_indices:
            end = (index[0] + 1) if add_special_token else index[0]

            input_slices.append(flat_input_ids[start:end])
            input_slices.append(
                np.full((num_image_tokens,), image_token_index, dtype=np.int64)
            )
            start = index[0] + 1  # Move start past the current image insertion point

        input_slices.append(flat_input_ids[start:])

        input_ids = np.concatenate(input_slices, axis=0)
        num_image_tokens_array = np.array(
            [num_image_tokens] * len(image_indices), dtype=np.int64
        )
        input_ids = input_ids.reshape(1, -1)

        return input_ids, num_image_tokens_array

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        image_token_index = self.config.image_token_index
        num_image_tokens = self.config.num_image_tokens

        image_token_mask = np.array(input_ids[0] == image_token_index).astype(bool)
        image_indices = np.nonzero(image_token_mask)

        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=np.array(input_ids),
            image_token_index=image_token_index,
            num_image_tokens=num_image_tokens,
        )

        input_ids = mx.array(input_ids)

        # Get the input embeddings from the language model
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        if self.config.vision_config.cls == "HybridVisionTower":
            hidden_states = self.vision_model(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
            )
        else:
            hidden_states, _, _ = self.vision_model(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
            )

        # Pass image features through the multi-modal projector
        image_features = self.aligner(hidden_states)

        # Insert special image tokens in the input_ids
        final_inputs_embeds = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids
        )
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        image_token_index = self.config.image_token_index

        # Positions of <image> tokens in input_ids, assuming batch size is 1
        image_positions = np.where(input_ids[0] == image_token_index)[0].tolist()
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
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):

        input_embeddings = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embeddings
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
        model_config.projector_config = ProjectorConfig.from_dict(
            model_config.projector_config
        )
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
