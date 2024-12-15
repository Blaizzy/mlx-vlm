import glob
import inspect
import math
import json
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
class ProjectorConfig:
    model_type: str
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

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
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        if config.projector_config.projector_type == "identity":
            modules = nn.Identity()
        elif config.projector_config.projector_type == "linear":
            modules = nn.Linear(config.projector_config.input_dim, config.projector_config.n_embed)
        elif config.projector_config.projector_type == "mlp_gelu":
            mlp_depth = config.projector_config.depth
            modules = [nn.Linear(config.projector_config.input_dim, config.projector_config.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.projector_config.n_embed, config.projector_config.n_embed))
            modules = nn.Sequential(*modules)
        elif config.projector_config.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.projector_config.depth
            mlp_ratio = config.projector_config.mlp_ratio
            modules = [nn.Linear(config.projector_config.input_dim * config.projector_config.downsample_ratio * config.projector_config.downsample_ratio,
                               config.projector_config.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.projector_config.n_embed * mlp_ratio, config.projector_config.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.projector_config.n_embed * mlp_ratio, config.projector_config.n_embed))
            modules = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_config.projector_type}")

        if config.projector_config.token_pooling:
            self.token_pooling_layer = nn.Linear(config.projector_config.input_dim * 4, config.projector_config.input_dim)
        self.layers = modules

    def __call__(self, x):
        if self.config.projector_config.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(math.sqrt(wxh))
            x = mx.reshape(x, (batch_size, w, h, channels))
            x = mx.transpose(x, (0, 3, 1, 2))  # B, C, H, W

            # Implement unfold operation manually since MLX doesn't have unfold
            patches = []
            for i in range(0, h-1, 2):
                for j in range(0, w-1, 2):
                    patch = x[:, :, i:i+2, j:j+2]
                    patches.append(patch)

            patches = mx.stack(patches, axis=2)  # B, C, N_patches, 2, 2
            batch_size, channels, n_patches, _, _ = patches.shape

            # Reshape and concatenate
            patches = mx.reshape(patches, (batch_size, channels, n_patches, -1))
            patches = mx.transpose(patches, (0, 2, 1, 3))
            patches = mx.reshape(patches, (batch_size, n_patches, channels * 4))
            x = self.token_pooling_layer(patches)

        elif self.config.projector_config.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int(math.sqrt(hw))

            # Compute padding
            pad = 0 if h % self.config.projector_config.downsample_ratio == 0 else \
                  self.config.projector_config.downsample_ratio - h % self.config.projector_config.downsample_ratio

            x = mx.reshape(x, (bs, h, w, input_dim))
            if pad > 0:
                x = mx.pad(x, [(0, 0), (0, pad), (0, pad), (0, 0)], constant_values=0)

            x = mx.transpose(x, (0, 3, 1, 2))  # B, C, H, W

            # Manual implementation of unfold for downsampling
            h_pad, w_pad = x.shape[2], x.shape[3]
            ds = self.config.projector_config.downsample_ratio
            patches = []

            for i in range(0, h_pad-ds+1, ds):
                for j in range(0, w_pad-ds+1, ds):
                    patch = x[:, :, i:i+ds, j:j+ds]
                    patches.append(mx.reshape(patch, (bs, -1)))

            x = mx.stack(patches, axis=1)  # B, N_patches, C*ds*ds

        return self.layers(x)

class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
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
            return self.language_model(input_ids)

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
            hidden_states = self.vision_tower(
                pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
            )
        else:
            hidden_states, _, _ = self.vision_tower(
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
        model_config.aligner_config = AlignerConfig.from_dict(
            model_config.aligner_config
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

    @staticmethod
    def sanitize(weights):
        def transform_key(key):
            if "language.model" in key:
                key = key.replace("language.model", "language_model.model")
            if "vision" in key:
                key = key.replace("vision", "vision.vision_tower")
            return key

        return {transform_key(k): v for k, v in weights.items()}
