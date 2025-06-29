import glob
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array

from ..base import expand2square
from .language import LanguageModel, TextConfig
from .processing_deepsek_vl_v2 import DeepseekVLV2Processor
from .vision import VisionConfig, VisionModel

AutoProcessor.register("deepseek_vl_v2", DeepseekVLV2Processor)


@dataclass
class ProjectorConfig:
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
    tile_tag: str = "2D"
    global_view_pos: str = "head"
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        if "language_config" in params:
            params["text_config"] = params["language_config"]
            del params["language_config"]

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class MlpProjector(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config
        if config.projector_config.projector_type == "identity":
            modules = nn.Identity()
        elif config.projector_config.projector_type == "linear":
            modules = nn.Linear(
                config.projector_config.input_dim, config.projector_config.n_embed
            )
        elif config.projector_config.projector_type == "mlp_gelu":
            mlp_depth = config.projector_config.depth
            modules = [
                nn.Linear(
                    config.projector_config.input_dim, config.projector_config.n_embed
                )
            ]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(
                        config.projector_config.n_embed, config.projector_config.n_embed
                    )
                )
        elif config.projector_config.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.projector_config.depth
            mlp_ratio = config.projector_config.mlp_ratio
            modules = [
                nn.Linear(
                    config.projector_config.input_dim
                    * config.projector_config.downsample_ratio
                    * config.projector_config.downsample_ratio,
                    config.projector_config.n_embed * mlp_ratio,
                )
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(
                    nn.Linear(
                        config.projector_config.n_embed * mlp_ratio,
                        config.projector_config.n_embed * mlp_ratio,
                    )
                )
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(
                    config.projector_config.n_embed * mlp_ratio,
                    config.projector_config.n_embed,
                )
            )
        else:
            raise ValueError(
                f"Unknown projector type: {config.projector_config.projector_type}"
            )

        if config.projector_config.token_pooling:
            self.token_pooling_layer = nn.Linear(
                config.projector_config.input_dim * 4, config.projector_config.input_dim
            )
        self.layers = modules

    def __call__(self, x):
        if self.config.projector_config.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(math.sqrt(wxh))
            x = mx.reshape(x, (batch_size, w, h, channels))
            x = mx.transpose(x, (0, 3, 1, 2))  # B, C, H, W

            # Implement unfold operation manually since MLX doesn't have unfold
            patches = []
            for i in range(0, h - 1, 2):
                for j in range(0, w - 1, 2):
                    patch = x[:, :, i : i + 2, j : j + 2]
                    patches.append(patch)

            patches = mx.stack(patches, axis=2)  # B, C, N_patches, 2, 2
            batch_size, channels, n_patches, _, _ = patches.shape

            # Reshape and concatenate
            patches = mx.reshape(patches, (batch_size, channels, n_patches, -1))
            patches = mx.transpose(patches, (0, 2, 1, 3))
            patches = mx.reshape(patches, (batch_size, n_patches, channels * 4))
            x = self.token_pooling_layer(patches)

        elif self.config.projector_config.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int(math.sqrt(hw))

            # Compute padding
            pad = (
                0
                if h % self.config.projector_config.downsample_ratio == 0
                else self.config.projector_config.downsample_ratio
                - h % self.config.projector_config.downsample_ratio
            )

            x = mx.reshape(x, (bs, h, w, input_dim))
            if pad > 0:
                x = mx.pad(x, [(0, 0), (0, pad), (0, pad), (0, 0)], constant_values=0)

            x = mx.transpose(x, (0, 3, 1, 2))  # B, C, H, W

            # Manual implementation of unfold for downsampling
            h_pad, w_pad = x.shape[2], x.shape[3]
            ds = self.config.projector_config.downsample_ratio
            patches = []

            for i in range(0, h_pad - ds + 1, ds):
                for j in range(0, w_pad - ds + 1, ds):
                    patch = x[:, :, i : i + ds, j : j + ds]
                    patches.append(mx.reshape(patch, (bs, -1)))

            x = mx.stack(patches, axis=1)  # B, N_patches, C*ds*ds

        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.projector = MlpProjector(config)
        self.vision_feature_layer = config.select_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
        embed_std = 1 / mx.sqrt(
            mx.array(config.projector_config.n_embed, dtype=mx.float32)
        )
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = mx.array(
                mx.random.normal((config.projector_config.n_embed,)) * embed_std
            )
            # fix the typo: view_seperater
            self.view_separator = mx.array(
                mx.random.normal((config.projector_config.n_embed,)) * embed_std
            )
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = mx.array(
                mx.random.normal(
                    (tile_variants_num + 1, config.projector_config.n_embed)
                )
                * embed_std
            )
        else:
            raise ValueError(
                f"tile tag should be either 1D or 2D, but got {self.tile_tag}"
            )

    def process_image_features(
        self,
        input_embeds,
        images_embeds,
        images_spatial_crop,
        images_seq_mask,
        h,
        w,
        n_dim,
    ):
        tile_index = 0
        all_batch_features = []

        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):
                # Extract global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = (num_width_tiles * num_height_tiles).tolist()

                # Get global features [hw, D]
                global_features = images_embeds[tile_index]

                # Get local features [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]

                tile_index += num_tiles_in_image + 1

                # Format global and local features
                if self.tile_tag == "2D":
                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = mx.reshape(global_features, (h, w, n_dim))

                    # [D] -> [h, 1, D]
                    new_lines_in_global = mx.expand_dims(self.image_newline, axis=0)
                    new_lines_in_global = mx.repeat(
                        new_lines_in_global, repeats=h, axis=0
                    )
                    new_lines_in_global = mx.expand_dims(new_lines_in_global, axis=1)

                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = mx.concatenate(
                        [global_features, new_lines_in_global], axis=1
                    )

                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = mx.reshape(global_features, (-1, n_dim))

                    # ----------------- local view add newline -----------------
                    # Rearrange local features
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = mx.reshape(
                        local_features, (num_height_tiles, num_width_tiles, h, w, n_dim)
                    )
                    local_features = mx.transpose(local_features, (0, 2, 1, 3, 4))
                    local_features = mx.reshape(
                        local_features,
                        (num_height_tiles * h, num_width_tiles * w, n_dim),
                    )

                    # Create newlines for local features
                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = mx.repeat(
                        mx.expand_dims(self.image_newline, axis=0),
                        repeats=num_height_tiles * h,
                        axis=0,
                    )
                    new_lines_in_local = mx.expand_dims(new_lines_in_local, axis=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = mx.concatenate(
                        [local_features, new_lines_in_local], axis=1
                    )

                    # [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = mx.reshape(local_features, (-1, n_dim))

                    # ----------------- merge global and local tiles -----------------
                    view_separator = mx.expand_dims(self.view_separator, axis=0)

                    if self.global_view_pos == "head":
                        global_local_features = mx.concatenate(
                            [global_features, view_separator, local_features], axis=0
                        )
                    else:
                        global_local_features = mx.concatenate(
                            [local_features, view_separator, global_features], axis=0
                        )

                else:
                    # 1D processing (legacy path)
                    global_features = mx.concatenate(
                        [
                            mx.expand_dims(self.tile_indicators[0], axis=0),
                            global_features,
                        ],
                        axis=0,
                    )

                    local_indicators = mx.expand_dims(
                        self.tile_indicators[1 : num_tiles_in_image + 1], axis=1
                    )
                    local_features = mx.concatenate(
                        [local_indicators, local_features], axis=1
                    )
                    local_features = mx.reshape(local_features, (-1, n_dim))

                    if self.global_view_pos == "head":
                        global_local_features = mx.concatenate(
                            [global_features, local_features], axis=0
                        )
                    else:
                        global_local_features = mx.concatenate(
                            [local_features, global_features], axis=0
                        )

                images_in_this_batch.append(global_local_features)

            if images_in_this_batch:
                images_in_this_batch = mx.concatenate(images_in_this_batch, axis=0)
                # Find positions where images should be placed
                image_indices = np.where(images_seq_mask[idx])[0].tolist()
                # Directly assign the image features to those positions
                input_embeds[idx, image_indices] = images_in_this_batch

        return input_embeds

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        images_spatial_crop: Optional[mx.array] = None,
        image_seq_mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.language_model.model.embed_tokens(input_ids)

        bs = pixel_values.shape[0]
        max_n_images = pixel_values.shape[1]

        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []

        # Total number of tiles in each batch
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx][jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (
                    1 + num_width_tiles * num_height_tiles
                ).tolist()

            total_tiles.append(pixel_values[idx, : batch_num_tiles[idx]])

        total_tiles = mx.concatenate(total_tiles, axis=0)

        if total_tiles.shape[0] == 0:
            return self.language_model.model.embed_tokens(input_ids)

        # Get the input embeddings from the language model
        input_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get the ouptut hidden states from the vision model
        hidden_states, *_ = self.vision(
            total_tiles.transpose(0, 2, 3, 1), output_hidden_states=True
        )

        # Pass image features through the multi-modal projector
        image_features = self.projector(hidden_states)

        _, hw, n_dim = image_features.shape
        h = w = int(hw**0.5)

        image_features = self.process_image_features(
            input_embeds,
            image_features,
            images_spatial_crop,
            image_seq_mask,
            h,
            w,
            n_dim,
        )

        return image_features

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):

        images_spatial_crop = kwargs.get("images_spatial_crop", None)
        images_seq_mask = kwargs.get("images_seq_mask", None)
        input_embeddings = self.get_input_embeddings(
            input_ids, pixel_values, images_spatial_crop, images_seq_mask
        )
        logits = self.language_model(
            input_ids, cache=cache, inputs_embeds=input_embeddings
        )
        return logits

    @staticmethod
    def sanitize(weights):
        def transform_key(key):
            if "language" in key and "language_model" not in key:
                if ".model" in key:
                    key = key.replace("language.model", "language_model.model")
                if ".lm_head" in key:
                    key = key.replace("language", "language_model")
            if "vision" in key and "vision_tower" not in key:
                key = key.replace("vision", "vision.vision_tower")
            if "view_seperator" in key:
                key = key.replace("view_seperator", "view_separator")
            return key

        return {transform_key(k): v for k, v in weights.items()}
