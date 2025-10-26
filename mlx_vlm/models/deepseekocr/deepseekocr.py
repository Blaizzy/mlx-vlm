import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoProcessor

from .config import ModelConfig, ProjectorConfig, SAMViTConfig
from .language import LanguageModel
from .processing_deepseekocr import DeepseekVLV2Processor
from .sam import SAMEncoder
from .vision import VisionModel

AutoProcessor.register("deepseekocr", DeepseekVLV2Processor)


class MlpProjector(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.config = config

        if config.projector_config.projector_type == "linear":
            modules = nn.Linear(
                config.projector_config.input_dim, config.projector_config.n_embed
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

        self.layers = modules

    def __call__(self, x):
        if self.config.projector_config.projector_type == "downsample_mlp_gelu":
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

        if self.config.projector_config.projector_type == "linear":
            x = self.layers(x)
        else:
            for layer in self.layers:
                x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_model = VisionModel(config.vision_config)
        sam_config = SAMViTConfig()
        self.sam_model = SAMEncoder(
            img_size=sam_config.image_size,
            patch_size=sam_config.patch_size,
            embed_dim=sam_config.width,
            depth=sam_config.layers,
            num_heads=sam_config.heads,
            window_size=sam_config.window_size,
            global_attn_indexes=sam_config.global_attn_indexes,
        )
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
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        images_spatial_crop: Optional[mx.array] = None,
        images_seq_mask: Optional[mx.array] = None,
    ):
        input_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return input_embeds

        if (
            self.sam_model is not None
            and (input_ids.shape[1] != 1 or self.training)
            and mx.sum(pixel_values[0][1]).item() != 0
        ):

            idx = 0

            for crop_shape in images_spatial_crop.tolist():
                images_in_this_batch = []
                patches = pixel_values[0]
                image_ori = pixel_values[1]
                if mx.sum(patches).item() != 0:
                    crop_flag = 1
                    local_features_1 = self.sam_model(patches.transpose(0, 2, 3, 1))

                    local_features_2 = self.vision_model(
                        patches.transpose(0, 2, 3, 1), patch_embeds=local_features_1
                    )

                    local_features = mx.concatenate(
                        (
                            local_features_2[:, 1:],
                            local_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )

                    local_features = self.projector(local_features)

                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )

                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)

                    print("=====================")
                    print("BASE: ", global_features.shape)
                    print("PATCHES: ", local_features.shape)
                    print("=====================")

                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    _2, hw2, n_dim2 = local_features.shape
                    h2 = w2 = int(hw2**0.5)

                    width_crop_num, height_crop_num = (
                        crop_shape[0],
                        crop_shape[1],
                    )

                    global_features = global_features.reshape(h, w, n_dim)

                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :], (h, 1, n_dim)
                            ),
                        ],
                        axis=1,
                    )

                    global_features = global_features.reshape(-1, n_dim)

                    local_features = (
                        local_features.reshape(
                            height_crop_num, width_crop_num, h2, w2, n_dim2
                        )
                        .transpose(0, 2, 1, 3, 4)
                        .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
                    )
                    local_features = mx.concatenate(
                        [
                            local_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :],
                                (height_crop_num * h2, 1, n_dim2),
                            ),
                        ],
                        axis=1,
                    )
                    local_features = local_features.reshape(-1, n_dim2)

                    global_local_features = mx.concatenate(
                        [local_features, global_features, self.view_separator[None, :]],
                        axis=0,
                    )

                else:
                    global_features_1 = self.sam_model(image_ori.transpose(0, 2, 3, 1))
                    global_features_2 = self.vision_model(
                        image_ori.transpose(0, 2, 3, 1), global_features_1
                    )
                    global_features = mx.concatenate(
                        (
                            global_features_2[:, 1:],
                            global_features_1.flatten(start_axis=1, end_axis=2),
                        ),
                        axis=-1,
                    )
                    global_features = self.projector(global_features)
                    print("=====================")
                    print("BASE: ", global_features.shape)
                    print("NO PATCHES")
                    print("=====================")
                    _, hw, n_dim = global_features.shape
                    h = w = int(hw**0.5)

                    global_features = global_features.reshape(h, w, n_dim)

                    global_features = mx.concatenate(
                        [
                            global_features,
                            mx.broadcast_to(
                                self.image_newline[None, None, :], (h, 1, n_dim)
                            ),
                        ],
                        axis=1,
                    )

                    global_features = global_features.reshape(-1, n_dim)

                    global_local_features = mx.concatenate(
                        [global_features, self.view_separator[None, :]], axis=0
                    )

                images_in_this_batch.append(global_local_features)

                if images_in_this_batch:
                    images_in_this_batch = mx.concatenate(images_in_this_batch, axis=0)
                    # Find positions where images should be placed
                    image_indices = np.where(images_seq_mask[idx])[0].tolist()
                    # Directly assign the image features to those positions
                    input_embeds[idx, image_indices] = images_in_this_batch

                idx += 1

        return input_embeds

    @property
    def layers(self):
        return self.language_model.model.layers

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
            if "model.layers" in key and "language_model" not in key:
                key = key.replace("model.layers", "language_model.model.layers")

            if "model.embed_tokens" in key and "language_model" not in key:
                key = key.replace(
                    "model.embed_tokens", "language_model.model.embed_tokens"
                )

            if "model.norm" in key and "language_model" not in key:
                key = key.replace("model.norm", "language_model.model.norm")

            if "model.vision_model" in key:
                key = key.replace("model.vision_model", "vision_model")

            if "model.sam_model" in key:
                key = key.replace("model.sam_model", "sam_model")

            if "model.projector" in key:
                key = key.replace("model.projector", "projector")

            if "model.view_seperator" in key:
                key = key.replace("model.view_seperator", "view_separator")

            if "model.image_newline" in key:
                key = key.replace("model.image_newline", "image_newline")

            if "lm_head.weight" in key and "language_model" not in key:
                key = key.replace("lm_head.weight", "language_model.lm_head.weight")

            return key

        return {transform_key(k): v for k, v in weights.items()}
