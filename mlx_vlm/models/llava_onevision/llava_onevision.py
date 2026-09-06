import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..interpolate import resize_bilinear
from . import processing_llava_onevision  # noqa: F401
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """Return the (rows, cols) tile grid AnyRes picked for ``image_size``."""
    from transformers.image_processing_utils import select_best_resolution

    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size):
    """Number of tiles AnyRes produces for ``image_size``, including the base tile."""
    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
        image_size, grid_pinpoints, patch_size
    )
    return num_patch_height * num_patch_width + 1


def unpad_image(tensor, original_size):
    """Strip the AnyRes padding from a (C, H, W) feature map."""
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1], tensor.shape[2]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        return tensor[:, padding : current_height - padding, :]

    scale_factor = current_height / original_height
    new_width = int(original_width * scale_factor)
    padding = (current_width - new_width) // 2
    return tensor[:, :, padding : current_width - padding]


class LlavaOnevisionMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(self.gelu(self.linear_1(x)))


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)

        self.image_newline = None
        if config.use_image_newline_parameter:
            embed_std = 1 / math.sqrt(config.text_config.hidden_size)
            self.image_newline = (
                mx.random.normal((config.text_config.hidden_size,)) * embed_std
            )

    @property
    def max_num_patches(self) -> int:
        return int(self.config.vision_aspect_ratio.strip("anyres_max_"))

    @property
    def patches_per_side(self) -> int:
        return (
            self.config.vision_config.image_size // self.config.vision_config.patch_size
        )

    def _select_features(self, hidden_states) -> mx.array:
        layer = self.config.vision_feature_layer
        if isinstance(layer, int):
            selected = hidden_states[layer]
        else:
            selected = mx.concatenate([hidden_states[i] for i in layer], axis=-1)
        if self.config.vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        return selected

    def pack_image_features(
        self, image_features: List[mx.array], image_sizes
    ) -> List[mx.array]:
        """Unpad, optionally downsample, and append newlines to each image's tiles."""
        packed = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                tiles = image_feature[1:]

                height = width = self.patches_per_side
                if height * width != base_image_feature.shape[0]:
                    raise ValueError(
                        "The number of patches is not consistent with the image size."
                    )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                tiles = tiles.reshape(
                    num_patch_height, num_patch_width, height, width, -1
                )
                tiles = mx.transpose(tiles, axes=(4, 0, 2, 1, 3))
                channels = tiles.shape[0]
                tiles = tiles.reshape(
                    channels, num_patch_height * height, num_patch_width * width
                )
                tiles = unpad_image(tiles, image_sizes[image_idx])

                curr_height, curr_width = tiles.shape[1], tiles.shape[2]
                ratio = math.sqrt(
                    curr_height * curr_width / (self.max_num_patches * height**2)
                )
                if ratio > 1.1:
                    tiles = mx.transpose(tiles, axes=(1, 2, 0))
                    tiles = resize_bilinear(
                        tiles,
                        (int(curr_height // ratio), int(curr_width // ratio)),
                        antialias=False,
                    )
                    tiles = mx.transpose(tiles, axes=(2, 0, 1))

                if self.image_newline is not None:
                    channels, curr_height, _ = tiles.shape
                    newline = mx.broadcast_to(
                        self.image_newline[:, None, None].astype(tiles.dtype),
                        (channels, curr_height, 1),
                    )
                    tiles = mx.concatenate([tiles, newline], axis=-1)

                channels = tiles.shape[0]
                tiles = tiles.reshape(channels, -1).T
                image_feature = mx.concatenate([base_image_feature, tiles], axis=0)
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = mx.concatenate(
                        [
                            image_feature,
                            self.image_newline[None].astype(image_feature.dtype),
                        ],
                        axis=0,
                    )
            packed.append(image_feature)
        return packed

    def get_image_features(self, pixel_values: mx.array, image_sizes) -> mx.array:
        """Tower + projector + AnyRes packing, flattened over all images."""
        if image_sizes is None:
            raise ValueError(
                "llava_onevision needs image_sizes to unpad AnyRes image features"
            )
        image_sizes = np.array(image_sizes).reshape(-1, 2).tolist()

        if pixel_values.ndim == 5:
            # The processor pads the tile axis to the batch maximum; drop the padding.
            num_patches = [
                image_size_to_num_patches(
                    size,
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                for size in image_sizes
            ]
            flat_pixel_values = mx.concatenate(
                [pixel_values[i, :count] for i, count in enumerate(num_patches)],
                axis=0,
            )
        elif pixel_values.ndim == 4:
            flat_pixel_values = pixel_values
            num_patches = [pixel_values.shape[0]]
        else:
            raise ValueError(
                f"pixel_values of shape {pixel_values.shape}, expect 4 or 5 dimensions"
            )

        *_, hidden_states = self.vision_tower(
            flat_pixel_values.transpose(0, 2, 3, 1), output_hidden_states=True
        )
        selected = self._select_features(hidden_states)
        features = self.multi_modal_projector(selected)

        split, start = [], 0
        for count in num_patches:
            split.append(features[start : start + count])
            start += count

        packed = self.pack_image_features(split, image_sizes)
        return mx.concatenate(packed, axis=0)

    def apply_pooling(self, features: mx.array) -> mx.array:
        """Bilinearly pool each frame's patch grid by 2x, as onevision does for video."""
        batch_frames, _, dim = features.shape
        side = self.patches_per_side
        pooled_side = math.ceil(side / 2)
        features = features.reshape(batch_frames, side, side, dim)
        features = mx.transpose(features, axes=(0, 3, 1, 2))
        features = resize_bilinear(
            features, (pooled_side, pooled_side), antialias=False
        )
        features = mx.transpose(features, axes=(0, 2, 3, 1))
        return features.reshape(batch_frames, pooled_side * pooled_side, dim)

    def get_video_features(self, pixel_values_videos: mx.array) -> mx.array:
        if pixel_values_videos.ndim == 5:
            batch_size, frames = pixel_values_videos.shape[:2]
            flat = pixel_values_videos.reshape(-1, *pixel_values_videos.shape[2:])
        else:
            batch_size, frames = 1, pixel_values_videos.shape[0]
            flat = pixel_values_videos

        *_, hidden_states = self.vision_tower(
            flat.transpose(0, 2, 3, 1), output_hidden_states=True
        )
        selected = self._select_features(hidden_states)
        features = self.multi_modal_projector(selected)
        features = self.apply_pooling(features)
        features = features.reshape(batch_size, frames * features.shape[1], -1)

        if self.image_newline is not None:
            newline = mx.broadcast_to(
                self.image_newline[None, None, :].astype(features.dtype),
                (features.shape[0], 1, features.shape[-1]),
            )
            features = mx.concatenate([features, newline], axis=1)

        return features.reshape(-1, features.shape[-1])

    def _scatter_features(
        self,
        inputs_embeds: mx.array,
        input_ids: mx.array,
        features: mx.array,
        token_index: int,
        kind: str,
    ) -> mx.array:
        """Replace every ``token_index`` placeholder with the next row of ``features``."""
        mask = input_ids == token_index
        num_placeholders = int(mask.sum().item())
        if num_placeholders != features.shape[0]:
            raise ValueError(
                f"{kind} features and {kind} tokens do not match: "
                f"tokens {num_placeholders}, features {features.shape[0]}"
            )

        features = features.astype(inputs_embeds.dtype)
        flat_positions = mx.cumsum(mask.reshape(-1).astype(mx.int32)) - 1
        positions = flat_positions.reshape(mask.shape)
        gathered = features[mx.clip(positions, 0, features.shape[0] - 1)]
        return mx.where(mask[..., None], gathered, inputs_embeds)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        if pixel_values is None and pixel_values_videos is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is not None:
            cached = kwargs.get("cached_image_features", None)
            if cached is not None:
                image_features = cached
            else:
                image_features = self.get_image_features(
                    pixel_values, kwargs.get("image_sizes", None)
                )
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                image_features,
                self.config.image_token_index,
                "Image",
            )

        if pixel_values_videos is not None:
            video_features = self.get_video_features(pixel_values_videos)
            inputs_embeds = self._scatter_features(
                inputs_embeds,
                input_ids,
                video_features,
                self.config.video_token_index,
                "Video",
            )

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    @staticmethod
    def sanitize(weights):
        """Accept both the published key layout and transformers' newer one.

        Hub checkpoints are exported as ``language_model.model.*`` / ``vision_tower.*``;
        transformers >= 5 nests the same tensors under ``model.*`` with the language
        model flattened.
        """
        renames = (
            ("model.vision_tower.", "vision_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("model.language_model.", "language_model.model."),
            ("model.image_newline", "image_newline"),
        )
        sanitized = {}
        for key, value in weights.items():
            for prefix, replacement in renames:
                if key.startswith(prefix):
                    key = replacement + key[len(prefix) :]
                    break
            else:
                if key == "lm_head.weight":
                    key = "language_model.lm_head.weight"
            if key.startswith("vision_tower.") and not key.startswith(
                "vision_tower.vision_model."
            ):
                key = "vision_tower.vision_model." + key[len("vision_tower.") :]
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array = None,
        mask: mx.array = None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=features.inputs_embeds
        )
