import math
from fractions import Fraction
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from . import processing_granite4_vision  # noqa: F401
from .config import ModelConfig
from .downsampling import WindowQFormerDownsampler
from .language import LanguageModel
from .vision import VisionModel


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """Calculate grid shape for AnyRes image processing."""
    from transformers.image_processing_utils import select_best_resolution

    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()
    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """Remove padding from image tensor based on original dimensions."""
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1], tensor.shape[2]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded = tensor[:, :, padding : current_width - padding]

    return unpadded


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

        # Deepstack projectors
        self.layerwise_projectors = [
            WindowQFormerDownsampler(config)
            for _ in range(len(config.deepstack_layer_map or []))
        ]

        # Spatial projectors
        self.spatial_projectors = None
        if config.use_spatial_sampling:
            self.spatial_projectors = [
                WindowQFormerDownsampler(config, spatial_offset=i) for i in range(4)
            ]

        # Image newline
        self.image_newline = None
        if config.use_image_newline_parameter:
            embed_std = 1 / math.sqrt(config.text_config.hidden_size)
            self.image_newline = (
                mx.random.normal((config.text_config.hidden_size,)) * embed_std
            )

    def _pack_and_unpad_image_features(
        self, image_features, image_sizes, vision_feature_select_strategy
    ):
        """Pack and unpad image features for AnyRes processing."""
        new_image_features = []
        ds_rate = Fraction(self.config.downsample_rate)
        patch_size = self.config.vision_config.image_size

        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]

                height = width = (
                    self.config.vision_config.image_size
                    // self.config.vision_config.patch_size
                )
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    patch_size,
                )

                # Apply downsample rate
                height = int(height * ds_rate)
                width = int(width * ds_rate)

                # Reshape: (num_patches, h*w, C) → (num_ph, num_pw, h, w, C)
                image_feature = image_feature.reshape(
                    num_patch_height, num_patch_width, height, width, -1
                )
                # Permute to (C, num_ph, h, num_pw, w) → flatten
                image_feature = mx.transpose(
                    image_feature, axes=(4, 0, 2, 1, 3)
                )  # (C, nph, h, npw, w)
                C = image_feature.shape[0]
                image_feature = image_feature.reshape(
                    C, num_patch_height * height, num_patch_width * width
                )

                # Unpad
                image_feature = unpad_image(image_feature, image_sizes[image_idx])

                # Add image newline
                if self.image_newline is not None:
                    C, H, W = image_feature.shape
                    newline = mx.broadcast_to(
                        self.image_newline[:, None, None], (C, H, 1)
                    )
                    image_feature = mx.concatenate([image_feature, newline], axis=-1)

                # Flatten to (seq_len, C)
                image_feature = image_feature.reshape(C, -1).T
                image_feature = mx.concatenate(
                    [base_image_feature, image_feature], axis=0
                )
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = mx.concatenate(
                        [image_feature, self.image_newline[None]], axis=0
                    )

            new_image_features.append(image_feature)

        return new_image_features

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        image_sizes = kwargs.get("image_sizes", None)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
        else:
            # Run vision tower once
            *_, hidden_states = self.vision_tower(
                pixel_values[0].transpose(0, 2, 3, 1), output_hidden_states=True
            )

        num_patches = pixel_values[0].shape[0]
        image_num_patches = [num_patches]

        # Build all deepstack + spatial features
        all_features = []
        target_layers = []

        # Deepstack features
        if self.config.deepstack_layer_map is not None:
            for proj_idx, (vision_layer, llm_layer) in enumerate(
                self.config.deepstack_layer_map
            ):
                selected = hidden_states[vision_layer]
                if self.config.vision_feature_select_strategy == "default":
                    selected = selected[:, 1:]

                projected = self.layerwise_projectors[proj_idx](selected)
                projected_split = [projected]

                if image_sizes is not None:
                    packed = self._pack_and_unpad_image_features(
                        projected_split,
                        image_sizes,
                        self.config.vision_feature_select_strategy,
                    )
                else:
                    packed = [projected[0]]

                all_features.append(packed)
                target_layers.append(llm_layer)

        # Spatial features
        if (
            self.config.use_spatial_sampling
            and self.spatial_projectors is not None
            and self.config.spatial_target_layers is not None
        ):
            spatial_hidden = hidden_states[self.config.spatial_vision_layer]
            if self.config.vision_feature_select_strategy == "default":
                spatial_hidden = spatial_hidden[:, 1:]

            for group_idx, llm_layer in enumerate(self.config.spatial_target_layers):
                projected = self.spatial_projectors[group_idx](spatial_hidden)
                projected_split = [projected]

                if image_sizes is not None:
                    packed = self._pack_and_unpad_image_features(
                        projected_split,
                        image_sizes,
                        self.config.vision_feature_select_strategy,
                    )
                else:
                    packed = [projected[0]]

                all_features.append(packed)
                target_layers.append(llm_layer)

        # Merge features into embeddings at image token positions
        image_token_index = self.config.image_token_index

        if len(all_features) > 0:
            # Use first feature set to determine token count and positions
            first_features = mx.concatenate([f for f in all_features[0]], axis=0)
            num_image_tokens = first_features.shape[0]

            # Create vision position mask
            vision_mask = np.array(input_ids) == image_token_index  # (1, seq_len)

            # Zero out image positions in embeddings
            vision_mask_mx = mx.array(vision_mask)
            inputs_embeds = mx.where(
                vision_mask_mx[..., None], mx.zeros_like(inputs_embeds), inputs_embeds
            )

            # Build full-sequence deepstack features via vectorized scatter
            seq_len = inputs_embeds.shape[1]
            hidden_size = self.config.text_config.hidden_size

            # Build cumulative index mapping: mask position i → feature index
            mask_int = vision_mask_mx.astype(mx.int32)  # (1, seq_len)
            feat_indices = mx.cumsum(mask_int, axis=1) - 1  # (1, seq_len)

            deepstack_list = []
            for feat_set in all_features:
                feat_concat = mx.concatenate(
                    [f for f in feat_set], axis=0
                )  # (num_image_tokens, hidden)
                feat_concat = feat_concat.astype(inputs_embeds.dtype)

                # Clamp indices to valid range
                clamped = mx.clip(feat_indices, 0, feat_concat.shape[0] - 1)
                # Gather features at mapped indices
                gathered = feat_concat[clamped[0]]  # (seq_len, hidden)
                # Zero out non-image positions
                full_feat = mx.where(
                    vision_mask_mx[..., None],
                    gathered[None],
                    mx.zeros_like(inputs_embeds),
                )
                deepstack_list.append(full_feat)

            deepstack_visual_embeds = mx.concatenate(deepstack_list, axis=0)
        else:
            deepstack_visual_embeds = None
            target_layers = None
            vision_mask_mx = None

        # Store target layers on the language model for use during forward
        if target_layers:
            self.language_model.model._deepstack_target_layers = target_layers
        else:
            self.language_model.model._deepstack_target_layers = None

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=(
                vision_mask_mx if deepstack_visual_embeds is not None else None
            ),
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: mx.array,
        cache=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )

        # Build target layer list for deepstack injection
        target_layers = []
        if self.config.deepstack_layer_map is not None:
            target_layers.extend([lyr for _, lyr in self.config.deepstack_layer_map])
        if (
            self.config.use_spatial_sampling
            and self.config.spatial_target_layers is not None
        ):
            target_layers.extend(self.config.spatial_target_layers)

        logits = self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            deepstack_visual_embeds=input_embeddings_features.deepstack_visual_embeds,
            deepstack_target_layers=target_layers if target_layers else None,
            visual_pos_masks=input_embeddings_features.visual_pos_masks,
        )
        return logits

    @staticmethod
    def sanitize(weights):
        # Merge LoRA adapters into base weights
        lora_a = {}
        lora_b = {}
        for k, v in weights.items():
            if "lora_A.weight" in k:
                # Strip base_model.model. prefix to get the module path
                base_key = k.replace("lora_A.weight", "weight")
                if base_key.startswith("base_model.model."):
                    base_key = base_key[len("base_model.model.") :]
                lora_a[base_key] = v
            elif "lora_B.weight" in k:
                base_key = k.replace("lora_B.weight", "weight")
                if base_key.startswith("base_model.model."):
                    base_key = base_key[len("base_model.model.") :]
                lora_b[base_key] = v

        # lora_alpha / r = 256 / 256 = 1.0 for this model
        lora_scaling = 1.0
        for base_key in lora_a:
            if base_key in weights and base_key in lora_b:
                A = lora_a[base_key]
                B = lora_b[base_key]
                weights[base_key] = weights[base_key] + (B @ A) * lora_scaling

        sanitized = {}
        for k, v in weights.items():
            # Skip LoRA adapter weights (already merged)
            if "lora_A" in k or "lora_B" in k or k.startswith("base_model."):
                continue

            new_k = k

            # Strip "model." prefix from non-vision, non-lm_head keys
            if new_k.startswith("model."):
                suffix = new_k[len("model.") :]
                if suffix.startswith("language_model."):
                    # model.language_model.X → language_model.model.X
                    lm_suffix = suffix[len("language_model.") :]
                    new_k = f"language_model.model.{lm_suffix}"
                else:
                    new_k = suffix

            # Handle lm_head
            if new_k == "lm_head.weight":
                new_k = "language_model.lm_head.weight"

            sanitized[new_k] = v

        # Handle tied embeddings
        lm_head_key = "language_model.lm_head.weight"
        embed_key = "language_model.model.embed_tokens.weight"
        if lm_head_key not in sanitized and embed_key in sanitized:
            sanitized[lm_head_key] = sanitized[embed_key]

        return sanitized
