from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoProcessor

from ..base import InputEmbeddingsFeatures
from ..deepseekocr.language import LanguageModel
from ..deepseekocr.sam import SAMEncoder
from .config import ModelConfig, SAMViTConfig
from .processing_deepseekocr import DeepseekVLV2Processor
from .vision import VisionModel

AutoProcessor.register("deepseekocr", DeepseekVLV2Processor)


class MlpProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.projector_config.projector_type == "linear":
            self.layers = nn.Linear(
                config.projector_config.input_dim, config.projector_config.n_embed
            )
        else:
            raise ValueError(
                f"Unknown projector type: {config.projector_config.projector_type}"
            )

    def __call__(self, x):
        return self.layers(x)


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
            final_out_chans=896,  # OCR-2 uses 896 output channels (vs 1024 in OCR)
        )
        self.language_model = LanguageModel(config.text_config)
        self.projector = MlpProjector(config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # view_separator is loaded from model weights (mapped from view_seperator)
        # Initialize with zeros - will be overwritten when weights are loaded
        if self.tile_tag == "2D":
            # <|view_separator|> - marks end of image features
            # Note: This must be defined as an mx.array for weight loading to work
            self.view_separator = mx.zeros((config.projector_config.n_embed,))
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
        **kwargs,
    ):
        input_embeds = self.language_model.model.embed_tokens(input_ids)

        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

        # pixel_values is a list: [patches, global_images]
        # For DeepSeek-OCR-2 with Qwen2 encoder, we only use global_images
        if isinstance(pixel_values, list):
            _, global_images = pixel_values
        else:
            global_images = pixel_values

        # Check if we have valid pixel values
        if mx.sum(global_images).item() == 0:
            return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

        # Process images through SAM -> Qwen2 -> Projector pipeline
        batch_size = input_ids.shape[0]

        for idx in range(batch_size):
            # Get the image for this batch item
            # global_images is (N, C, H, W) where N is number of images
            image = global_images[idx : idx + 1]  # Keep batch dimension

            # Transpose to (B, H, W, C) for MLX conv2d
            image_hwc = image.transpose(0, 2, 3, 1)

            # SAM encoder: (B, H, W, C) -> (B, H', W', 896)
            sam_features = self.sam_model(image_hwc)

            # Qwen2 encoder: (B, H', W', 896) -> (B, 256, 896)
            vision_features = self.vision_model(image_hwc, sam_features)

            # Linear projector: (B, 256, 896) -> (B, 256, 1280)
            vision_features = self.projector(vision_features)

            # Remove batch dimension: (256, 1280)
            vision_features = vision_features[0]

            # Add view_separator: (257, 1280)
            vision_features = mx.concatenate(
                [vision_features, self.view_separator[None, :]], axis=0
            )

            # Find positions where images should be placed
            if images_seq_mask is not None:
                image_indices = np.where(images_seq_mask[idx])[0].tolist()
                # Assign image features to those positions
                if len(image_indices) > 0:
                    # Truncate or pad vision_features to match number of image tokens
                    num_positions = len(image_indices)
                    if vision_features.shape[0] >= num_positions:
                        input_embeds[idx, image_indices] = vision_features[
                            :num_positions
                        ]
                    else:
                        # Pad with zeros if needed
                        input_embeds[idx, image_indices[: vision_features.shape[0]]] = (
                            vision_features
                        )

        return InputEmbeddingsFeatures(inputs_embeds=input_embeds)

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

        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, images_spatial_crop, images_seq_mask
        )

        logits = self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
        )
        return logits

    @staticmethod
    def sanitize(weights):
        def transform_key(key):
            # Handle Qwen2 encoder weights from HuggingFace format
            # HuggingFace: model.qwen2_model.model.model.layers.X...
            # MLX: vision_model.qwen2_encoder.layers.X...
            if "qwen2_model.model.model.layers" in key:
                key = key.replace(
                    "model.qwen2_model.model.model.layers",
                    "vision_model.qwen2_encoder.layers",
                )

            # Handle Qwen2 encoder norm
            if "qwen2_model.model.model.norm" in key:
                key = key.replace(
                    "model.qwen2_model.model.model.norm",
                    "vision_model.qwen2_encoder.norm",
                )

            # Handle query weights (learnable queries for Qwen2 encoder)
            # For 1024x1024 images, SAM outputs 16x16=256 features, so use query_1024
            # query_1024: (256, 896) - used for 1024x1024 images
            # query_768: (144, 896) - used for 768x768 images
            if "model.qwen2_model.query_1024.weight" in key:
                key = key.replace(
                    "model.qwen2_model.query_1024.weight",
                    "vision_model.qwen2_encoder.query_1024",
                )
            elif "model.qwen2_model.query_1024" in key:
                key = key.replace(
                    "model.qwen2_model.query_1024",
                    "vision_model.qwen2_encoder.query_1024",
                )
            # Also handle query_768 for smaller images (not currently used but keep for future)
            if "model.qwen2_model.query_768.weight" in key:
                key = key.replace(
                    "model.qwen2_model.query_768.weight",
                    "vision_model.qwen2_encoder.query_768",
                )
            elif "model.qwen2_model.query_768" in key:
                key = key.replace(
                    "model.qwen2_model.query_768",
                    "vision_model.qwen2_encoder.query_768",
                )

            # Language model layers
            if (
                "model.layers" in key
                and "language_model" not in key
                and "qwen2" not in key
            ):
                key = key.replace("model.layers", "language_model.model.layers")

            if (
                "model.embed_tokens" in key
                and "language_model" not in key
                and "qwen2" not in key
            ):
                key = key.replace(
                    "model.embed_tokens", "language_model.model.embed_tokens"
                )

            if (
                "model.norm" in key
                and "language_model" not in key
                and "qwen2" not in key
            ):
                key = key.replace("model.norm", "language_model.model.norm")

            if "model.vision_model" in key:
                key = key.replace("model.vision_model", "vision_model")

            if "model.sam_model" in key:
                key = key.replace("model.sam_model", "sam_model")

            if "model.projector" in key:
                key = key.replace("model.projector", "projector")

            # Note: HuggingFace has typo "view_seperator" (e instead of a)
            if "model.view_seperator" in key:
                key = key.replace("model.view_seperator", "view_separator")

            if "lm_head.weight" in key and "language_model" not in key:
                key = key.replace("lm_head.weight", "language_model.lm_head.weight")

            return key

        return {transform_key(k): v for k, v in weights.items()}
