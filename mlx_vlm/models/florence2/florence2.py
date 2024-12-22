import glob
import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_map

from .language import LanguageModel, TextConfig
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    """Configuration class for Florence2."""

    vision_config: VisionConfig
    text_config: TextConfig
    model_type: str = "florence2"
    vocab_size: int = 50265
    max_position_embeddings: int = 1024
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    image_token_index: int = 0
    image_feature_source: List[str] = field(
        default_factory=lambda: ["temporal_avg_pool", "spatial_avg_pool"]
    )
    visual_temporal_embedding: Optional[dict] = field(
        default_factory=lambda: {"type": "COSINE", "max_temporal_embeddings": 100}
    )
    image_pos_embed: Optional[dict] = field(
        default_factory=lambda: {"type": "learned_abs_2d", "max_pos_embeddings": 50}
    )

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def shift_tokens_right(
    input_ids: mx.array, pad_token_id: int, decoder_start_token_id: int
) -> mx.array:
    """Shift input tokens right, adding decoder start token at beginning."""
    shifted = mx.roll(input_ids, 1, axis=-1)
    shifted = tree_map(lambda x: x.at[:, 0].set(decoder_start_token_id), shifted)
    shifted = mx.where(shifted == -100, pad_token_id, shifted)
    return shifted


class LearnedPositionEmbedding2D(nn.Module):
    """2D learned position embeddings."""

    def __init__(self, embedding_dim: int = 256, num_pos: int = 50):
        super().__init__()
        self.row_embeddings = nn.Embedding(num_pos, embedding_dim // 2)
        self.column_embeddings = nn.Embedding(
            num_pos, embedding_dim - (embedding_dim // 2)
        )

    def __call__(self, x):
        batch_size, height, width, channels = x.shape
        width_pos = mx.arange(width)
        height_pos = mx.arange(height)

        x_emb = self.column_embeddings(width_pos)
        y_emb = self.row_embeddings(height_pos)

        pos = mx.concatenate(
            [
                mx.broadcast_to(x_emb[None, :, :], (height, width, x_emb.shape[-1])),
                mx.broadcast_to(y_emb[:, None, :], (height, width, y_emb.shape[-1])),
            ],
            axis=-1,
        )

        return mx.broadcast_to(pos[None, ...], (batch_size, height, width, channels))


class PositionalEmbeddingCosine1D(nn.Module):
    """
    MLX implementation of 1D cosine positional embeddings.

    Args:
        embed_dim: The dimension of the embeddings
        max_seq_len: The maximum length to precompute the positional encodings
    """

    def __init__(self, embed_dim: int = 512, max_seq_len: int = 1024) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Generate position indices and dimension indices
        position = mx.arange(max_seq_len)
        dim_pos = mx.arange(0, embed_dim // 2)  # Half the dimensions for sin/cos pairs

        # Calculate frequency bands
        factor = math.log(10000)
        denominator = mx.exp(-factor * dim_pos / embed_dim)

        # Create position-frequency product matrix [max_seq_len, embed_dim//2]
        frequencies = mx.reshape(position, (-1, 1)) * denominator

        # Calculate sin and cos values [max_seq_len, embed_dim//2]
        sin_values = mx.sin(frequencies)
        cos_values = mx.cos(frequencies)

        # Interleave sin and cos values to create final embeddings
        pos_idx_to_embed = mx.zeros((max_seq_len, embed_dim))
        pos_idx_to_embed = mx.concatenate(
            [mx.expand_dims(sin_values, -1), mx.expand_dims(cos_values, -1)], axis=-1
        ).reshape(max_seq_len, embed_dim)

        # Store the positional embeddings
        self.pos_idx_to_embed = pos_idx_to_embed

    def __call__(self, seq_embeds: mx.array) -> mx.array:
        """
        Apply positional embeddings to the input sequence.

        Args:
            seq_embeds: Input sequence embeddings with shape:
                - [T, D] where T is sequence length and D is embedding dimension
                - [B, T, D] where B is batch size

        Returns:
            Positional embeddings matching input shape
        """
        shape_len = len(seq_embeds.shape)
        assert 2 <= shape_len <= 3, "Input must be 2D or 3D tensor"

        len_seq = seq_embeds.shape[-2]
        assert (
            len_seq <= self.max_seq_len
        ), f"Sequence length {len_seq} exceeds maximum length {self.max_seq_len}"

        # Get relevant portion of pre-computed embeddings
        pos_embeds = self.pos_idx_to_embed[:len_seq]

        # Add batch dimension if input is 3D
        if shape_len == 3:
            pos_embeds = mx.expand_dims(pos_embeds, 0)

        return pos_embeds


class Model(nn.Module):
    """Florence-2 model for conditional generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Initialize vision model
        self.vision_tower = VisionModel(config.vision_config)

        # Initialize language model
        self.language_model = LanguageModel(config.text_config)

        # Image projection layers
        image_dim = config.vision_config.dim_embed[-1]
        text_dim = config.text_config.d_model
        self.image_projection = mx.zeros((image_dim, text_dim))

        self.image_proj_norm = nn.LayerNorm(text_dim)

        # Position embeddings
        if config.image_pos_embed["type"] == "learned_abs_2d":
            self.image_pos_embed = LearnedPositionEmbedding2D(
                embedding_dim=image_dim,
                num_pos=config.image_pos_embed["max_pos_embeddings"],
            )
        else:
            raise NotImplementedError(
                f"Position embedding type {config.image_pos_embed['type']} not supported"
            )

        # Temporal embeddings
        if config.visual_temporal_embedding["type"] == "COSINE":
            self.visual_temporal_embed = PositionalEmbeddingCosine1D(
                embed_dim=image_dim,
                max_seq_len=config.visual_temporal_embedding["max_temporal_embeddings"],
            )
        else:
            raise NotImplementedError(
                f"Temporal embedding type {config.visual_temporal_embedding['type']} not supported"
            )

        self.image_feature_source = config.image_feature_source

    def _encode_image(self, pixel_values, extract_features=True):
        """Encode image using vision model and add position embeddings."""
        T = 1  # Single frame for now

        # Get vision features
        if extract_features:
            batch_size, C, H, W = pixel_values.shape
            x = self.vision_tower(pixel_values)
        else:
            x = pixel_values
            batch_size = pixel_values.shape[0]

        # Assuming this is part of a class method, keeping the same structure
        if self.image_pos_embed is not None:
            # Reshape to (batch_size * T, -1, feature_dim)
            x = mx.reshape(x, (batch_size * T, -1, x.shape[-1]))
            num_tokens = x.shape[-2]
            h, w = int(num_tokens**0.5), int(num_tokens**0.5)
            assert h * w == num_tokens, "only support square feature maps for now"
            # Reshape to (batch_size * T, h, w, feature_dim)
            x = mx.reshape(x, (batch_size * T, h, w, x.shape[-1]))
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            # Reshape to (batch_size, T * h * w, feature_dim)
            x = mx.reshape(x, (batch_size, T * h * w, x.shape[-1]))

        if self.visual_temporal_embed is not None:
            # Reshape for temporal embedding
            x_temp = mx.reshape(x, (batch_size, T, -1, x.shape[-1]))
            temporal_input = x_temp[:, :, 0]
            visual_temporal_embed = self.visual_temporal_embed(temporal_input)
            # Expand dims for broadcasting
            visual_temporal_embed = mx.expand_dims(visual_temporal_embed, axis=2)
            x = mx.reshape(x, (batch_size, T, -1, x.shape[-1])) + visual_temporal_embed

        x_feat_dict = {}

        # Spatial average pooling
        x_spatial = mx.reshape(x, (batch_size, T, -1, x.shape[-1]))
        spatial_avg_pool_x = mx.mean(x_spatial, axis=2)
        x_feat_dict["spatial_avg_pool"] = spatial_avg_pool_x

        # Temporal average pooling
        x_temporal = mx.reshape(x, (batch_size, T, -1, x.shape[-1]))
        temporal_avg_pool_x = mx.mean(x_temporal, axis=1)
        x_feat_dict["temporal_avg_pool"] = temporal_avg_pool_x

        # Last frame features
        x_last = mx.reshape(x, (batch_size, T, -1, x.shape[-1]))
        x = x_last[:, -1]
        x_feat_dict["last_frame"] = x

        # Gather features based on source configuration
        new_x = []
        for _image_feature_source in self.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError(
                    f"invalid image feature source: {_image_feature_source}"
                )
            new_x.append(x_feat_dict[_image_feature_source])

        # Concatenate features
        x = mx.concatenate(new_x, axis=1)

        # Final projection and normalization
        x = x @ self.image_projection
        x = self.image_proj_norm(x)

        return x

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds=None):
        batch_size, image_token_length, _ = image_features.shape
        image_attention_mask = mx.ones((batch_size, image_token_length))

        if inputs_embeds is None:
            return image_features, image_attention_mask

        task_prefix_embeds = inputs_embeds
        task_prefix_attention_mask = mx.ones((batch_size, task_prefix_embeds.shape[1]))

        if len(task_prefix_attention_mask.shape) == 3:
            task_prefix_attention_mask = task_prefix_attention_mask[:, 0]

        # Concatenate image features and task prefix embeddings
        inputs_embeds = mx.concatenate([image_features, task_prefix_embeds], axis=1)
        attention_mask = mx.concatenate(
            [image_attention_mask, task_prefix_attention_mask], axis=1
        )
        return inputs_embeds, attention_mask

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        cache=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """Forward pass."""
        attention_mask = None
        decoder_inputs_embeds = None

        # Process image if provided
        if pixel_values is not None:
            image_features = self._encode_image(pixel_values)

            # Get input embeddings if needed
            inputs_embeds = None
            if input_ids is not None:
                inputs_embeds = self.language_model.model.shared(input_ids)

            # Merge image features with text embeddings
            inputs_embeds, attention_mask = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds
            )
        else:
            inputs_embeds = None
            attention_mask = None

        # Handle decoder input IDs
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.bos_token_id
            )

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_start_token_id = getattr(
                self.config, "decoder_start_token_id", 0
            )  # 2 is common for many models
            if decoder_start_token_id is None:
                decoder_start_token_id = 0

            decoder_input_ids = mx.array([decoder_start_token_id])[None, :]
            decoder_inputs_embeds = self.language_model.model.shared(decoder_input_ids)
            decoder_input_ids = None

        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            cache=cache,
        )

        return outputs

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
        model_config.text_config = TextConfig.from_dict(model_config)

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
        sanitized_weights = {}
        for k, v in weights.items():
            if "final_logits_bias" in k:
                continue
            sanitized_weights[k] = v
        return sanitized_weights
