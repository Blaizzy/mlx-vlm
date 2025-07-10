from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .audio import AudioModel
from .config import ModelConfig, TextConfig
from .language import Gemma3nRMSNorm, LanguageModel
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    """MLX implementation of PyTorch's masked_scatter"""

    # Convert mask to boolean once
    mask = mask.astype(mx.bool_)

    # Early exit
    if not mask.any():
        return mx.broadcast_to(input_tensor, mask.shape)

    # Flatten everything once
    input_shape = mask.shape
    result_flat = mx.broadcast_to(input_tensor, input_shape).flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    # Create selection indices using cumulative sum
    selection_mask = mx.cumsum(mask_flat.astype(mx.int32)) - 1

    # Bound check and create source selection
    source_len = len(source_flat)
    bounded_indices = selection_mask % source_len

    # Vectorized selection from source
    selected_values = source_flat[bounded_indices]

    result_flat = mx.where(mask_flat, selected_values, result_flat)

    return result_flat.reshape(input_shape)


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens into language model space."""

    def __init__(self, multimodal_config: ModelConfig, text_config: TextConfig):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = nn.Embedding(self.vocab_size, self.multimodal_hidden_size)
        self.hard_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size, eps=self.eps
        )
        self.soft_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size, eps=self.eps
        )
        self.embedding_projection = nn.Linear(
            self.multimodal_hidden_size, self.text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            self.text_hidden_size, eps=self.eps, with_scale=False
        )

    def __call__(
        self, input_ids: mx.array = None, inputs_embeds: mx.array = None
    ) -> mx.array:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:

            hard_emb = self.embedding(input_ids - self.vocab_offset)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj = self.embedding_projection(emb_norm)
        projected = self.embedding_post_projection_norm(emb_norm_proj)
        return projected


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        # Text
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input

        # Vision
        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = Gemma3nMultimodalEmbedder(
            config.vision_config, text_config=config.text_config
        )

        # Audio
        self.audio_tower = AudioModel(config.audio_config)
        self.embed_audio = Gemma3nMultimodalEmbedder(
            config.audio_config, text_config=config.text_config
        )

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        **kwargs,
    ):

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        per_layer_inputs_mask = mx.logical_and(
            input_ids >= 0, input_ids < self.vocab_size_per_layer_input
        )
        per_layer_inputs_tokens = mx.where(
            per_layer_inputs_mask, input_ids, mx.zeros_like(input_ids)
        )
        per_layer_inputs = self.language_model.model.get_per_layer_inputs(
            per_layer_inputs_tokens
        )
        if pixel_values is None and input_features is None:
            return inputs_embeds, per_layer_inputs

        if input_ids is not None:

            # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
            vision_mask = mx.logical_and(
                input_ids >= self.embed_vision.vocab_offset,
                input_ids < self.embed_audio.vocab_offset,
            )
            dummy_vision_token_id = (
                self.embed_vision.vocab_offset + self.embed_vision.vocab_size - 1
            )
            vision_tokens = mx.where(vision_mask, input_ids, dummy_vision_token_id)
            vision_embeds_flat = self.embed_vision(input_ids=vision_tokens)
            inputs_embeds = mx.where(
                vision_mask[..., None], vision_embeds_flat, inputs_embeds
            )

            # Handle audio tokens (>= embed_audio.vocab_offset)
            audio_mask = input_ids >= self.embed_audio.vocab_offset
            dummy_audio_token_id = (
                self.embed_audio.vocab_offset + self.embed_audio.vocab_size - 1
            )

            audio_tokens = mx.where(audio_mask, input_ids, dummy_audio_token_id)
            audio_embeds_flat = self.embed_audio(input_ids=audio_tokens)
            inputs_embeds = mx.where(
                audio_mask[..., None], audio_embeds_flat, inputs_embeds
            )
        else:
            per_layer_inputs = None

        # Vision features
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values, self.vision_tower, self.config, self.embed_vision
            )

            modality = "image"
            inputs_embeds = self.merge_multimodal_and_text(
                inputs_embeds,
                image_features,
                self.construct_special_modality_mask(
                    input_ids,
                    inputs_embeds,
                    self.config.image_token_id,
                    modality=modality,
                ),
                modality=modality,
            )

        # Audio features
        if input_features is not None:
            audio_features, audio_mask = self.get_audio_features(
                input_features, ~input_features_mask
            )
            audio_padding_ids = mx.array([[self.vocab_size - 1]])
            audio_padding_embs = self.embed_audio(input_ids=audio_padding_ids)
            audio_features = mx.where(
                audio_mask[..., None], audio_padding_embs, audio_features
            )

            audio_batch_size, audio_seq_len, audio_embed_dim = audio_features.shape
            extra_padding_tokens = (
                self.config.audio_soft_tokens_per_image - audio_seq_len
            )
            extra_padding_features = mx.broadcast_to(
                audio_padding_embs,
                (audio_batch_size, extra_padding_tokens, audio_embed_dim),
            )

            audio_features = mx.concatenate(
                (audio_features, extra_padding_features), axis=1
            )
            modality = "audio"
            inputs_embeds = self.merge_multimodal_and_text(
                inputs_embeds,
                audio_features,
                self.construct_special_modality_mask(
                    input_ids,
                    inputs_embeds,
                    self.config.audio_token_id,
                    modality=modality,
                ),
                modality=modality,
            )

        return inputs_embeds, per_layer_inputs

    def get_audio_features(self, input_features, input_features_mask):
        audio_outputs, audio_mask = self.audio_tower(
            input_features, input_features_mask
        )
        return self.embed_audio(inputs_embeds=audio_outputs), audio_mask

    @staticmethod
    def get_image_features(pixel_values, vision_tower, config, embed_vision):
        vision_outputs = vision_tower(
            pixel_values,
            output_hidden_states=True,
        )
        vision_outputs = vision_outputs.transpose(0, 3, 1, 2)
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            config.vision_config.hidden_size,
            config.vision_soft_tokens_per_image,
        ).transpose(0, 2, 1)

        # Normalize and embed the soft tokens into language model space.
        vision_outputs *= config.vision_config.hidden_size**0.5
        return embed_vision(inputs_embeds=vision_outputs)

    def construct_special_modality_mask(
        self, input_ids, inputs_embeds, token_id, modality="image"
    ):
        if input_ids is None:
            embed_fn = (
                self.embed_audio
                if modality == "audio"
                else self.language_model.model.embed_tokens
            )
            special_modality_mask = inputs_embeds == embed_fn(
                input_ids=mx.array([token_id])
            )
        else:
            special_modality_mask = mx.expand_dims(input_ids == token_id, -1)
            special_modality_mask = mx.broadcast_to(
                special_modality_mask, inputs_embeds.shape
            )
        return special_modality_mask

    @staticmethod
    def merge_multimodal_and_text(
        inputs_embeds, features, special_modality_mask, modality="image"
    ):
        # Count special tokens by summing the mask
        modality_tokens_in_text = special_modality_mask.sum()
        feature_tokens = features.size

        if modality_tokens_in_text != feature_tokens:
            raise ValueError(
                f"Number of {modality}s does not match number of special {modality} tokens in the input text. "
                f"Got {modality_tokens_in_text} {modality} tokens in the text and "
                f"{feature_tokens} tokens from {modality} embeddings."
            )
        features = features.astype(inputs_embeds.dtype)

        inputs_embeds = masked_scatter(inputs_embeds, special_modality_mask, features)
        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        # Audio features
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        inputs_embeds, per_layer_inputs = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            input_features=input_features,
            input_features_mask=input_features_mask,
            **kwargs,
        )

        logits = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
        )
        return logits

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            # if "vision_tower" not in k and "embed_vision" not in k:
            if k.startswith("model."):
                sanitized_weights[".".join(k.split(".")[1:])] = v
            else:
                sanitized_weights[k] = v
        return sanitized_weights

    @property
    def layers(self):
        return self.language_model.model.layers
