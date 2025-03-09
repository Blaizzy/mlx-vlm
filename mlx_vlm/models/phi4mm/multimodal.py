import math
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .audio import AudioModel
from .vision import VisionModel

# Special token ids for different modalities
_IMAGE_SPECIAL_TOKEN_ID = 200010  # '<|endoftext10|>'
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'
_COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE = [-9999, -1]  # For backward compatibility
_COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE = [
    float("-inf"),
    -10000,
]  # For backward compatibility


class Phi4MMImageAudioEmbedding(nn.Module):
    """Image-audio embedding combined module."""

    def __init__(self, config, **kwargs):
        super().__init__()

        # Store vocabulary size
        self.vocab_size = config.vocab_size

        # Configure token IDs for different modalities
        self.image_input_id = kwargs.get("image_input_id", -1)
        self.audio_input_id = kwargs.get("audio_input_id", -10000)
        assert (
            self.image_input_id != self.audio_input_id
        ), "image_input_id and audio_input_id should be different"

        # Create embedding layers for different modalities
        self.image_embd_layer_kwargs = kwargs.get("image_embd_layer", {})
        self.image_embed = VisionModel(**self.image_embd_layer_kwargs)

        self.audio_embd_layer_kwargs = kwargs.get("audio_embd_layer", {})
        self.audio_embed = AudioModel(config, **self.audio_embd_layer_kwargs)

        # Initialize cache attributes
        self.input_image_embeds = None
        self.image_sizes = None
        self.image_attention_mask = None
        self.input_audio_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        """Initialize audio embedding with pretrained weights."""
        self.audio_embed.post_init(audio_config)

    def set_input_image_embeds(self, input_image_embeds):
        self.input_image_embeds = input_image_embeds

    def set_image_sizes(self, image_sizes):
        self.image_sizes = image_sizes

    def set_img_attn_mask(self, image_attention_mask):
        self.image_attention_mask = image_attention_mask

    def set_input_audio_embeds(self, input_audio_embeds):
        self.input_audio_embeds = input_audio_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes):
        self.audio_embed_sizes = audio_embed_sizes

    def __call__(
        self,
        input_ids,
        input_embeds,
        input_image_embeds=None,
        input_audio_embeds=None,
        image_sizes=None,
        image_attention_mask=None,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        wte=None,
    ):
        """
        Forward pass that handles both image and audio embeddings.

        Args:
            input_ids: Input token IDs
            input_embeds: Input embeddings (not used, kept for API compatibility)
            input_image_embeds: Image embeddings
            input_audio_embeds: Audio embeddings
            image_sizes: Sizes of input images
            image_attention_mask: Attention mask for image tokens
            audio_embed_sizes: Sizes of audio embeddings
            audio_attention_mask: Attention mask for audio tokens
            audio_projection_mode: Mode for audio projection ('speech' or 'vision')
            wte: Word token embedder function
        """
        MAX_INPUT_ID = int(1e9)
        assert -MAX_INPUT_ID < self.audio_input_id < self.image_input_id

        # Override cached values if available
        if self.input_image_embeds is not None:
            assert input_image_embeds is None
            input_image_embeds = self.input_image_embeds.copy()
            # Reset cache after use
            self.input_image_embeds = None

        if self.image_sizes is not None:
            assert image_sizes is None
            image_sizes = self.image_sizes

        if self.input_audio_embeds is not None:
            assert input_audio_embeds is None
            input_audio_embeds = self.input_audio_embeds.copy()
            self.input_audio_embeds = None

        if self.audio_embed_sizes is not None:
            assert audio_embed_sizes is None
            audio_embed_sizes = self.audio_embed_sizes.copy()
            self.audio_embed_sizes = None

        if self.image_attention_mask is not None:
            assert image_attention_mask is None
            image_attention_mask = self.image_attention_mask.copy()
            self.image_attention_mask = None

        # Reshape input ids if necessary
        input_shape = input_ids.shape
        input_ids = mx.reshape(input_ids, (-1, input_shape[-1]))

        # Handle backward compatibility for special token IDs
        # Replace compatible token ranges with standard tokens
        new_input_ids = input_ids

        # Create masks for different token ranges
        image_mask = (input_ids >= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[0]) & (
            input_ids <= _COMPATIBLE_IMAGE_SPECIAL_TOKEN_ID_RANGE[1]
        )
        audio_mask = (input_ids >= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[0]) & (
            input_ids <= _COMPATIBLE_AUDIO_SPECIAL_TOKEN_ID_RANGE[1]
        )

        # Update IDs based on masks
        if mx.any(image_mask):
            new_input_ids = mx.where(image_mask, _IMAGE_SPECIAL_TOKEN_ID, new_input_ids)

        if mx.any(audio_mask):
            new_input_ids = mx.where(audio_mask, _AUDIO_SPECIAL_TOKEN_ID, new_input_ids)

        input_ids = new_input_ids

        # Create a mask for image positions
        image_position_mask = input_ids == _IMAGE_SPECIAL_TOKEN_ID
        non_image_position_mask = ~image_position_mask

        # Make sure input_embeds is None
        assert (
            input_embeds is None
        ), "input_embeds should be None for Phi4MMImageAudioEmbedding"

        # Process image and audio embeddings
        if input_image_embeds is not None:
            image_hidden_states = self.image_embed(
                input_ids=input_ids,
                input_embeds=input_image_embeds,
                image_sizes=image_sizes,
                wte=wte,
                image_attention_mask=image_attention_mask,
            )

        if input_audio_embeds is not None:
            audio_hidden_states = self.audio_embed(
                input_ids=input_ids,
                input_embeds=input_audio_embeds,
                audio_embed_sizes=audio_embed_sizes,
                audio_attention_mask=audio_attention_mask,
                wte=wte,
                audio_projection_mode=audio_projection_mode,
            )

        # Merge image and audio hidden states
        # For non-image-audio tokens, we use the token from the audio hidden states
        if input_image_embeds is not None and input_audio_embeds is not None:
            # Create mask for selecting between image and audio hidden states
            image_mask_expanded = mx.expand_dims(image_position_mask, -1)
            non_image_mask_expanded = mx.expand_dims(non_image_position_mask, -1)

            # Convert to proper dtype
            image_mask_expanded = image_mask_expanded.astype(image_hidden_states.dtype)
            non_image_mask_expanded = non_image_mask_expanded.astype(
                audio_hidden_states.dtype
            )

            # Combine using masks
            hidden_states = (image_hidden_states * image_mask_expanded) + (
                audio_hidden_states * non_image_mask_expanded
            )
        elif input_image_embeds is not None:
            hidden_states = image_hidden_states
        elif input_audio_embeds is not None:
            hidden_states = audio_hidden_states
        else:
            # If no special embeddings are provided, just use the word embeddings
            assert wte is not None
            hidden_states = wte(input_ids)

        return hidden_states
