import math
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# Special token id for audio
_AUDIO_SPECIAL_TOKEN_ID = 200011  # '<|endoftext11|>'


class ConformerConvModule(nn.Module):
    """Conformer convolution module implementation."""

    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2

        self.net = [
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2),
            nn.GLU(axis=-1),
            nn.Conv1d(
                inner_dim,
                inner_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
                groups=inner_dim,
            ),
            nn.BatchNorm(inner_dim),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        ]

    def __call__(self, x):
        # Expected input: (batch_size, seq_len, channels)

        # Apply layer norm
        x = self.net[0](x)

        # Linear projection
        x = self.net[1](x)

        # GLU activation
        x = self.net[2](x)

        # Transpose for depthwise conv: (batch_size, channels, seq_len)
        x = mx.transpose(x, (0, 2, 1))

        # Depthwise convolution
        x = self.net[3](x)

        # Batch normalization
        x = self.net[4](x)

        # SiLU activation
        x = self.net[5](x)

        # Pointwise convolution
        x = self.net[6](x)

        # Transpose back: (batch_size, seq_len, channels)
        x = mx.transpose(x, (0, 2, 1))

        # Dropout
        if self.net[7].p > 0:
            x = self.net[7](x)

        return x


class FeedForward(nn.Module):
    """Feed Forward module for Conformer."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.net = [
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]

    def __call__(self, x):
        # Layer normalization
        x = self.net[0](x)

        # Linear projection to hidden dim
        x = self.net[1](x)

        # SiLU activation
        x = self.net[2](x)

        # Dropout
        if self.net[3].p > 0:
            x = self.net[3](x)

        # Linear projection back to original dim
        x = self.net[4](x)

        # Dropout
        if self.net[5].p > 0:
            x = self.net[5](x)

        return x


class ConformerAttention(nn.Module):
    """Multi-headed attention module for Conformer."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv_proj = nn.Linear(dim, inner_dim * 3, bias=True)
        self.output_proj = nn.Linear(inner_dim, dim, bias=True)
        self.dropout = dropout

    def __call__(self, x, mask=None):
        # Apply layer normalization
        x = self.norm(x)

        # Project to query, key, value
        qkv = self.qkv_proj(x)

        # Split into q, k, v
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention
        batch_size, seq_len, _ = x.shape
        q = mx.reshape(q, (batch_size, seq_len, self.heads, -1))
        k = mx.reshape(k, (batch_size, seq_len, self.heads, -1))
        v = mx.reshape(v, (batch_size, seq_len, self.heads, -1))

        # Transpose to (batch_size, heads, seq_len, dim_head)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Compute attention scores
        attention_scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting with attention scores
            mask = mx.expand_dims(mx.expand_dims(mask, 1), 1)
            attention_scores = mx.where(mask, attention_scores, mx.array(float("-inf")))

        # Apply softmax to get attention weights
        attention_weights = nn.softmax(attention_scores, axis=-1)

        # Apply dropout to attention weights during training
        if self.dropout > 0:
            attention_weights = nn.Dropout(self.dropout)(attention_weights)

        # Apply attention weights to values
        context = mx.matmul(attention_weights, v)

        # Transpose and reshape back
        context = mx.transpose(context, (0, 2, 1, 3))
        context = mx.reshape(context, (batch_size, seq_len, -1))

        # Final projection
        output = self.output_proj(context)

        return output


class ConformerBlock(nn.Module):
    """A single Conformer block."""

    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        dropout=0.0,
    ):
        super().__init__()

        self.ff1 = FeedForward(dim, dim * ff_mult, dropout)
        self.attn = ConformerAttention(dim, heads, dim_head, dropout)
        self.conv = ConformerConvModule(
            dim, conv_expansion_factor, conv_kernel_size, dropout
        )
        self.ff2 = FeedForward(dim, dim * ff_mult, dropout)

        self.norm = nn.LayerNorm(dim)

    def __call__(self, x, mask=None):
        # First feed-forward module (with 0.5 scaling)
        x = x + 0.5 * self.ff1(x)

        # Self-attention module
        x = x + self.attn(x, mask)

        # Convolution module
        x = x + self.conv(x)

        # Second feed-forward module (with 0.5 scaling)
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.norm(x)

        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio processing."""

    def __init__(
        self,
        input_size=80,  # Mel frequency bins
        attention_dim=512,
        attention_heads=8,
        linear_units=2048,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        input_layer="conv2d",
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.attention_dim = attention_dim

        # Input layer
        if input_layer == "conv2d":
            self.embed = [
                nn.Conv2d(1, attention_dim // 4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(
                    attention_dim // 4,
                    attention_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    attention_dim // 2,
                    attention_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
            ]
        else:
            self.embed = [nn.Linear(input_size, attention_dim)]

        # Positional encoding - using sinusoidal positional embedding
        # In a complete implementation, we would use a proper positional encoding here

        # Conformer blocks
        self.blocks = [
            ConformerBlock(
                dim=attention_dim,
                dim_head=attention_dim // attention_heads,
                heads=attention_heads,
                ff_mult=linear_units // attention_dim,
                dropout=dropout_rate,
            )
            for _ in range(num_blocks)
        ]

        self.norm = nn.LayerNorm(attention_dim)

    def post_init(self, init_model=None):
        """Initialize model weights from a pretrained model."""
        # In a complete implementation, this would load weights from a checkpoint
        pass

    def __call__(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, time_length, feat_dim)
            mask: Mask tensor for valid data (batch_size, time_length)

        Returns:
            x: Encoded features (batch_size, new_length, attention_dim)
            mask: Mask for the encoded features
        """

        # Handle input layer based on its type
        if len(self.embed) > 1:  # conv2d case
            # Add channel dimension: (batch_size, time_length, feat_dim) -> (batch_size, 1, time_length, feat_dim)
            x = mx.expand_dims(x, axis=1)

            # Apply convolutional layers
            for layer in self.embed[:2]:
                x = layer(x)

            # For each conv layer, the sequence length is reduced
            if mask is not None:
                # Downsample the mask to match the reduced sequence length
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[2:4]:
                x = layer(x)

            # Downsample mask again
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            for layer in self.embed[4:]:
                x = layer(x)

            # Downsample mask once more
            if mask is not None:
                mask = mx.reshape(mask, (mask.shape[0], -1, 2))[:, :, 0]

            # Rearrange from (batch_size, channels, time, freq) -> (batch_size, time, channels*freq)
            batch_size, channels, time, freq = x.shape
            x = mx.transpose(x, (0, 2, 1, 3))
            x = mx.reshape(x, (batch_size, time, channels * freq))

        else:  # Linear case
            x = self.embed[0](x)

        # Apply Conformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.norm(x)

        return x, mask


class AudioModel(nn.Module):
    """Audio embedding."""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Get hidden size for text LM
        hidden_size = config.n_embd if hasattr(config, "n_embd") else config.hidden_size

        # Dropout setting
        self.drop = None
        if hasattr(config, "embd_pdrop") or hasattr(config, "embed_pdrop"):
            embd_drop = (
                config.embd_pdrop
                if hasattr(config, "embd_pdrop")
                else config.embed_pdrop
            )
            self.embd_drop = embd_drop  # Store dropout rate

        # Configure audio processor
        if (
            isinstance(config.audio_processor, dict)
            and config.audio_processor.get("name", None) == "cascades"
        ):
            encoder_config = config.audio_processor.get("config", None)
            assert encoder_config is not None

            # Create encoder
            self.encoder = ConformerEncoder(**encoder_config)

            # Initialize placeholder for post_init
            self.encoder.post_init({})

            audio_dim_out = encoder_config["attention_dim"]
            n_mels = encoder_config["input_size"]
        else:
            raise NotImplementedError("Audio processor not implemented")

        self.audio_dim_out = audio_dim_out
        self.audio_dim_in = n_mels

        # Configuration
        self.freeze_audio_processor = kwargs.get("freeze_audio_processor", False)
        self.downsample_rate = kwargs.get("downsample_rate", 1)

        # Projection layer
        projection_cls = kwargs.get("projection_cls", "linear")
        if projection_cls == "linear":
            self.audio_projection = nn.Linear(audio_dim_out, hidden_size)
        elif projection_cls == "mlp":
            # Follow llava-v1.5's implementation
            dim_projection = hidden_size
            depth = 2
            self.linear_downsample_rate = self.downsample_rate

            # Create projection for speech mode
            layers_for_speech = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_speech.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_speech = layers_for_speech

            # Create projection for vision mode
            layers_for_vision = [
                nn.Linear(audio_dim_out * self.linear_downsample_rate, dim_projection)
            ]
            for _ in range(1, depth):
                layers_for_vision.extend(
                    [nn.GELU(), nn.Linear(dim_projection, dim_projection)]
                )

            audio_projection_for_vision = layers_for_vision

            # Store as a dictionary
            self.audio_projection = {
                "speech": audio_projection_for_speech,
                "vision": audio_projection_for_vision,
            }
        else:
            raise NotImplementedError(f"projection_cls = {projection_cls}")

        self.vocab_size = config.vocab_size
        self.input_embeds = None
        self.audio_embed_sizes = None

    def post_init(self, audio_config):
        """Initialize the audio encoder with pretrained weights."""
        if audio_config.get("name", None) == "cascades":
            init_model_config = audio_config.get("init_model", {})
            self.encoder.post_init(init_model_config)

            # Remove init_model to save memory
            if "init_model" in audio_config:
                audio_config.pop("init_model")

    def set_audio_embeds(self, input_embeds):
        self.input_embeds = input_embeds

    def set_audio_embed_sizes(self, audio_embed_sizes):
        self.audio_embed_sizes = audio_embed_sizes

    def get_audio_features(
        self, input_embeds, audio_attention_mask, audio_projection_mode="speech"
    ):
        """Process audio inputs through the encoder and projection layers."""

        # Apply encoder with or without gradient based on freeze setting
        if self.freeze_audio_processor:
            # In MLX, we would implement a mechanism to stop gradient flow
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)
        else:
            audio_features, masks = self.encoder(input_embeds, audio_attention_mask)

        # Apply projection based on its type
        if isinstance(self.audio_projection, dict):
            # Sequential projection for the specified mode
            projection_layers = self.audio_projection[audio_projection_mode]

            # Apply the layers in sequence
            audio_set_tensor = audio_features
            for layer in projection_layers:
                audio_set_tensor = layer(audio_set_tensor)
        else:
            # Single linear projection
            audio_set_tensor = self.audio_projection(audio_features)

        return audio_set_tensor

    def __call__(
        self,
        input_ids,
        input_embeds,
        audio_embed_sizes=None,
        audio_attention_mask=None,
        audio_projection_mode="speech",
        **kwargs,
    ):
        """
        Forward pass for audio embeddings.

        Args:
            input_ids: Input text ids (B, U)
            input_embeds: Audio features (B, T, D)  B: num audios in a sequence
        """
        # Use cached inputs if available
        if self.input_embeds is not None:
            input_embeds = self.input_embeds.copy()
            self.input_embeds = None

        if self.audio_embed_sizes is not None:
            audio_embed_sizes = self.audio_embed_sizes.copy()
            self.audio_embed_sizes = None

        # Reshape input_ids if needed
        input_shape = input_ids.shape
        input_ids = mx.reshape(input_ids, (-1, input_shape[-1]))

        # Find positions of audio token IDs
        positions = mx.nonzero(input_ids == _AUDIO_SPECIAL_TOKEN_ID)

        # Determine target device and dtype from projection layer
        if isinstance(self.audio_projection, dict):
            target_dtype = mx.float32  # Default dtype
        else:
            target_dtype = mx.float32

        # Convert input_embeds to target dtype if available
        if input_embeds is not None:
            input_embeds = input_embeds.astype(target_dtype)

        # Process audio if audio tokens are present
        if len(positions) > 0:
            audio_set_tensor = self.get_audio_features(
                input_embeds, audio_attention_mask, audio_projection_mode
            )
        else:
            # Create dummy audio tensor for training
            if True:  # Equivalent to self.training in PyTorch
                audio_embeds = mx.zeros((1, 500, self.audio_dim_in), dtype=target_dtype)
                audio_attention_mask = mx.ones((1, 500), dtype=mx.int32)
                audio_set_tensor = self.get_audio_features(
                    audio_embeds, audio_attention_mask, audio_projection_mode
                )

        # Get token embeddings
        hidden_states = kwargs["wte"](input_ids)

        if len(positions) > 0:
            # Validate that we have correct number of positions
            assert audio_embed_sizes.sum().item() == len(
                positions
            ), f"Number of encoder outputs ({audio_embed_sizes.sum().item()}) must match number of audio tokens ({len(positions)})"

            # Create a list of audio features based on sizes
            merged_audio_set_tensor = []
            start_idx = 0
            for i in range(len(audio_embed_sizes)):
                size = audio_embed_sizes[i]
                merged_audio_set_tensor.append(audio_set_tensor[i, :size])
                start_idx += size

            # Concatenate all features
            merged_audio_set_tensor = mx.concatenate(merged_audio_set_tensor, axis=0)
            merged_audio_set_tensor = merged_audio_set_tensor.astype(
                hidden_states.dtype
            )

            # Create a new hidden_states with audio embeddings inserted
            for i, pos in enumerate(positions):
                batch_idx, seq_idx = pos
                # Update the tensor at the specified position
                hidden_states = mx.indexed_update(
                    hidden_states,
                    ((batch_idx, seq_idx),),
                    mx.expand_dims(merged_audio_set_tensor[i], axis=0),
                )
        else:
            # For training with no audio tokens, add a small contribution to maintain gradient flow
            if True:  # Equivalent to self.training
                hidden_states = hidden_states + (0 * audio_set_tensor[:, 0]).sum()

        # Apply dropout if configured
        if self.drop is not None:
            hidden_states = nn.Dropout(self.embd_drop)(hidden_states)

        return hidden_states
