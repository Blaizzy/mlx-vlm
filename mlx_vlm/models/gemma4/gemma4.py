from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .audio import AudioEncoder
from .config import ModelConfig
from .language import LanguageModel, RMSNormNoScale
from .vision import VisionModel


def masked_scatter(input_tensor, mask, source):
    mask_flat = mask.flatten().astype(mx.int32)
    indices = mx.cumsum(mask_flat) - 1
    aligned = source.flatten()[indices % source.size]
    return mx.where(mask_flat, aligned, input_tensor.flatten()).reshape(
        input_tensor.shape
    )


class MultimodalEmbedder(nn.Module):
    """Projects soft tokens from vision/audio into language model space."""

    def __init__(self, embedding_dim: int, text_hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.embedding_projection = nn.Linear(
            embedding_dim, text_hidden_size, bias=False
        )
        self.embedding_post_projection_norm = RMSNormNoScale(text_hidden_size, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        proj = self.embedding_projection(inputs_embeds)
        return self.embedding_post_projection_norm(proj)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        # Text
        self.language_model = LanguageModel(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        # Vision
        self.vision_tower = VisionModel(config.vision_config)
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            eps=config.vision_config.rms_norm_eps,
        )

        # Audio
        if config.audio_config is not None:
            self.audio_tower = AudioEncoder(config.audio_config)
            audio_output_dim = (
                config.audio_config.output_proj_dims or config.audio_config.hidden_size
            )
            self.embed_audio = MultimodalEmbedder(
                embedding_dim=audio_output_dim,
                text_hidden_size=config.text_config.hidden_size,
                eps=config.audio_config.rms_norm_eps,
            )
        else:
            self.audio_tower = None
            self.embed_audio = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_mask: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if input_features is not None and audio_features is None:
            audio_features = input_features
        if input_features_mask is not None and audio_mask is None:
            audio_mask = ~input_features_mask.astype(mx.bool_)
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        per_layer_inputs = None
        if self.language_model.model.hidden_size_per_layer_input:
            image_mask_ids = input_ids == self.config.image_token_id
            audio_mask_ids = input_ids == self.config.audio_token_id
            text_mask = ~(image_mask_ids | audio_mask_ids)
            per_layer_inputs_tokens = mx.where(
                text_mask, input_ids, mx.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                per_layer_inputs_tokens
            )

        if pixel_values is not None:
            image_features = self.vision_tower(pixel_values)
            image_features = self.embed_vision(image_features)
            image_features = image_features.astype(inputs_embeds.dtype)

            image_mask = input_ids == self.config.image_token_id
            image_mask_expanded = mx.expand_dims(image_mask, -1)
            image_mask_expanded = mx.broadcast_to(
                image_mask_expanded, inputs_embeds.shape
            )

            inputs_embeds = masked_scatter(
                inputs_embeds, image_mask_expanded, image_features
            )

        # Scatter audio features
        if (
            audio_features is not None
            and self.audio_tower is not None
            and self.embed_audio is not None
        ):
            if audio_mask is None:
                audio_mask = mx.zeros(audio_features.shape[:2], dtype=mx.bool_)
            audio_encodings, _ = self.audio_tower(audio_features, audio_mask)
            audio_encodings = self.embed_audio(audio_encodings)
            audio_encodings = audio_encodings.astype(inputs_embeds.dtype)

            audio_token_mask = input_ids == self.config.audio_token_id
            audio_mask_expanded = mx.expand_dims(audio_token_mask, -1)
            audio_mask_expanded = mx.broadcast_to(
                audio_mask_expanded, inputs_embeds.shape
            )

            inputs_embeds = masked_scatter(
                inputs_embeds, audio_mask_expanded, audio_encodings
            )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        logits = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            per_layer_inputs=input_embeddings_features.per_layer_inputs,
        )
        return logits

    def sanitize(self, weights):
        use_clipped = getattr(self.config.vision_config, "use_clipped_linears", False)
        sanitized = {}
        for k, v in weights.items():
            # Skip clipping parameters when not used
            if any(
                s in k for s in ["input_max", "input_min", "output_max", "output_min"]
            ):
                if "vision_tower" in k and not use_clipped:
                    continue
                if "vision_tower" not in k and "audio_tower" not in k:
                    continue
            if "rotary_emb.inv_freq" in k or "rotary_emb" in k:
                continue
            if self.audio_tower is None and ("audio_tower" in k or "embed_audio" in k):
                continue

            if k.startswith("model."):
                new_key = k[len("model.") :]
            else:
                new_key = k

            if new_key.startswith("language_model."):
                rest = new_key[len("language_model.") :]
                new_key = "language_model.model." + rest

            # Conv2d: PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in]
            if (
                "subsample_conv_projection" in new_key
                and "conv.weight" in new_key
                and v.ndim == 4
            ):
                v = v.transpose(0, 2, 3, 1)
            # Conv1d: PyTorch [out, in, kW] -> MLX [out, kW, in]
            if "depthwise_conv1d.weight" in new_key and v.ndim == 3:
                v = v.transpose(0, 2, 1)

            # MoE: experts.down_proj -> experts.switch_glu.down_proj.weight
            # experts.gate_up_proj -> split into switch_glu.gate_proj + switch_glu.up_proj
            if new_key.endswith(".experts.down_proj"):
                new_key = new_key.replace(
                    ".experts.down_proj", ".experts.switch_glu.down_proj.weight"
                )
            if new_key.endswith(".experts.gate_up_proj"):
                gate_key = new_key.replace(
                    ".experts.gate_up_proj",
                    ".experts.switch_glu.gate_proj.weight",
                )
                up_key = new_key.replace(
                    ".experts.gate_up_proj",
                    ".experts.switch_glu.up_proj.weight",
                )

                v = v.swapaxes(-1, -2)
                mid_dim = v.shape[-1] // 2
                sanitized[gate_key] = v[..., :mid_dim].swapaxes(-1, -2)
                sanitized[up_key] = v[..., mid_dim:].swapaxes(-1, -2)
                continue

            sanitized[new_key] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.model.layers
