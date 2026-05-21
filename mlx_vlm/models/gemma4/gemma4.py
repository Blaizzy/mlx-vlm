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
        self.embedding_pre_projection_norm = RMSNormNoScale(embedding_dim, eps=eps)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


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
        pixel_values_videos: Optional[mx.array] = None,
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
        inputs_embeds = inputs_embeds * self.language_model.model.embed_scale

        video_token_id = getattr(self.config, "video_token_id", None)

        per_layer_inputs = None
        if self.language_model.model.hidden_size_per_layer_input:
            image_mask_ids = input_ids == self.config.image_token_id
            audio_mask_ids = input_ids == self.config.audio_token_id
            mm_mask = image_mask_ids | audio_mask_ids
            if video_token_id is not None:
                mm_mask = mm_mask | (input_ids == video_token_id)
            text_mask = ~mm_mask
            per_layer_inputs_tokens = mx.where(
                text_mask, input_ids, mx.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.model.get_per_layer_inputs(
                per_layer_inputs_tokens
            )

        def _scatter(source, token_id, encode, cached_kw=None, key_kw=None):
            if source is None or token_id is None:
                return inputs_embeds
            vision_cache = kwargs.get("vision_cache", None) if key_kw else None
            cache_key = kwargs.get(key_kw) if key_kw else None
            cached = kwargs.get(cached_kw, None) if cached_kw else None
            if cached is None and vision_cache is not None:
                cached = vision_cache.get(cache_key)
            if cached is not None:
                features = cached.astype(inputs_embeds.dtype)
            else:
                features = encode(source).astype(inputs_embeds.dtype)
                if vision_cache is not None and cache_key is not None:
                    mx.eval(features)
                    vision_cache.put(cache_key, features)

            mask_expanded = mx.broadcast_to(
                mx.expand_dims(input_ids == token_id, -1), inputs_embeds.shape
            )
            return masked_scatter(inputs_embeds, mask_expanded, features)

        def _encode_vision(pixels):
            return self.embed_vision(self.vision_tower(pixels))

        inputs_embeds = _scatter(
            pixel_values,
            self.config.image_token_id,
            _encode_vision,
            "cached_image_features",
            "_image_key",
        )
        inputs_embeds = _scatter(
            pixel_values_videos,
            video_token_id,
            _encode_vision,
            "cached_video_features",
            "_video_key",
        )

        if self.audio_tower is not None and self.embed_audio is not None:

            def _encode_audio(feat):
                mask = (
                    audio_mask
                    if audio_mask is not None
                    else mx.zeros(feat.shape[:2], dtype=mx.bool_)
                )
                enc, _ = self.audio_tower(feat, mask)
                return self.embed_audio(enc)

            inputs_embeds = _scatter(
                audio_features,
                self.config.audio_token_id,
                _encode_audio,
            )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
        )

    def encode_image(self, pixel_values: mx.array) -> mx.array:
        """Encode pixel_values through vision_tower + embed_vision.

        Returns projected image features suitable for passing as
        cached_image_features to get_input_embeddings.
        """
        image_features = self.vision_tower(pixel_values)
        image_features = self.embed_vision(image_features)
        return image_features

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

        # Forward speculative-decoding hooks straight through to the LM.
        lm_kwargs = {
            k: kwargs[k]
            for k in (
                "capture_layer_ids",
                "hidden_sink",
                "shared_kv_sink",
                "return_hidden",
                "return_shared_kv",
            )
            if k in kwargs
        }

        logits = self.language_model(
            input_ids=None,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            per_layer_inputs=input_embeddings_features.per_layer_inputs,
            **lm_kwargs,
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

            if new_key.startswith("language_model.") and not new_key.startswith(
                "language_model.model."
            ):
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
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def layers(self):
        return self.language_model.model.layers
