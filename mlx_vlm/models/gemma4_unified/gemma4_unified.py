from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from ..gemma4.gemma4 import MultimodalEmbedder, masked_scatter
from ..gemma4.language import LanguageModel
from .config import ModelConfig, VisionConfig


def _compact_prefix_rows(features: mx.array, valid_mask: mx.array) -> mx.array:
    rows = []
    for batch_idx, row in enumerate(valid_mask.tolist()):
        length = sum(bool(v) for v in row)
        if length:
            rows.append(features[batch_idx, :length])
    if not rows:
        return features.reshape(-1, features.shape[-1])[:0]
    return mx.concatenate(rows, axis=0)


class VisionEmbedder(nn.Module):
    """Encoder-free Gemma 4 unified vision embedder."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.model_type = config.model_type
        patch_dim = config.model_patch_size * config.model_patch_size * 3
        self.patch_dim = patch_dim
        self.patch_ln1 = nn.LayerNorm(patch_dim)
        self.patch_dense = nn.Linear(patch_dim, config.mm_embed_dim)
        self.patch_ln2 = nn.LayerNorm(config.mm_embed_dim)
        self.pos_embedding = mx.zeros((config.mm_posemb_size, 2, config.mm_embed_dim))
        self.pos_norm = nn.LayerNorm(config.mm_embed_dim)

    def __call__(
        self,
        pixel_values: mx.array,
        image_position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if pixel_values.ndim == 4 and pixel_values.shape[-1] == self.patch_dim:
            pixel_values = pixel_values.reshape(
                pixel_values.shape[0], -1, self.patch_dim
            )
        hidden_states = self.patch_ln1(pixel_values)
        hidden_states = self.patch_dense(hidden_states)
        hidden_states = self.patch_ln2(hidden_states)

        if image_position_ids is not None:
            clamped = mx.maximum(image_position_ids, 0).astype(mx.int32)
            valid = (image_position_ids != -1).astype(hidden_states.dtype)
            x_pos = self.pos_embedding[clamped[..., 0], 0]
            y_pos = self.pos_embedding[clamped[..., 1], 1]
            hidden_states = hidden_states + (
                x_pos * mx.expand_dims(valid[..., 0], -1)
                + y_pos * mx.expand_dims(valid[..., 1], -1)
            )

        return self.pos_norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self._base_no_chunked_prefill = (
            getattr(config.text_config, "use_bidirectional_attention", None) == "vision"
        )
        self.no_chunked_prefill = self._base_no_chunked_prefill

        self.language_model = LanguageModel(config.text_config)
        self.language_model.no_chunked_prefill = self.no_chunked_prefill
        self.vocab_size = config.text_config.vocab_size

        if config.vision_config is not None:
            self.vision_embedder = VisionEmbedder(config.vision_config)
            self.embed_vision = MultimodalEmbedder(
                embedding_dim=config.vision_config.output_proj_dims,
                text_hidden_size=config.text_config.hidden_size,
                eps=config.vision_config.rms_norm_eps,
            )
        else:
            self.vision_embedder = None
            self.embed_vision = None

        if config.audio_config is not None:
            self.embed_audio = MultimodalEmbedder(
                embedding_dim=config.audio_config.output_proj_dims,
                text_hidden_size=config.text_config.hidden_size,
                eps=config.audio_config.rms_norm_eps,
            )
        else:
            self.embed_audio = None

    def _should_disable_chunked_prefill(self, input_ids=None, **kwargs) -> bool:
        """Disable chunking only when a prompt needs vision bidirectional masks."""
        if not self._base_no_chunked_prefill:
            return False

        token_types = kwargs.get("mm_token_type_ids", None)
        if token_types is None:
            token_types = kwargs.get("token_type_ids", None)

        if token_types is not None:
            has_visual = int(mx.sum((token_types == 1) | (token_types == 2)).item()) > 0
            has_audio = int(mx.sum(token_types == 3).item()) > 0
            return has_visual and not has_audio

        if input_ids is None:
            return True

        video_token_id = getattr(self.config, "video_token_id", None)
        image_token_id = getattr(self.config, "image_token_id", None)
        audio_token_id = getattr(self.config, "audio_token_id", None)
        has_visual = (
            image_token_id is not None
            and int(mx.sum(input_ids == image_token_id).item()) > 0
        )
        if video_token_id is not None:
            has_visual = (
                has_visual or int(mx.sum(input_ids == video_token_id).item()) > 0
            )
        has_audio = (
            audio_token_id is not None
            and int(mx.sum(input_ids == audio_token_id).item()) > 0
        )
        return has_visual and not has_audio

    def _update_chunked_prefill_mode(self, input_ids=None, **kwargs):
        self.no_chunked_prefill = self._should_disable_chunked_prefill(
            input_ids, **kwargs
        )
        self.language_model.no_chunked_prefill = self.no_chunked_prefill

    def chunked_prefill_policy(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        prompt_cache=None,
        draft_model=None,
        draft_kind=None,
        prefill_kwargs=None,
    ) -> bool:
        del inputs_embeds, prompt_cache
        prefill_kwargs = prefill_kwargs or {}
        if self._should_disable_chunked_prefill(input_ids, **prefill_kwargs):
            return False

        if draft_model is not None:
            return (
                draft_kind == "mtp"
                and bool(prefill_kwargs.get("return_hidden", False))
                and bool(prefill_kwargs.get("return_shared_kv", False))
            )

        return True

    def _placeholder_mask(
        self,
        input_ids: mx.array,
        token_id: Optional[int],
        inputs_embeds: mx.array,
    ) -> Optional[mx.array]:
        if input_ids is None or token_id is None:
            return None
        return mx.broadcast_to(
            mx.expand_dims(input_ids == token_id, -1), inputs_embeds.shape
        )

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if self.vision_embedder is None or self.embed_vision is None:
            raise ValueError("Vision inputs were provided, but vision_config is None.")
        embedded = self.vision_embedder(pixel_values, image_position_ids)
        projected = self.embed_vision(embedded)
        if image_position_ids is None:
            return projected.reshape(-1, projected.shape[-1])
        padding_mask = mx.all(image_position_ids == -1, axis=-1)
        return _compact_prefix_rows(projected, ~padding_mask)

    def get_video_features(
        self,
        pixel_values_videos: mx.array,
        video_position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        if pixel_values_videos.ndim == 4:
            n_videos, n_frames, n_patches, patch_dim = pixel_values_videos.shape
            pixel_values_videos = pixel_values_videos.reshape(
                n_videos * n_frames, n_patches, patch_dim
            )
            if video_position_ids is not None:
                video_position_ids = video_position_ids.reshape(
                    n_videos * n_frames, n_patches, 2
                )
        return self.get_image_features(pixel_values_videos, video_position_ids)

    def get_audio_features(
        self,
        input_features: mx.array,
        input_features_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self.embed_audio is None:
            raise ValueError("Audio inputs were provided, but audio_config is None.")
        projected = self.embed_audio(input_features)
        if input_features_mask is None:
            return projected.reshape(-1, projected.shape[-1])
        return _compact_prefix_rows(projected, input_features_mask.astype(mx.bool_))

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        pixel_values_videos: Optional[mx.array] = None,
        audio_features: Optional[mx.array] = None,
        audio_mask: Optional[mx.array] = None,
        input_features: Optional[mx.array] = None,
        input_features_mask: Optional[mx.array] = None,
        image_position_ids: Optional[mx.array] = None,
        video_position_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        if input_features is not None and audio_features is None:
            audio_features = input_features
        if input_features_mask is not None and audio_mask is None:
            audio_mask = input_features_mask

        self._update_chunked_prefill_mode(input_ids, **kwargs)

        if inputs_embeds is None:
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * self.language_model.model.embed_scale

        video_token_id = getattr(self.config, "video_token_id", None)

        per_layer_inputs = None
        if (
            input_ids is not None
            and self.language_model.model.hidden_size_per_layer_input
        ):
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

        def _scatter(source, token_id, encode, mask, cached_kw=None, key_kw=None):
            if source is None or token_id is None or mask is None:
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
            return masked_scatter(inputs_embeds, mask, features)

        image_mask = self._placeholder_mask(
            input_ids, self.config.image_token_id, inputs_embeds
        )
        video_mask = self._placeholder_mask(input_ids, video_token_id, inputs_embeds)
        audio_token_mask = self._placeholder_mask(
            input_ids, self.config.audio_token_id, inputs_embeds
        )

        inputs_embeds = _scatter(
            pixel_values,
            self.config.image_token_id,
            lambda pixels: self.get_image_features(pixels, image_position_ids),
            image_mask,
            "cached_image_features",
            "_image_key",
        )
        inputs_embeds = _scatter(
            pixel_values_videos,
            video_token_id,
            lambda pixels: self.get_video_features(pixels, video_position_ids),
            video_mask,
            "cached_video_features",
            "_video_key",
        )
        inputs_embeds = _scatter(
            audio_features,
            self.config.audio_token_id,
            lambda features: self.get_audio_features(features, audio_mask),
            audio_token_mask,
        )

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds, per_layer_inputs=per_layer_inputs
        )

    def encode_image(
        self,
        pixel_values: mx.array,
        image_position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        return self.get_image_features(pixel_values, image_position_ids)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
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

        return self.language_model(
            None,
            cache=cache,
            inputs_embeds=input_embeddings_features.inputs_embeds,
            mask=mask,
            per_layer_inputs=input_embeddings_features.per_layer_inputs,
            **lm_kwargs,
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k or "rotary_emb" in k:
                continue
            if k == "lm_head.weight":
                continue
            if self.embed_audio is None and "embed_audio" in k:
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

    def make_cache(self):
        return self.language_model.make_cache()


VisionModel = VisionEmbedder
