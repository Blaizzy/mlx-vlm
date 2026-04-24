from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        spatial_shapes = kwargs.get("spatial_shapes", None)

        if pixel_values is None:
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        dtype = self.vision_tower.embeddings.patch_embedding.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get text embeddings
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        # Get vision features
        hidden_states = self.vision_tower(pixel_values, spatial_shapes)

        # Merge vision features into text embeddings at image token positions
        final_inputs_embeds = self.merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            hidden_states,
            inputs_embeds,
            input_ids,
        )

        return InputEmbeddingsFeatures(inputs_embeds=final_inputs_embeds)

    @staticmethod
    def merge_input_ids_with_image_features(
        image_token_id,
        video_token_id,
        image_features,
        inputs_embeds,
        input_ids,
    ):
        """Merge image features into input embeddings at image token positions."""
        # Find positions of image tokens
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        batch_size, seq_len = input_ids.shape

        batch_outputs = []
        feature_start_idx = 0

        for batch_idx in range(batch_size):
            image_mask = image_positions[batch_idx]
            num_positions = mx.sum(image_mask).item()

            if num_positions > 0:
                batch_features = image_features[
                    feature_start_idx : feature_start_idx + num_positions
                ]

                if batch_features.shape[0] != num_positions:
                    raise ValueError(
                        f"Number of image token positions ({num_positions}) does not match "
                        f"number of image features ({batch_features.shape[0]}) for batch {batch_idx}"
                    )

                cumsum = mx.cumsum(image_mask.astype(mx.int32))
                feature_indices = mx.where(image_mask, cumsum - 1, 0)
                gathered_features = batch_features[feature_indices]

                image_mask_expanded = mx.expand_dims(image_mask, axis=-1)
                batch_output = mx.where(
                    image_mask_expanded, gathered_features, inputs_embeds[batch_idx]
                )

                feature_start_idx += num_positions
            else:
                batch_output = inputs_embeds[batch_idx]

            batch_outputs.append(batch_output)

        return mx.stack(batch_outputs, axis=0)

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
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )

        logits = self.language_model(
            input_ids,
            input_embeddings_features.inputs_embeds,
            mask=mask,
            cache=cache,
        )

        return logits

    def sanitize(self, weights):
        def transform_key(key):
            # Map HF weight keys to our model structure
            # siglip2.vision_model.xxx -> vision_tower.xxx
            if key.startswith("siglip2.vision_model."):
                key = key.replace("siglip2.vision_model.", "vision_tower.")
            elif key.startswith("siglip2."):
                key = key.replace("siglip2.", "vision_tower.")

            # merger.xxx stays as merger.xxx but needs to go under vision_tower
            if key.startswith("merger."):
                key = "vision_tower." + key

            # model.xxx -> language_model.model.xxx
            if key.startswith("model."):
                key = key.replace("model.", "language_model.model.", 1)

            # lm_head.xxx -> language_model.lm_head.xxx
            if key.startswith("lm_head."):
                key = key.replace("lm_head.", "language_model.lm_head.")

            return key

        sanitized = {}
        for k, v in weights.items():
            new_key = transform_key(k)
            # Skip position_ids and position_embedding (RoPE used instead)
            if "position_ids" in new_key or "position_embedding" in new_key:
                continue
            sanitized[new_key] = v

        # Handle tie_word_embeddings
        if self.config.text_config.tie_word_embeddings:
            sanitized.pop("language_model.lm_head.weight", None)

        # Split kv_b_proj into per-head embed_q (k) and unembed_out (v)
        tcfg = self.config.text_config
        num_heads = tcfg.num_attention_heads
        qk_nope = tcfg.qk_nope_head_dim
        v_head = tcfg.v_head_dim
        head_dim = qk_nope + v_head
        for layer_idx in range(tcfg.num_hidden_layers):
            prefix = f"language_model.model.layers.{layer_idx}.self_attn"
            kv_b_key = f"{prefix}.kv_b_proj.weight"
            if kv_b_key not in sanitized:
                continue
            w = sanitized.pop(kv_b_key)
            w = w.reshape(num_heads, head_dim, -1)
            wk = mx.contiguous(w[:, :qk_nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(w[:, qk_nope:, :])
            sanitized[f"{prefix}.embed_q.weight"] = wk
            sanitized[f"{prefix}.unembed_out.weight"] = wv

        return sanitized
