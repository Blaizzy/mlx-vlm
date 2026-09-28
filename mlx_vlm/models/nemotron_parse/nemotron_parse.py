from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(nn.Module):
    """Nemotron-Parse 2.0 model for image-to-text generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)

    @property
    def layers(self):
        return self.language_model.model.decoder.layers

    def make_cache(self):
        """Create cache for encoder-decoder model."""
        return self.language_model.make_cache()

    def _encode_image(self, pixel_values):
        """Encode image to encoder hidden states."""
        return self.vision_tower(pixel_values)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        # Nemotron-Parse is image-to-text only: text input_ids are ignored
        # (the HF reference also ignores them; generation starts from
        # decoder_start_token_id).
        if pixel_values is not None:
            encoder_hidden_states = self._encode_image(pixel_values)
        elif kwargs.get("encoder_outputs") is not None:
            encoder_hidden_states = kwargs["encoder_outputs"]
        elif kwargs.get("cross_attention_states") is not None:
            encoder_hidden_states = kwargs["cross_attention_states"]
        else:
            raise ValueError("You have to specify pixel_values")
        batch_size = encoder_hidden_states.shape[0]
        encoder_seq_len = encoder_hidden_states.shape[1]
        attention_mask = mx.ones((batch_size, encoder_seq_len))

        decoder_start_token_id = self.config.text_config.decoder_start_token_id
        decoder_input_ids = mx.array([[decoder_start_token_id]])
        # The decoder scales embeddings by embed_scale (scale_embedding=True).
        decoder_inputs_embeds = self.language_model.model.shared(decoder_input_ids)

        return InputEmbeddingsFeatures(
            inputs_embeds=encoder_hidden_states,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
        )

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        cache=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        input_embeddings_features = self.get_input_embeddings(
            input_ids, pixel_values, **kwargs
        )
        encoder_hidden_states = input_embeddings_features.inputs_embeds
        # Prefer caller-provided decoder_input_ids over the start-token embedding.
        decoder_inputs_embeds = (
            None
            if decoder_input_ids is not None
            else input_embeddings_features.decoder_inputs_embeds
        )

        outputs = self.language_model(
            inputs=None,
            inputs_embeds=None,
            encoder_outputs=encoder_hidden_states,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            cache=cache,
        )
        return outputs

    @staticmethod
    def sanitize(weights):
        sanitized_weights = {}

        # Map HF checkpoint names to MLX module paths.
        for k, v in weights.items():
            if k.startswith("encoder.model_encoder.radio_model.model.patch_generator."):
                new_k = k.replace(
                    "encoder.model_encoder.radio_model.model.patch_generator.",
                    "vision_tower.",
                )
                if "embedder.weight" in new_k:
                    new_k = new_k.replace("embedder.weight", "patch_embed.weight")
                elif "pos_embed" in new_k:
                    new_k = "vision_tower.pos_embed"
                elif "cls_token.token" in new_k:
                    new_k = "vision_tower.cls_token"
                sanitized_weights[new_k] = v
            elif k.startswith("encoder.model_encoder.radio_model.model.blocks."):
                new_k = k.replace(
                    "encoder.model_encoder.radio_model.model.blocks.",
                    "vision_tower.blocks.",
                )
                sanitized_weights[new_k] = v
            elif k.startswith("encoder."):
                new_k = k.replace("encoder.", "vision_tower.neck.")
                sanitized_weights[new_k] = v
            elif k.startswith("decoder."):
                new_k = k.replace("decoder.", "language_model.model.decoder.")
                sanitized_weights[new_k] = v
            else:
                sanitized_weights[k] = v

        # The checkpoint stores the tied embedding under the decoder; map it to
        # the shared embedding table used by the language model.
        if "language_model.model.decoder.embed_tokens.weight" in sanitized_weights:
            sanitized_weights["language_model.model.shared.weight"] = (
                sanitized_weights.pop(
                    "language_model.model.decoder.embed_tokens.weight"
                )
            )

        # Nemotron-Parse v1.x checkpoints carry a real, untied output head;
        # map it to the language model's lm_head (MLX nn.Linear stores
        # (out_features, in_features), matching the HF (vocab_size, d_model)
        # layout, so no transpose is needed).
        if "lm_head.weight" in sanitized_weights:
            sanitized_weights["language_model.lm_head.weight"] = sanitized_weights.pop(
                "lm_head.weight"
            )

        # Otherwise reconstruct it from the tied embedding.
        if (
            "language_model.lm_head.weight" not in sanitized_weights
            and "language_model.model.shared.weight" in sanitized_weights
        ):
            sanitized_weights["language_model.lm_head.weight"] = sanitized_weights[
                "language_model.model.shared.weight"
            ]

        # Conv layout transposes and unused-buffer drops happen in
        # VisionModel.sanitize, which the loader pipeline applies separately.

        return sanitized_weights
