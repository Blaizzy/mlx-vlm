from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from ..pixtral.language import LanguageModel
from .audio import AudioModel
from .config import ModelConfig


class VisionModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def sanitize(self, weights: Dict[str, mx.array]):
        return weights


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.audio_tower = AudioModel(config.audio_config)
        self.multi_modal_projector = VoxtralMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

    @property
    def layers(self):
        return self.language_model.model.layers

    def _get_audio_features(self, input_features: mx.array) -> mx.array:
        audio_hidden_states = self.audio_tower(input_features)
        hidden_size = self.config.audio_config.hidden_size
        group_size = self.config.audio_config.intermediate_size // hidden_size
        bsz, seq_len, _ = audio_hidden_states.shape
        if seq_len % group_size != 0:
            raise ValueError("Audio sequence length must be divisible by group size")
        audio_hidden_states = audio_hidden_states.reshape(
            (bsz, seq_len // group_size, hidden_size * group_size)
        )
        audio_hidden_states = audio_hidden_states.reshape(
            (-1, self.config.audio_config.intermediate_size)
        )
        return self.multi_modal_projector(audio_hidden_states)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        input_features: Optional[mx.array] = None,
        **kwargs,
    ):
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if input_features is not None:
            audio_embeds = self._get_audio_features(input_features)
            audio_token_id = self.config.audio_token_id
            if audio_token_id is None:
                raise ValueError("audio_token_id is required for Voxtral audio inputs")
            audio_mask = (input_ids == audio_token_id)[..., None]
            inputs_embeds = masked_scatter(inputs_embeds, audio_mask, audio_embeds)
        return self.language_model(
            input_ids, inputs_embeds=inputs_embeds, mask=mask, cache=cache, **kwargs
        )

    def sanitize(self, weights: Dict[str, mx.array]):
        def transform_key(key: str) -> str:
            if key.startswith("model.language_model.lm_head."):
                return key.replace(
                    "model.language_model.lm_head.", "language_model.lm_head."
                )
            if key.startswith("model.lm_head."):
                return key.replace("model.lm_head.", "language_model.lm_head.")
            if key.startswith("lm_head."):
                return key.replace("lm_head.", "language_model.lm_head.")
            if key.startswith("model.language_model.model."):
                return key.replace(
                    "model.language_model.model.", "language_model.model."
                )
            if key.startswith("model.language_model."):
                return key.replace("model.language_model.", "language_model.model.")
            if key.startswith("model."):
                suffix = key[len("model.") :]
                if suffix.startswith(("embed_tokens", "layers", "norm")):
                    return "language_model.model." + suffix
            return key

        remapped = {transform_key(k): v for k, v in weights.items()}
        return {
            k: v
            for k, v in remapped.items()
            if k.startswith(("language_model.", "audio_tower.", "multi_modal_projector."))
        }


def masked_scatter(inputs_embeds: mx.array, mask: mx.array, source: mx.array) -> mx.array:
    mask = mask.astype(mx.bool_)
    if mask.shape != inputs_embeds.shape:
        mask = mx.broadcast_to(mask, inputs_embeds.shape)
    if not mask.any():
        return inputs_embeds

    result_flat = inputs_embeds.flatten()
    mask_flat = mask.flatten()
    source_flat = source.flatten()

    selection_mask = mx.cumsum(mask_flat.astype(mx.int32)) - 1
    bounded_indices = selection_mask % len(source_flat)
    selected_values = source_flat[bounded_indices]

    result_flat = mx.where(mask_flat, selected_values, result_flat)
    return result_flat.reshape(inputs_embeds.shape)


class VoxtralMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size, config.text_config.hidden_size, bias=False
        )
        projector_act = config.projector_hidden_act or "gelu"
        if projector_act == "gelu":
            self.act = nn.gelu
        else:
            raise ValueError(
                f"Unsupported projector_hidden_act: {projector_act}"
            )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=False
        )

    def __call__(self, audio_features: mx.array) -> mx.array:
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        return self.linear_2(hidden_states)
