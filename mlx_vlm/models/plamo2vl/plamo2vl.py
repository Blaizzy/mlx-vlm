import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .language import LanguageModel, TextConfig, RMSNorm
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int | list[int]] = None
    pad_token_id: Optional[int] = None
    image_token_index: int = 100002

    @classmethod
    def from_dict(cls, params):
        values = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        if "vision_config" in params and "image_token_index" not in values:
            values["image_token_index"] = params["vision_config"].get("image_token_id", 100002)
        return cls(**values)


class Bias(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.bias


class MLPImageProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.vision_config.image_proj_hidden_size
        self.norm0 = RMSNorm(
            config.vision_config.image_feature_size,
            eps=config.text_config.rms_norm_eps,
        )
        self.bias0 = Bias(config.vision_config.image_feature_size)
        self.linear1 = nn.Linear(
            config.vision_config.image_feature_size, hidden_size, bias=False
        )
        self.bias1 = Bias(hidden_size)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(
            hidden_size, config.text_config.hidden_size, bias=False
        )
        self.bias2 = Bias(config.text_config.hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.norm0(hidden_states)
        hidden_states = self.bias0(hidden_states)
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.bias1(hidden_states)
        hidden_states = self.act1(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.bias2(hidden_states)
        return hidden_states


class Model(nn.Module):
    no_chunked_prefill = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = LanguageModel(config.text_config)
        self.vision_model = VisionModel(config.vision_config)
        self.image_proj = MLPImageProjector(config)

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("vision_model.vision_encoder.model.head."):
                continue
            if key.startswith("model."):
                key = f"language_model.{key}"
            if key.startswith("lm_head."):
                key = f"language_model.{key}"
            if "._bias" in key:
                key = key.replace("._bias", ".bias")
            if "vision_model.vision_encoder.model.embeddings.patch_embedding.weight" in key:
                if value.ndim == 4 and value.shape[1] == 3:
                    value = value.transpose(0, 2, 3, 1)
            if "language_model.model.layers.layers." in key and "conv1d.weight" in key:
                if value.ndim == 3 and value.shape[-1] != 1:
                    value = value.moveaxis(2, 1)
            sanitized[key] = value
        return sanitized

    def get_input_embeddings(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        del kwargs
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        pixel_values = pixel_values.astype(
            self.vision_model.vision_encoder.model.embeddings.patch_embedding.weight.dtype
        )
        image_features = self.vision_model(input_ids, pixel_values)
        image_embeds = self.image_proj(image_features).astype(mx.float32)
        inputs_embeds = inputs_embeds.astype(mx.float32)
        image_mask = mx.expand_dims(
            input_ids == self.config.vision_config.image_token_id, -1
        )
        return InputEmbeddingsFeatures(
            inputs_embeds=mx.where(image_mask, image_embeds, inputs_embeds)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array],
        mask: Optional[mx.array],
        cache=None,
        **kwargs,
    ):
        del mask, kwargs
        embedding_output = self.get_input_embeddings(input_ids, pixel_values)
        return self.language_model(
            None,
            cache=cache,
            inputs_embeds=embedding_output.inputs_embeds,
        )
