import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import InputEmbeddingsFeatures, LanguageModelOutput

_is_text_model_arch = True


@dataclass
class AttributeConfig:
    """Attribute-style config wrapper."""

    values: Dict[str, Any]

    def __post_init__(self):
        for key, value in self.values.items():
            setattr(self, key, value)

    def to_dict(self):
        return dict(self.values)


class TextConfig(AttributeConfig):
    @classmethod
    def from_dict(cls, params):
        return cls(dict(params or {}))


class ModelConfig(AttributeConfig):
    @classmethod
    def from_dict(cls, params):
        params = dict(params or {})
        from mlx_lm.utils import _get_classes

        model_class, model_args_class = _get_classes(params)
        return cls(params, model_class, model_args_class)

    def __init__(self, values, model_class, model_args_class):
        self.model_class = model_class
        self.model_args_class = model_args_class
        super().__init__(values)
        self.text_config = TextConfig.from_dict(values.get("text_config", {}))

    def __post_init__(self):
        for key, value in self.values.items():
            if key in {
                "text_config",
                "vision_config",
                "audio_config",
                "projector_config",
                "perceiver_config",
            }:
                continue
            setattr(self, key, value)


class LanguageModel(nn.Module):
    """Adapter that makes an mlx-lm model return mlx-vlm language outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model
        self.model_type = getattr(model, "model_type", None)

    @property
    def model(self):
        return getattr(self._model, "model", self._model)

    @property
    def layers(self):
        if hasattr(self._model, "layers"):
            return self._model.layers
        return []

    def make_cache(self):
        if hasattr(self._model, "make_cache"):
            return self._model.make_cache()

        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in self.layers]

    def shard(self, group: Optional[mx.distributed.Group] = None):
        if not hasattr(self._model, "shard"):
            raise ValueError("The underlying mlx-lm model does not support sharding")
        return self._model.shard(group)

    @staticmethod
    def _accepted_params(callable_obj) -> set[str]:
        try:
            return set(inspect.signature(callable_obj).parameters)
        except (TypeError, ValueError):
            return set()

    def _token_embedding(self):
        candidates = (
            (self.model, "embed_tokens"),
            (self.model, "wte"),
            (self.model, "embeddings"),
            (self.model, "tok_embeddings"),
            (self.model, "word_embeddings"),
            (self._model, "embed_tokens"),
            (self._model, "wte"),
            (self._model, "embeddings"),
        )
        for module, attr in candidates:
            if hasattr(module, attr):
                return getattr(module, attr)
        return None

    def input_embeds(self, input_ids: mx.array) -> mx.array:
        embedding = self._token_embedding()
        if embedding is None:
            raise ValueError(
                "The wrapped mlx-lm model does not expose token embeddings."
            )
        return embedding(input_ids)

    def _project_hidden_states(self, hidden_states):
        if hasattr(self._model, "lm_head"):
            return self._model.lm_head(hidden_states)

        embedding = self._token_embedding()
        if embedding is not None and hasattr(embedding, "as_linear"):
            return embedding.as_linear(hidden_states)

        return None

    def _call_inner_with_embeddings(self, inputs, cache=None, input_embeddings=None):
        inner = self.model
        if inner is self._model:
            return None

        params = self._accepted_params(inner.__call__)
        if "input_embeddings" not in params:
            return None

        call_kwargs = {"input_embeddings": input_embeddings}
        if "cache" in params:
            call_kwargs["cache"] = cache

        hidden_states = inner(inputs, **call_kwargs)
        return self._project_hidden_states(hidden_states)

    def _call_model(self, inputs, cache=None, **kwargs):
        params = self._accepted_params(self._model.__call__)
        input_embeddings = kwargs.get("inputs_embeds", kwargs.get("input_embeddings"))

        call_kwargs = {}
        if "cache" in params:
            call_kwargs["cache"] = cache

        for source, target in (
            ("mask", "mask"),
            ("inputs_embeds", "input_embeddings"),
            ("input_embeddings", "input_embeddings"),
            ("per_layer_inputs", "per_layer_inputs"),
        ):
            if target in params and source in kwargs and kwargs[source] is not None:
                call_kwargs[target] = kwargs[source]

        if "input_embeddings" not in params and input_embeddings is not None:
            logits = self._call_inner_with_embeddings(
                inputs, cache=cache, input_embeddings=input_embeddings
            )
            if logits is not None:
                return logits

        return self._model(inputs, **call_kwargs)

    def __call__(self, inputs: mx.array, cache=None, **kwargs):
        logits = self._call_model(inputs, cache=cache, **kwargs)
        return LanguageModelOutput(logits=logits)


class Model(nn.Module):
    """VLM-shaped wrapper around a text-only model."""

    _is_text_model = True

    def __init__(self, model_or_config, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        if isinstance(model_or_config, ModelConfig):
            self.config = model_or_config
            model_args = self.config.model_args_class.from_dict(self.config.values)
            model = self.config.model_class(model_args)
        else:
            model = model_or_config
            self.config = AttributeConfig(dict(config or {}))
        self.language_model = LanguageModel(model)

    def sanitize(self, weights):
        model = self.language_model._model
        if hasattr(model, "sanitize"):
            return model.sanitize(weights)
        return weights

    def load_weights(self, weights, *args, **kwargs):
        return self.language_model._model.load_weights(weights, *args, **kwargs)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Text-only models do not accept image inputs.")
        if (
            kwargs.get("input_features") is not None
            or kwargs.get("audio_values") is not None
        ):
            raise ValueError("Text-only models do not accept audio inputs.")
        if input_ids is None:
            raise ValueError("input_ids are required for text-only models.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.input_embeds(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Text-only models do not accept image inputs.")
        if attention_mask is not None:
            kwargs.setdefault("mask", attention_mask)
        return self.language_model(input_ids, **kwargs)

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group: Optional[mx.distributed.Group] = None):
        return self.language_model.shard(group)

    @property
    def layers(self):
        return self.language_model.layers


TextOnlyModel = Model
