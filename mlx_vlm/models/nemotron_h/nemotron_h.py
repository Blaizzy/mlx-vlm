from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .language import LanguageModel
from .modelopt import (
    install_modelopt_mxfp8_linears,
    install_modelopt_nvfp4_switch_linears,
)


class Model(nn.Module):
    _is_text_model = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        install_modelopt_mxfp8_linears(self, config.quantization_config)
        install_modelopt_nvfp4_switch_linears(self, config.quantization_config)

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        if pixel_values is not None:
            raise ValueError("Nemotron H is a text-only model.")
        if (
            kwargs.get("input_features") is not None
            or kwargs.get("audio_values") is not None
        ):
            raise ValueError("Nemotron H is a text-only model.")
        if input_ids is None:
            raise ValueError("input_ids are required for Nemotron H.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.get_input_embeddings(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if pixel_values is not None:
            raise ValueError("Nemotron H is a text-only model.")
        if attention_mask is not None:
            kwargs.setdefault("mask", attention_mask)
        return self.language_model(input_ids, **kwargs)

    @staticmethod
    def _prefixed_key(key: str) -> str:
        if key.startswith("language_model."):
            return key
        if key.startswith("backbone.") or key.startswith("lm_head."):
            return f"language_model.{key}"
        return key

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("mtp."):
                continue
            if key.endswith(
                (".input_scale", ".weight_scale_2", ".k_scale", ".v_scale")
            ):
                continue
            key = self._prefixed_key(key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            sanitized[key] = value

        weights = sanitized
        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"language_model.backbone.layers.{layer_idx}.mixer"
            for hf_name, mlx_name in (("down_proj", "fc2"), ("up_proj", "fc1")):
                expert_prefix = f"{prefix}.experts.0.{hf_name}"
                if f"{expert_prefix}.weight" not in weights:
                    continue

                stacked_prefix = f"{prefix}.switch_mlp.{mlx_name}"
                for suffix in ("weight", "scales", "global_scale"):
                    first_key = f"{expert_prefix}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{stacked_prefix}.{suffix}"] = mx.stack(
                        [
                            weights.pop(
                                f"{prefix}.experts.{expert_idx}.{hf_name}.{suffix}"
                            )
                            for expert_idx in range(self.config.n_routed_experts)
                        ]
                    )

        return weights

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate


__all__ = ["Model", "LanguageModel", "ModelConfig"]
