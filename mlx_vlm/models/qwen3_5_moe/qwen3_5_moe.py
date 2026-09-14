import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import (
    NORM_WEIGHT_SUFFIXES,
    sanitize_key,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # The MTP draft shard is separate from the base model. Its presence
        # must not select the base model's RMSNorm loading convention.
        weights = {key: value for key, value in weights.items() if "mtp." not in key}
        shift_norm_weights = should_shift_norm_weights(weights)

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                # process gate_up_proj [num_experts, 2 * intermediate_size, hidden_size]
                gate_up_weight = weights.pop(gate_up_key)
                mid = gate_up_weight.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up_weight[
                    ..., :mid, :
                ]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up_weight[
                    ..., mid:, :
                ]
                gate_up_scales_key = f"{gate_up_key}_scales"
                if gate_up_scales_key in weights:
                    gate_up_scales = weights.pop(gate_up_scales_key)
                    weights[f"{prefix}.switch_mlp.gate_proj.scales"] = gate_up_scales[
                        ..., :mid, :
                    ]
                    weights[f"{prefix}.switch_mlp.up_proj.scales"] = gate_up_scales[
                        ..., mid:, :
                    ]
                # down_proj
                down_key = f"{prefix}.experts.down_proj"
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(down_key)
                if f"{down_key}_scales" in weights:
                    weights[f"{prefix}.switch_mlp.down_proj.scales"] = weights.pop(
                        f"{down_key}_scales"
                    )
            elif f"{prefix}.experts.0.up_proj.weight" in weights:
                for name in ["up_proj", "down_proj", "gate_proj"]:
                    for suffix in ["weight", "scales", "biases"]:
                        first_key = f"{prefix}.experts.0.{name}.{suffix}"
                        if first_key not in weights:
                            continue
                        weights[f"{prefix}.switch_mlp.{name}.{suffix}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.experts.{e}.{name}.{suffix}")
                                for e in range(self.config.text_config.num_experts)
                            ]
                        )

        sanitized_weights = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in NORM_WEIGHT_SUFFIXES):
                if value.ndim == 1 and should_offset_norm_weight(
                    original_key, shift_norm_weights
                ):
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
