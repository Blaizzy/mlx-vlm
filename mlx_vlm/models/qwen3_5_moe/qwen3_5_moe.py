import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import sanitize_key
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
        # ignore mtp weights
        weights = {key: value for key, value in weights.items() if "mtp." not in key}

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{l}.mlp"
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
                # down_proj
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.up_proj.weight" in weights:
                for name in ["up_proj", "down_proj", "gate_proj"]:
                    weights[f"{prefix}.switch_mlp.{name}.weight"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{name}.weight")
                            for e in range(self.config.text_config.num_experts)
                        ]
                    )

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )

        sanitized_weights = {}
        for key, value in weights.items():
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
