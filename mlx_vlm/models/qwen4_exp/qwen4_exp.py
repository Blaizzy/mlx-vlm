import re

import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import (
    sanitize_key,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig
from .fp8 import convert_qwen4_exp_fp8_weights
from .language import LanguageModel
from .vision import VisionModel

_NGRAM_SHARD_RE = re.compile(r"\.ngram_embedding\.shard_(\d+)(?=\.)")

NORM_WEIGHT_SUFFIXES = (
    ".q_norm.weight",
    ".k_norm.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
)


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # The MTP predictor is a separate speculative artifact and is not part
        # of the base conditional-generation model.
        weights = {
            key: value for key, value in weights.items() if not key.startswith("mtp.")
        }
        weights = convert_qwen4_exp_fp8_weights(weights)
        shift_norm_weights = should_shift_norm_weights(weights)
        if self.config.text_config.ple_storage:
            weights = {
                key: value
                for key, value in weights.items()
                if ".ple.ple_embedding.ngram_embedding." not in key
            }

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for layer_idx in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{layer_idx}.mlp"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            if gate_up_key in weights:
                gate_up = weights.pop(gate_up_key)
                midpoint = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[
                    ..., :midpoint, :
                ]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[
                    ..., midpoint:, :
                ]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )

        sanitized = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)
            key = _NGRAM_SHARD_RE.sub(r".ngram_embedding.shards.\1", key)
            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if (
                value.ndim == 1
                and key.endswith(NORM_WEIGHT_SUFFIXES)
                and should_offset_norm_weight(original_key, shift_norm_weights)
            ):
                value += 1.0
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
