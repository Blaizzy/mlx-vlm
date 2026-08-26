import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.qwen3_5 import (
    sanitize_key,
    should_offset_norm_weight,
    should_shift_norm_weights,
)
from .config import ModelConfig
from .language import build_layer_multipliers  # noqa: F401
from .language import LanguageModel, build_ngram_head_tables
from .vision import VisionModel

# Every RMSNorm in the text model stores a zero-centered weight, i.e. it scales
# by ``1 + weight``. The vision tower uses plain LayerNorms and is untouched.
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

# The n-gram embedding table is stored as `split_ngram_parts` shards.
_NGRAM_SHARD = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")

# Hash constants, recomputed from the config instead of being loaded.
_PLE_TABLES = (
    "ple_embedding.layer_multipliers",
    "ple_embedding.ngram_heads_vocab_sizes",
    "ple_embedding.ngram_heads_offsets",
)


def merge_ngram_embedding_shards(weights):
    shards = {}
    for key in list(weights):
        match = _NGRAM_SHARD.match(key)
        if match:
            shards.setdefault(match.group(1), {})[int(match.group(2))] = weights.pop(
                key
            )
    for prefix, parts in shards.items():
        weights[f"{prefix}.weight"] = mx.concatenate(
            [parts[i] for i in sorted(parts)], axis=0
        )
    return weights


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def _check_ple_table(self, key, value):
        """Fail loudly if a checkpoint's n-gram hash constants are not ours."""
        args = self.config.text_config
        layer_idx = int(re.search(r"layers\.(\d+)\.", key).group(1))
        ple_index = args.ple_layer_ids.index(layer_idx + 1)
        if key.endswith("layer_multipliers"):
            expected = build_layer_multipliers(
                args.vocab_size, args.ngram_size, ple_index, args.seed
            )
        else:
            sizes, offsets = build_ngram_head_tables(
                (args.ngram_size - 1) * args.heads_per_ngram,
                ple_index,
                args.ngram_vocab_size_base,
            )
            expected = sizes if key.endswith("vocab_sizes") else offsets
        if tuple(int(v) for v in value.tolist()) != tuple(expected):
            raise ValueError(
                f"{key} does not match the values derived from the config; the "
                "checkpoint was built with different n-gram hash constants"
            )

    def shard(self, group: Optional[mx.distributed.Group] = None) -> None:
        # The vision tower is small and stays replicated; only the language
        # model's experts are split. See `LanguageModel.shard`.
        self.language_model.shard(group)

    def sanitize(self, weights):
        # The MTP draft shard is separate from the base model. Its presence
        # must not select the base model's RMSNorm loading convention.
        weights = {key: value for key, value in weights.items() if "mtp." not in key}
        shift_norm_weights = should_shift_norm_weights(weights)

        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        weights = merge_ngram_embedding_shards(weights)
        for key in list(weights):
            if key.endswith(_PLE_TABLES):
                self._check_ple_table(key, weights.pop(key))

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

        sanitized_weights = {}
        for key, value in weights.items():
            original_key = key
            key = sanitize_key(key)

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if key.endswith(NORM_WEIGHT_SUFFIXES):
                if value.ndim == 1 and should_offset_norm_weight(
                    original_key, shift_norm_weights
                ):
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
