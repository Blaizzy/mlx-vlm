from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput
from ..cache import ArraysCache, KVCache
from ..lfm2.language import Lfm2Model
from .config import ModelConfig


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Lfm2Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        out = self.model(inputs, cache, inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        sanitized_weights = {}
        for name, param in weights.items():
            if "conv.weight" in name and param.shape[-1] > param.shape[1]:
                param = param.transpose(0, 2, 1)

            replacements = {
                "w1.weight": "gate_proj.weight",
                "w2.weight": "down_proj.weight",
                "w3.weight": "up_proj.weight",
            }
            for old, new in replacements.items():
                if old in name:
                    name = name.replace(old, new)

            sanitized_weights[name] = param

        return self._stack_experts(sanitized_weights)

    def _stack_experts(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}.feed_forward"
            for proj in ["gate_proj", "down_proj", "up_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.experts.0.{proj}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{proj}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                    )
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("feed_forward.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            KVCache() if layer.is_attention_layer else ArraysCache(size=1)
            for layer in self.layers
        ]
