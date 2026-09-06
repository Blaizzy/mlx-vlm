import mlx.core as mx

from ..lfm2.language import LanguageModel as Lfm2LanguageModel


class LanguageModel(Lfm2LanguageModel):
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
