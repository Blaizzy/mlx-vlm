import re

import mlx.core as mx
import mlx.nn as nn

from ..mistral3.mistral3 import Mistral3MultiModalProjector
from ..mistral3.mistral3 import Model as Mistral3Model
from ..pixtral import VisionModel
from .config import ModelConfig
from .language import LanguageModel


class Model(Mistral3Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.vision_feature_layer = config.vision_feature_layer

    def shard(self, group=None):
        self.language_model.shard(group)

    def sanitize(self, weights):
        """Map native Mistral names and stack experts after shared conversion."""
        args = self.language_model.args
        sanitized = {}
        aliases = {
            "tok_embeddings.weight": "language_model.model.embed_tokens.weight",
            "norm.weight": "language_model.model.norm.weight",
            "lerope.alpha": "language_model.model.rotary_emb.alpha",
            "output.weight": "language_model.lm_head.weight",
            "pre_mm_projector_norm.weight": "multi_modal_projector.norm.weight",
            "patch_merger.merging_layer.weight": (
                "multi_modal_projector.patch_merger.merging_layer.weight"
            ),
            "vision_language_adapter.w_in.weight": (
                "multi_modal_projector.linear_1.weight"
            ),
            "vision_language_adapter.w_out.weight": (
                "multi_modal_projector.linear_2.weight"
            ),
        }
        for key, value in weights.items():
            if key.startswith("mtp") or key.endswith(".input_scale"):
                continue
            if key.startswith("layers."):
                _, index, rest = key.split(".", 2)
                expert = re.match(r"experts\.(\d+)\.", rest)
                if expert and not args.expert_start <= int(expert[1]) < args.expert_end:
                    continue
                if rest.startswith(("experts.", "shared_experts.", "gate.")):
                    rest = "feed_forward." + rest
                key = f"language_model.model.layers.{index}.{rest}"
            elif key.startswith("vision_encoder."):
                key = key.replace("vision_encoder.", "vision_tower.vision_model.", 1)
                for source, target in (
                    ("wq", "q_proj"),
                    ("wk", "k_proj"),
                    ("wv", "v_proj"),
                    ("wo", "o_proj"),
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    key = key.replace(f".{source}.", f".{target}.")
                if key.endswith("patch_conv.weight"):
                    value = value.transpose(0, 2, 3, 1)
            else:
                key = aliases.get(key, key)
            sanitized[key] = value
        weights.clear()

        for i in range(args.first_k_dense_replace, args.num_hidden_layers):
            prefix = f"language_model.model.layers.{i}.feed_forward.experts"
            for projection in ("w1", "w2", "w3"):
                source = f"{prefix}.{args.expert_start}.{projection}"
                if f"{source}.weight" not in sanitized:
                    continue  # Already stacked MLX weights are left unchanged.
                suffixes = ["weight"]
                if args.nvfp4:
                    suffixes.extend(("scales", "weight_scale_2"))
                for suffix in suffixes:
                    sanitized[f"{prefix}.{projection}.{suffix}"] = mx.stack(
                        [
                            sanitized.pop(f"{prefix}.{expert}.{projection}.{suffix}")
                            for expert in range(args.expert_start, args.expert_end)
                        ]
                    )
        return sanitized

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids, inputs_embeds=features.inputs_embeds, mask=mask, cache=cache
        )

    @property
    def quant_predicate(self):
        # Native expert matrices are already quantized and have no to_quantized.
        return lambda path, module: hasattr(module, "to_quantized")
