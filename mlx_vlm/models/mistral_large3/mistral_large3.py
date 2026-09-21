from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..deepseek_v3.language import LanguageModel
from ..mistral3.mistral3 import Mistral3MultiModalProjector
from ..mistral3.mistral3 import Model as Mistral3VLM
from ..pixtral import VisionModel
from .config import ModelConfig

_EXPERT_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

_VISION_RENAMES = (
    (".attention.wq.", ".attention.q_proj."),
    (".attention.wk.", ".attention.k_proj."),
    (".attention.wv.", ".attention.v_proj."),
    (".attention.wo.", ".attention.o_proj."),
    (".feed_forward.w1.", ".feed_forward.gate_proj."),
    (".feed_forward.w2.", ".feed_forward.down_proj."),
    (".feed_forward.w3.", ".feed_forward.up_proj."),
)

_ATTN_RENAMES = {
    "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attention.wq_a": "self_attn.q_a_proj",
    "attention.q_a_norm": "self_attn.q_a_layernorm",
    "attention.wq_b": "self_attn.q_b_proj",
    "attention.wkv_a_with_mqa": "self_attn.kv_a_proj_with_mqa",
    "attention.kv_a_norm": "self_attn.kv_a_layernorm",
    "attention.wkv_b": "self_attn.kv_b_proj",
    "attention.wo": "self_attn.o_proj",
    "gate": "mlp.gate",
}


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)
        self.vision_feature_layer = config.vision_feature_layer

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        return Mistral3VLM.get_input_embeddings(self, input_ids, pixel_values, **kwargs)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        return Mistral3VLM.__call__(
            self, input_ids, pixel_values, mask, cache, **kwargs
        )

    @property
    def layers(self):
        return self.language_model.model.layers

    @staticmethod
    def _map_language_base(base: str) -> str:
        if base == "tok_embeddings":
            return "model.embed_tokens"
        if base == "norm":
            return "model.norm"
        if base == "output":
            return "lm_head"

        _, layer, *tail = base.split(".")
        rest = ".".join(tail)
        prefix = f"model.layers.{layer}"

        if rest in _ATTN_RENAMES:
            return f"{prefix}.{_ATTN_RENAMES[rest]}"
        if rest.startswith("experts."):
            _, expert, proj = rest.split(".")
            return f"{prefix}.mlp.experts.{expert}.{_EXPERT_PROJ[proj]}"
        if rest.startswith("shared_experts."):
            _, proj = rest.split(".")
            return f"{prefix}.mlp.shared_experts.{_EXPERT_PROJ[proj]}"
        if rest.startswith("feed_forward."):
            _, proj = rest.split(".")
            return f"{prefix}.mlp.{_EXPERT_PROJ[proj]}"
        raise ValueError(f"Unmapped language key: {base}")

    def _map_language_key(self, key: str) -> str:
        # Source fp8 block scales feed the deepseek_v3 dequant path as `weight_scale_inv`;
        # affine `scales`/`biases` (a re-quantized checkpoint) pass through unchanged.
        for suffix, mapped in (
            (".weight_scale", ".weight_scale_inv"),
            (".scales", ".scales"),
            (".biases", ".biases"),
            (".weight", ".weight"),
        ):
            if key.endswith(suffix):
                return self._map_language_base(key[: -len(suffix)]) + mapped
        return self._map_language_base(key)

    def sanitize(self, weights):
        if any(
            k.startswith(("language_model.", "vision_tower.", "multi_modal_projector."))
            for k in weights
        ):
            return weights

        lang, vis, proj = {}, {}, {}

        for k, v in weights.items():
            if k == "pre_mm_projector_norm.weight":
                proj["multi_modal_projector.norm.weight"] = v
            elif k.startswith("patch_merger."):
                proj["multi_modal_projector." + k] = v
            elif k.startswith("vision_language_adapter."):
                _, leaf, tail = k.split(".", 2)
                name = {"w_in": "linear_1", "w_out": "linear_2"}[leaf]
                proj[f"multi_modal_projector.{name}.{tail}"] = v
            elif k.startswith("vision_encoder."):
                nk = k[len("vision_encoder.") :]
                for src, dst in _VISION_RENAMES:
                    nk = nk.replace(src, dst)
                vis["vision_model." + nk] = v
            else:
                lang[self._map_language_key(k)] = v

        tc = self.config.text_config
        for layer in range(tc.first_k_dense_replace, tc.num_hidden_layers):
            key = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
            if key not in lang:
                lang[key] = mx.zeros((tc.n_routed_experts,))

        lang = self.language_model.sanitize(lang)
        vis = self.vision_tower.sanitize(vis)

        out = {f"language_model.{k}": v for k, v in lang.items()}
        out.update({f"vision_tower.{k}": v for k, v in vis.items()})
        out.update(proj)
        return out
