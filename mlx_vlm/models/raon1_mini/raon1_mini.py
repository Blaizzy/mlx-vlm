import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        return self.language_model(input_ids, cache=cache, **kwargs)

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def cast_predicate(self):
        return lambda path: not (
            path.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "e_score_correction_bias",
                    "res.eta",
                    "weight_scale_2",
                )
            )
            or (self.config.raon_fp32_export_head and path.endswith("lm_head.weight"))
        )

    @property
    def quant_predicate(self):
        return lambda path, module: not (
            path.endswith("moe.gate")
            or (self.config.raon_fp32_export_head and path.endswith("lm_head"))
        )

    def sanitize(self, weights):
        # Normalize names before transforming values so converted checkpoints
        # follow exactly the same idempotent path as the source checkpoint.
        weights = {
            key if key.startswith("language_model.") else "language_model." + key: value
            for key, value in weights.items()
        }
        args = self.config
        for i in range(
            args.num_hidden_layers,
            args.num_hidden_layers + args.num_nextn_predict_layers,
        ):
            prefix = f"language_model.model.layers.{i}."
            weights = {
                key: value
                for key, value in weights.items()
                if not key.startswith(prefix)
            }
        for key in list(weights):
            if key.endswith(".global_scale"):
                scale = weights.pop(key)
                # Routed experts store an absmax for native NVFP4 gather;
                # dense projections store the multiplicative scale directly.
                if ".moe.switch_mlp." in key or ".moe.experts." in key:
                    scale = scale.astype(mx.float32) / (448 * 6)
                weights[key.removesuffix("global_scale") + "weight_scale_2"] = scale
            elif key.endswith(".weight_scale_inv"):
                prefix = key.removesuffix("weight_scale_inv")
                value, scales = weights[prefix + "weight"], weights.pop(key)
                if value.dtype != mx.uint8 or scales.dtype != mx.uint8:
                    raise ValueError(
                        "Raon MXFP8 requires E4M3 weight bytes and E8M0 scale bytes"
                    )
                weights[prefix + "weight"] = mx.contiguous(value).view(mx.uint32)
                weights[prefix + "scales"] = scales
        for i, layer in enumerate(self.layers):
            prefix = f"language_model.model.layers.{i}."
            if layer.is_linear:
                sources = [
                    prefix + f"self_attn.{name}_conv1d.weight"
                    for name in ("q", "k", "v")
                ]
                target = prefix + "self_attn.qkv_conv1d.weight"
                if all(key in weights for key in sources):
                    values = [weights.pop(key) for key in sources]
                    values = [
                        (
                            value[..., None]
                            if value.ndim == 2
                            else (
                                value.transpose(0, 2, 1)
                                if value.shape[1] == 1
                                else value
                            )
                        )
                        for value in values
                    ]
                    weights[target] = mx.concatenate(values, axis=0)
                if target in weights:
                    value = weights[target]
                    if value.ndim == 2:
                        value = value[..., None]
                    elif value.shape[1] == 1 and value.shape[2] != 1:
                        value = value.transpose(0, 2, 1)
                    weights[target] = value
            else:
                key = prefix + "self_attn.kv_b_proj.weight"
                if key in weights:
                    value = weights.pop(key).reshape(
                        args.num_attention_heads,
                        args.qk_nope_head_dim + args.v_head_dim,
                        args.kv_lora_rank,
                    )
                    weights[prefix + "self_attn.embed_q.weight"] = value[
                        :, : args.qk_nope_head_dim
                    ].swapaxes(-1, -2)
                    weights[prefix + "self_attn.unembed_out.weight"] = value[
                        :, args.qk_nope_head_dim :
                    ]
            if layer.is_moe:
                experts = prefix + "moe.experts."
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases", "weight_scale_2"):
                        if experts + f"0.{projection}.{suffix}" in weights:
                            weights[
                                prefix + f"moe.switch_mlp.{projection}.{suffix}"
                            ] = mx.stack(
                                [
                                    weights.pop(
                                        experts + f"{expert}.{projection}.{suffix}"
                                    )
                                    for expert in range(args.num_experts)
                                ]
                            )
        if args.tie_word_embeddings:
            weights.pop("language_model.lm_head.weight", None)
        return weights
