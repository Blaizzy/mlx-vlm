from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import CacheList, KVCache
from ..deepseek_v32.language import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Model,
    DeepseekV32MoE,
)
from .config import ModelConfig
from .fused_switch_glu import FusedSwitchGLU
from .hyper_connection import IdentityHyperConnection, IdentityHyperHead, hc_expand
from .indexer import HyV4Indexer
from .moe import weighted_expert_sum


def make_quantization_config(model):
    return {"group_size": 32, "bits": 8, "mode": "mxfp8"}


class HyV4Attention(DeepseekV32Attention):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        if self.skip_topk:
            self.indexer = None
        else:
            self.indexer = HyV4Indexer(config)
        self.linear_gate = nn.Linear(
            config.hidden_size,
            self.num_heads * self.v_head_dim,
            bias=False,
        )
        self.learnable_sink_param = mx.zeros((self.num_heads,), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        batch_size, sequence_length, _ = x.shape
        gate = self.linear_gate(x).reshape(
            batch_size, sequence_length, self.num_heads, self.v_head_dim
        )
        gate = mx.sigmoid(gate).transpose(0, 2, 1, 3)

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(
            batch_size, sequence_length, self.num_heads, self.q_head_dim
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(
            batch_size, sequence_length, 1, self.qk_rope_head_dim
        ).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None] * 2

        if self.indexer is not None:
            topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        else:
            topk_indices = prev_topk_indices

        if topk_indices is not None:
            if sequence_length == 1:
                indices = topk_indices[:, :, 0, :, None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(
                        indices, indices.shape[:-1] + (kv_latent.shape[-1],)
                    ),
                    axis=2,
                )
                k_pe = mx.take_along_axis(
                    k_pe,
                    mx.broadcast_to(indices, indices.shape[:-1] + (k_pe.shape[-1],)),
                    axis=2,
                )
                if mask is not None:
                    mask = mx.take_along_axis(mask, topk_indices, axis=-1)
            else:
                sparse_mask_shape = list(topk_indices.shape)
                sparse_mask_shape[-1] = kv_latent.shape[2]
                sparse_mask = mx.zeros(sparse_mask_shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, topk_indices, mx.array(True), axis=-1
                )
                if mask is not None:
                    sparse_mask = sparse_mask & mask
                mask = sparse_mask

        if self.indexer is not None and cache is not None and cache[0] is not None:
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if sequence_length == 1:
            q_nope = self.embed_q(q_nope)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
        output = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=pe_scores,
            sinks=self.learnable_sink_param.astype(q_nope.dtype),
        )
        if sequence_length == 1:
            output = self.unembed_out(output)

        output = output * gate
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)
        return self.o_proj(output), topk_indices


class HyV4MoE(DeepseekV32MoE):
    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        indices, scores = self.gate(x)
        y = weighted_expert_sum(self.switch_mlp(x, indices), scores)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class HyV4DecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = HyV4Attention(config, layer_idx)
        if hasattr(self.mlp, "switch_mlp"):
            self.mlp = HyV4MoE(config)
            self.mlp.switch_mlp = FusedSwitchGLU(
                config.hidden_size,
                config.moe_intermediate_size,
                config.n_routed_experts,
            )
        self.hc_attn_layer = IdentityHyperConnection(config)
        self.hc_mlp_layer = IdentityHyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        prev_topk_indices: Optional[mx.array] = None,
    ):
        residual = h
        x, post = self.hc_attn_layer(h)
        x, topk_indices = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            prev_topk_indices,
        )
        h = hc_expand(x, residual, post)

        residual = h
        x, post = self.hc_mlp_layer(h)
        x = self.mlp(self.post_attention_layernorm(x))
        return hc_expand(x, residual, post), topk_indices


class HyV4Model(DeepseekV32Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        self.layers = [
            HyV4DecoderLayer(config, index) for index in range(config.num_hidden_layers)
        ]
        self.hc_head = IdentityHyperHead(config)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(x) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :],
            (h.shape[0], h.shape[1], self.config.hc_mult, h.shape[2]),
        )
        h = mx.contiguous(h)

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(
            h[:, :, 0, :], cache[0][0] if cache[0] else None, return_array=True
        )

        prev_topk_indices = None
        for index in range(self.num_layers):
            h, prev_topk_indices = self.layers[self.start_idx + index](
                h,
                mask,
                cache[index],
                prev_topk_indices,
            )

        return self.norm(self.hc_head(h))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = HyV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        logits = self.lm_head(
            self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        )
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return Model.sanitize(self, weights)

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return Model.cast_predicate.fget(self)

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.self_attn.skip_topk:
                caches.append(CacheList(KVCache()))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = HyV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        return LanguageModelOutput(
            logits=self.lm_head(
                self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
            )
        )

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        supported_model_prefixes = (
            "model.embed_tokens.",
            "model.layers.",
            "model.norm.",
        )
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("model.") or key.startswith(supported_model_prefixes)
        }

        def pack_mxfp8(weight: mx.array) -> mx.array:
            if weight.dtype != mx.uint8:
                weight = weight.view(mx.uint8)
            return weight.view(mx.uint32)

        def unpack_mxfp8(weight: mx.array, scales: mx.array) -> mx.array:
            return mx.dequantize(
                pack_mxfp8(weight),
                scales,
                None,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )

        def repack_mxfp8(prefix: str, weight: mx.array):
            packed, scales = mx.quantize(weight, group_size=32, bits=8, mode="mxfp8")
            weights[f"{prefix}.weight"] = packed
            weights[f"{prefix}.scales"] = scales

        for layer_index in range(self.args.num_hidden_layers):
            mlp_prefix = f"model.layers.{layer_index}.mlp"
            fused_weight_key = f"{mlp_prefix}.experts.gate_up_proj"
            fused_scale_key = f"{mlp_prefix}.experts.gate_up_proj_scale"
            if fused_weight_key in weights:
                gate_up = weights.pop(fused_weight_key)
                gate_up_scale = weights.pop(fused_scale_key)
                weights[f"{mlp_prefix}.switch_mlp.gate_up_proj.weight"] = pack_mxfp8(
                    gate_up
                )
                weights[f"{mlp_prefix}.switch_mlp.gate_up_proj.scales"] = gate_up_scale

            switch_prefix = f"{mlp_prefix}.switch_mlp"
            for suffix in ("weight", "scales", "biases"):
                gate_key = f"{switch_prefix}.gate_proj.{suffix}"
                up_key = f"{switch_prefix}.up_proj.{suffix}"
                if gate_key in weights and up_key in weights:
                    weights[f"{switch_prefix}.gate_up_proj.{suffix}"] = mx.concatenate(
                        [weights.pop(gate_key), weights.pop(up_key)], axis=1
                    )

            down_weight_key = f"{mlp_prefix}.experts.down_proj"
            down_scale_key = f"{mlp_prefix}.experts.down_proj_scale"
            if down_weight_key in weights:
                weights[f"{mlp_prefix}.switch_mlp.down_proj.weight"] = pack_mxfp8(
                    weights.pop(down_weight_key)
                )
                weights[f"{mlp_prefix}.switch_mlp.down_proj.scales"] = weights.pop(
                    down_scale_key
                )

            attn_prefix = f"model.layers.{layer_index}.self_attn"
            kv_weight_key = f"{attn_prefix}.kv_b_proj.weight"
            kv_scale_key = f"{attn_prefix}.kv_b_proj.weight_scale"
            if kv_weight_key in weights:
                kv = unpack_mxfp8(weights.pop(kv_weight_key), weights.pop(kv_scale_key))
                kv = kv.reshape(
                    self.args.num_attention_heads,
                    self.args.qk_nope_head_dim + self.args.v_head_dim,
                    self.args.kv_lora_rank,
                )
                embed_q = mx.contiguous(
                    kv[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
                )
                unembed_out = mx.contiguous(kv[:, self.args.qk_nope_head_dim :, :])
                repack_mxfp8(f"{attn_prefix}.embed_q", embed_q)
                repack_mxfp8(f"{attn_prefix}.unembed_out", unembed_out)

        transformed = {}
        for key, value in weights.items():
            if key.endswith(".weight_scale"):
                weight_key = key[: -len("_scale")]
                if weight_key not in weights:
                    raise ValueError(f"Missing MXFP8 weight for {weight_key}.")
                transformed[weight_key] = pack_mxfp8(weights[weight_key])
                transformed[f"{weight_key[:-len('weight')]}scales"] = value
            elif key.endswith(".weight") and f"{key}_scale" in weights:
                continue
            else:
                transformed[key] = value
        return transformed

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(path):
            return not (
                "e_score_correction_bias" in path
                or "learnable_sink_param" in path
                or ".hc_attn_layer." in path
                or ".hc_mlp_layer." in path
                or ".hc_head." in path
            )

        return predicate

    def make_cache(self):
        return LanguageModel.make_cache(self)
