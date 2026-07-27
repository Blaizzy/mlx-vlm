import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients

from ...turboquant import BatchTurboQuantKVCache, TurboQuantKVCache
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..mla import MultiLinear
from ..mlp import DeepseekMLP as Glm4MoeLiteMLP
from ..pipeline import PipelineMixin
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class Glm4MoeLiteAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scale = self.q_head_dim**-0.5

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=config.attention_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(
                self.q_lora_rank,
                eps=config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.num_heads,
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank,
            self.v_head_dim,
            self.num_heads,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        rope_params = config.rope_scaling
        if rope_params is not None:
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            if mscale_all_dim:
                scaling_factor = rope_params["factor"]
                if scaling_factor > 1:
                    scale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                    self.scale *= scale * scale

        self.rope = initialize_rope(
            dims=self.qk_rope_head_dim,
            base=self.rope_theta,
            traditional=True,
            max_position_embeddings=self.max_position_embeddings,
            scaling_config=rope_params,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.q_head_dim,
        ).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(
            compressed_kv,
            [self.kv_lora_rank],
            axis=-1,
        )
        k_pe = k_pe.reshape(
            batch_size,
            sequence_length,
            1,
            self.qk_rope_head_dim,
        ).transpose(0, 2, 1, 3)
        kv_latent = self.kv_a_layernorm(compressed_kv)

        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        attention_cache = cache
        if cache is not None:
            kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)
            if isinstance(cache, (TurboQuantKVCache, BatchTurboQuantKVCache)):
                kv_latent, k_pe = cache.dequantize(kv_latent, k_pe)
                kv_latent = kv_latent.astype(x.dtype)
                k_pe = k_pe.astype(x.dtype)
                attention_cache = None

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask,
                pe_scores,
                mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
            )

        if sequence_length == 1:
            q_nope = self.embed_q(q_nope)
            keys = values = kv_latent
        else:
            keys = self.embed_q(kv_latent, transpose=False)
            values = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q_nope,
            keys,
            values,
            cache=attention_cache,
            scale=self.scale,
            mask=pe_scores,
        )
        if sequence_length == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            sequence_length,
            -1,
        )
        return self.o_proj(output)


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
):
    scores = mx.sigmoid(gates.astype(mx.float32))
    original_scores = scores
    scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores,
            mx.stop_gradient(group_idx),
            mx.array(0.0),
            axis=-2,
        )
        scores = mx.flatten(scores, -2, -1)

    indices = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(original_scores, indices, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    scores = scores * routed_scaling_factor
    return indices, scores


class MoEGate(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))
        if config.topk_method != "noaux_tc":
            raise ValueError(f"Unsupported top-k method: {config.topk_method}")

    def __call__(self, x):
        return group_expert_select(
            x @ self.weight.T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm4MoeLiteMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
        )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeLiteMLP(
                config=config,
                intermediate_size=intermediate_size,
            )
        self.sharding_group = None

    def __call__(self, x):
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        indices, scores = self.gate(x)
        y = self.switch_mlp(x, indices)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(x)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Glm4MoeLiteAttention(config)
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = Glm4MoeLiteMoE(config) if use_moe else Glm4MoeLiteMLP(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        attention_output = self.self_attn(self.input_layernorm(x), mask, cache)
        hidden_states = x + attention_output
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + mlp_output


class Glm4MoeLiteModel(PipelineMixin, nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm4MoeLiteDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        )

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        if cache is None:
            cache = [None] * len(self.pipeline_layers)
        mask = create_attention_mask(
            hidden_states,
            cache[0],
            return_array=True,
        )

        if pipeline_rank < pipeline_size - 1:
            hidden_states = mx.distributed.recv_like(
                hidden_states,
                pipeline_rank + 1,
            )

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            hidden_states = layer(hidden_states, mask, cache=layer_cache)

        if pipeline_rank != 0:
            hidden_states = mx.distributed.send(
                hidden_states,
                (pipeline_rank - 1) % pipeline_size,
            )
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, hidden_states)

        if pipeline_size > 1:
            hidden_states = mx.distributed.all_gather(hidden_states)[
                : hidden_states.shape[0]
            ]

        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = Glm4MoeLiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        output = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        return LanguageModelOutput(logits=self.lm_head(output))

    def sanitize(self, weights):
        def is_mtp_layer(key):
            subkeys = key.split(".")
            return (
                len(subkeys) >= 3
                and subkeys[1] == "layers"
                and int(subkeys[2]) >= self.args.num_hidden_layers
            )

        weights = {
            key: value for key, value in weights.items() if not is_mtp_layer(key)
        }

        for layer_idx in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}"
            for target in ("gate_proj", "down_proj", "up_proj"):
                for suffix in ("weight", "scales", "biases"):
                    expert_key = f"{prefix}.mlp.experts.0.{target}.{suffix}"
                    if expert_key not in weights:
                        continue
                    expert_weights = [
                        weights.pop(f"{prefix}.mlp.experts.{expert}.{target}.{suffix}")
                        for expert in range(self.args.n_routed_experts)
                    ]
                    weights[f"{prefix}.mlp.switch_mlp.{target}.{suffix}"] = mx.stack(
                        expert_weights
                    )

            attention_prefix = f"{prefix}.self_attn"
            kv_b_key = f"{attention_prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue

            quantized = f"{attention_prefix}.kv_b_proj.scales" in weights
            kv_b = weights.pop(kv_b_key)
            head_dim = self.args.qk_nope_head_dim + self.args.v_head_dim

            if quantized:
                dims = self.args.kv_lora_rank
                scales = weights.pop(f"{attention_prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{attention_prefix}.kv_b_proj.biases")
                bits = (kv_b.shape[-1] * 32) // dims
                group_size = dims // scales.shape[-1]
                kv_b = mx.dequantize(
                    kv_b,
                    scales,
                    biases,
                    bits=bits,
                    group_size=group_size,
                )

            kv_b = kv_b.reshape(self.args.num_attention_heads, head_dim, -1)
            embed_q = mx.contiguous(
                kv_b[:, : self.args.qk_nope_head_dim, :].swapaxes(-1, -2)
            )
            unembed_out = mx.contiguous(kv_b[:, self.args.qk_nope_head_dim :, :])

            if quantized:
                embed_q, embed_scales, embed_biases = mx.quantize(
                    embed_q,
                    bits=bits,
                    group_size=group_size,
                )
                unembed_out, unembed_scales, unembed_biases = mx.quantize(
                    unembed_out,
                    bits=bits,
                    group_size=group_size,
                )
                weights[f"{attention_prefix}.embed_q.scales"] = embed_scales
                weights[f"{attention_prefix}.embed_q.biases"] = embed_biases
                weights[f"{attention_prefix}.unembed_out.scales"] = unembed_scales
                weights[f"{attention_prefix}.unembed_out.biases"] = unembed_biases

            weights[f"{attention_prefix}.embed_q.weight"] = embed_q
            weights[f"{attention_prefix}.unembed_out.weight"] = unembed_out

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        rank = group.rank()
        world_size = group.size()

        for layer in self.model.layers:
            attention = layer.self_attn
            if attention.q_lora_rank is None:
                attention.q_proj = shard_linear(
                    attention.q_proj,
                    "all-to-sharded",
                    group=group,
                )
            else:
                attention.q_b_proj = shard_linear(
                    attention.q_b_proj,
                    "all-to-sharded",
                    group=group,
                )
            attention.num_heads //= world_size
            start_head = rank * attention.num_heads
            end_head = start_head + attention.num_heads
            attention.embed_q.apply(lambda weight: weight[start_head:end_head])
            attention.unembed_out.apply(lambda weight: weight[start_head:end_head])
            attention.o_proj = shard_linear(
                attention.o_proj,
                "sharded-to-all",
                group=group,
            )

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = shard_linear(
                    layer.mlp.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                layer.mlp.down_proj = shard_linear(
                    layer.mlp.down_proj,
                    "sharded-to-all",
                    group=group,
                )
                layer.mlp.up_proj = shard_linear(
                    layer.mlp.up_proj,
                    "all-to-sharded",
                    group=group,
                )
                continue

            layer.mlp.sharding_group = group
            if getattr(layer.mlp, "shared_experts", None) is not None:
                shard_inplace(
                    layer.mlp.shared_experts.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                shard_inplace(
                    layer.mlp.shared_experts.down_proj,
                    "sharded-to-all",
                    group=group,
                )
                shard_inplace(
                    layer.mlp.shared_experts.up_proj,
                    "all-to-sharded",
                    group=group,
                )
            shard_inplace(
                layer.mlp.switch_mlp.gate_proj,
                "all-to-sharded",
                group=group,
            )
            shard_inplace(
                layer.mlp.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            shard_inplace(
                layer.mlp.switch_mlp.up_proj,
                "all-to-sharded",
                group=group,
            )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def cast_predicate(self):
        def predicate(key):
            return "e_score_correction_bias" not in key

        return predicate

    def make_cache(self):
        from ..cache import KVCache

        return [KVCache() for _ in self.layers]
