import math
from functools import lru_cache
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ...kernels import moe_softmax_topk, moe_swiglu_4bit
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..mlp import DeepseekMLP as DeepseekV2MLP
from ..switch_layers import QuantizedSwitchLinear, SwitchGLU
from .config import TextConfig


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity

    linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (max_val - min_val)
    return mx.clip(linear_func, 0, 1)


class DeepseekV2YarnRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()
        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        freq_inter = scaling_factor * base ** (
            mx.arange(0, dim, 2, dtype=mx.float32) / dim
        )
        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_max_position_embeddings,
        )
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x = self.mscale * x
        return mx.fast.rope(
            x,
            x.shape[-1],
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class DeepseekV2Attention(nn.Module):
    def __init__(self, config: TextConfig):
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
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        if self.config.rope_scaling is None:
            self.rope = nn.RoPE(
                self.qk_rope_head_dim,
                traditional=self.config.rope_traditional,
                base=self.rope_theta,
            )
        else:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling.get("factor", 1)
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scale = self.scale * mscale * mscale

                rope_kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rope = DeepseekV2YarnRotaryEmbedding(
                    dim=self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **rope_kwargs,
                )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        if cache is not None:
            q_pe = self.rope(q_pe, cache.offset)
            k_pe = self.rope(k_pe, cache.offset)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys, values = cache.update_and_fetch(
                mx.concatenate([k_nope, k_pe], axis=-1), values
            )
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class LlamaAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        self.head_dim = head_dim = config.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if config.attention_bias:
            attention_bias = config.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        rope_scale = (
            1 / config.rope_scaling["factor"]
            if config.rope_scaling is not None
            and config.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=config.rope_traditional,
            base=config.rope_theta,
            scale=rope_scale,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


def _fused_decode_token(x: mx.array) -> bool:
    """One bfloat16 token on Metal: the fused MoE decode kernels apply."""
    return (
        x.ndim == 3
        and x.shape[0] * x.shape[1] == 1
        and x.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


def _affine4_stack(linear, dtype):
    """``(weight, scales, biases)`` of an affine 4-bit quantized projection,
    or None when the fused kernels do not cover it. A dense
    ``nn.QuantizedLinear`` is one expert; a ``QuantizedSwitchLinear`` a stack."""
    if (
        not isinstance(linear, (QuantizedSwitchLinear, nn.QuantizedLinear))
        or getattr(linear, "mode", "affine") != "affine"
        or linear.bits != 4
        or linear.group_size % 8
        or linear.get("biases") is None
        or "bias" in linear
        or linear.scales.dtype != dtype
        or linear.biases.dtype != dtype
    ):
        return None
    return linear.weight, linear.scales, linear.biases


def _fused_swiglu_stacks(owner, dtype):
    """The gate/up/down stacks and group size of ``owner`` (a ``SwitchGLU``
    or dense SwiGLU MLP) for ``moe_swiglu_4bit``, or None unless every
    projection is affine 4-bit with kernel-friendly shapes. The answer is
    cached on the owner until one of its weights is replaced."""
    projections = (owner.gate_proj, owner.up_proj, owner.down_proj)
    key = (*(id(p.get("weight")) for p in projections), dtype)
    cached = owner.__dict__.get("_fused_swiglu_cache")
    if cached is not None and cached[0] == key:
        return cached[1]
    stacks = _fused_swiglu_stacks_uncached(projections, dtype)
    # Bypasses nn.Module's attribute registration: not a parameter.
    object.__setattr__(owner, "_fused_swiglu_cache", (key, stacks))
    return stacks


def _fused_swiglu_stacks_uncached(projections, dtype):
    gate_proj, up_proj, down_proj = projections
    stacks = [_affine4_stack(p, dtype) for p in projections]
    if any(s is None for s in stacks):
        return None
    group_size = gate_proj.group_size
    gate, up, down = stacks
    I, D8 = gate[0].shape[-2:]
    D = D8 * 8
    E = gate[0].size // (I * D8)
    if (
        up_proj.group_size != group_size
        or down_proj.group_size != group_size
        or up[0].shape != gate[0].shape
        or down[0].shape[-2:] != (D, I // 8)
        or down[0].size != E * D * (I // 8)
        or I % 8
        or D % group_size
        or I % group_size
        or D % 4
    ):
        return None
    return gate, up, down, group_size


def fused_swiglu_experts(
    x: mx.array,
    inds: mx.array,
    scores: mx.array,
    switch_mlp,
    shared_mlp=None,
    addends=(),
) -> Optional[mx.array]:
    """The weighted routed SwiGLU experts of one token, plus the shared MLP
    and ``addends``, as two kernels bit-identical to the ``gather_qmm``
    graph; None when the inputs are outside the fused path (the caller
    keeps its graph). x is (1, 1, D) and the result (1, 1, D); inds and
    scores hold K values."""
    if not _fused_decode_token(x) or not 1 <= inds.size <= 30:
        return None
    stacks = _fused_swiglu_stacks(switch_mlp, x.dtype)
    if stacks is None:
        return None
    gate, up, down, group_size = stacks
    D = x.shape[-1]
    if gate[0].shape[-1] * 8 != D or any(a.shape != x.shape for a in addends):
        return None
    shared = None
    if shared_mlp is not None:
        shared = _fused_swiglu_stacks(shared_mlp, x.dtype)
        if (
            shared is None
            or shared[3] != group_size
            or shared[0][0].ndim != 2
            or shared[0][0].shape[-1] * 8 != D
        ):
            return None
        shared = shared[:3]
    return moe_swiglu_4bit(
        x,
        inds,
        scores,
        gate,
        up,
        down,
        group_size=group_size,
        shared=shared,
        addends=addends,
    )


@lru_cache
def _one_expert(dtype: mx.Dtype):
    """Expert 0 with weight one: a dense MLP as a one-expert stack."""
    return mx.array([0], dtype=mx.uint32), mx.array([1.0], dtype=dtype)


def fused_dense_swiglu(mlp, x: mx.array, addends=()) -> Optional[mx.array]:
    """A dense SwiGLU MLP (``gate_proj``/``up_proj``/``down_proj``) of one
    token as a single expert of the fused kernels, plus ``addends``."""
    if not _fused_decode_token(x):
        return None
    inds, scores = _one_expert(x.dtype)
    return fused_swiglu_experts(x, inds, scores, mlp, None, addends)


class MoEGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.scoring_func = config.scoring_func
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = mx.zeros((self.n_routed_experts))
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x):
        bsz, seq_len = x.shape[:2]
        # Layer-specific forced gate takes precedence
        layer_idx = getattr(self, "_layer_index", None)

        gates = x @ self.weight.T

        if (
            self.scoring_func == "softmax"
            and self.topk_method == "greedy"
            and self.n_routed_experts <= 128
            and gates.dtype == mx.bfloat16
            and _fused_decode_token(x)
        ):
            # softmax + argpartition + take_along_axis (+ scale) in one kernel,
            # bit-identical to the graph below.
            return moe_softmax_topk(
                gates, top_k=self.top_k, route_scale=self.routed_scaling_factor
            )

        if self.scoring_func == "softmax":
            scores = mx.softmax(gates, axis=-1, precise=True)  # type: ignore[call-arg]
        elif self.scoring_func == "sigmoid":
            scores = mx.sigmoid(gates)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")

        if self.topk_method == "greedy":
            flat_scores = scores
            k = self.top_k
            inds = mx.argpartition(flat_scores, kth=-k, axis=-1)[..., -k:]
            inds = mx.stop_gradient(inds)
            scores_selected = mx.take_along_axis(flat_scores, inds, axis=-1)

        elif self.topk_method == "noaux_tc":
            experts_per_group = self.n_routed_experts // self.n_group
            topk_group = self.topk_group if self.topk_group else self.n_group

            scores_bt = scores.reshape(bsz, seq_len, self.n_routed_experts)
            bias = self.e_score_correction_bias.astype(mx.float32)
            bias = mx.reshape(bias, (1, 1, self.n_routed_experts))
            corrected = scores_bt + bias
            corrected_groups = corrected.reshape(
                bsz, seq_len, self.n_group, experts_per_group
            )

            sorted_groups = mx.sort(corrected_groups, axis=-1)
            if experts_per_group >= 2:
                top_values = sorted_groups[..., -2:]
            else:
                top_values = sorted_groups
            group_scores = mx.sum(top_values, axis=-1)

            if topk_group < self.n_group:
                kth_group = self.n_group - topk_group
                group_idx = mx.argpartition(group_scores, kth=kth_group, axis=-1)[
                    ..., -topk_group:
                ]
                group_axis = mx.reshape(
                    mx.arange(self.n_group, dtype=mx.int32), (1, 1, self.n_group)
                )
                mask = mx.zeros(group_scores.shape, dtype=mx.bool_)
                for slot in range(topk_group):
                    idx = mx.expand_dims(group_idx[..., slot], axis=-1)
                    mask = mx.logical_or(mask, mx.equal(idx, group_axis))
            else:
                mask = mx.ones(group_scores.shape, dtype=mx.bool_)

            mask_expanded = mx.expand_dims(mask, axis=-1)
            corrected_masked = mx.where(
                mask_expanded,
                corrected_groups,
                mx.zeros_like(corrected_groups),
            )

            corrected_flat = corrected_masked.reshape(bsz, seq_len, -1)
            raw_flat = scores_bt

            total_experts = corrected_flat.shape[-1]
            kth_value = max(total_experts - self.top_k, 0)
            inds = mx.argpartition(corrected_flat, kth=kth_value, axis=-1)[
                ..., -self.top_k :
            ].astype(mx.int32)
            scores_selected = mx.take_along_axis(raw_flat, inds, axis=-1)

        else:
            raise ValueError(f"Unknown topk method: {self.topk_method}")

        scores_selected = scores_selected * self.routed_scaling_factor
        return inds, scores_selected


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )

        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size
            )

    def __call__(self, x, residual: Optional[mx.array] = None):
        """The MoE output, plus ``residual`` when given (the decoder layer's
        ``h + mlp(norm(h))``, so a single-token decode step can fold the
        shared experts and the residual into the fused expert kernel)."""
        inds, scores = self.gate(x)
        has_shared = self.config.n_shared_experts is not None
        if not self.training and isinstance(self.switch_mlp, SwitchGLU):
            y = fused_swiglu_experts(
                x,
                inds,
                scores,
                self.switch_mlp,
                self.shared_experts if has_shared else None,
                () if residual is None else (residual,),
            )
            if y is not None:
                return y
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        if has_shared:
            y = y + self.shared_experts(x)
        if residual is not None:
            y = y + residual
        return y


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.attn_type = config.attn_type
        self.self_attn = (
            DeepseekV2Attention(config)
            if self.attn_type == "DeepseekV2Attention"
            else LlamaAttention(config)
        )
        self.mlp = (
            DeepseekV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV2MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return self.mlp_residual(h)

    def mlp_residual(self, h: mx.array) -> mx.array:
        """``h + mlp(post_attention_layernorm(h))``; a single-token decode
        step folds the residual into the fused expert kernel."""
        normed = self.post_attention_layernorm(h)
        if isinstance(self.mlp, DeepseekV2MoE):
            return self.mlp(normed, residual=h)
        if isinstance(self.mlp, DeepseekV2MLP) and not self.training:
            out = fused_dense_swiglu(self.mlp, normed, (h,))
            if out is not None:
                return out
        return h + self.mlp(normed)


class DeepseekV2Model(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DeepseekV2DecoderLayer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:

        if inputs_embeds is None:
            h = self.embed_tokens(x)
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        **kwargs,
    ):
        out = self.model(inputs, mask=mask, inputs_embeds=inputs_embeds, cache=cache)
        out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        for l in range(self.config.num_hidden_layers):
            prefix = f"language_model.model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.config.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        if self.config.attn_type == "DeepseekV2Attention":
            return (
                self.config.qk_nope_head_dim + self.config.qk_rope_head_dim,
                self.config.v_head_dim,
            )
        else:
            return self.config.hidden_size // self.config.num_key_value_heads

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads
