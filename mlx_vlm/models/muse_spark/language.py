# Copyright © 2026 Apple Inc.

from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import shard_linear, sum_gradients

from ...kernels import (
    expert_routing,
    norm_add_norm,
    post_expert_norm_combine,
    pre_norm,
    qk_norm,
    qk_norm_rope,
    rms_norm_sigmoid_gate,
)
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..pipeline import PipelineMixin
from ..switch_layers import SwitchGLU
from .config import TextConfig


def gather_last_axis(x, group):
    shape = x.shape
    x = mx.distributed.all_gather(x, group=group)
    x = x.reshape(group.size(), *shape)
    return mx.moveaxis(x, 0, -2).reshape(*shape[:-1], -1)


@mx.compile
def precise_swiglu(x, gate):
    return (nn.silu(gate.astype(mx.float32)) * x.astype(mx.float32)).astype(x.dtype)


class InputScaleRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(self.weight * x, None, self.eps)


def residual_gate_coefficients(
    gate: mx.array, *, temperature: float = 0.3, eps: float = 1e-3
) -> tuple[mx.array, mx.array]:
    """Return the weights of the stream and of the branch for a gate."""
    w = gate.astype(mx.float32) / temperature
    beta = mx.sigmoid(w)
    alpha = mx.sqrt(mx.clip(mx.sigmoid(-w) * (1.0 + beta), eps, None))
    return alpha, beta


POOLED_TP_MAX_TOKENS = 8

# Added to the sum of the top-k router scores.
_ROUTER_SCORE_EPS = 1e-15


def row_parallel_linear(
    linear: nn.Linear, x: mx.array, group: mx.distributed.Group
) -> mx.array:
    assert "bias" not in linear, "Linear layers with a bias are not supported."
    rows = linear.weight.shape[0] // group.size()
    r = group.rank()
    part = x @ linear.weight[r * rows : (r + 1) * rows].T
    return gather_last_axis(part, group)


# TODO: mx.contiguous of a slice fails on arrays this large; copy in chunks.
_MAX_COPY_ELEMENTS = 1 << 31


def shard_experts_inplace(
    module: nn.Module, sharding: str, group: mx.distributed.Group
) -> None:
    N, r = group.size(), group.rank()
    updates = {}
    for name, w in module.parameters().items():
        if not isinstance(w, mx.array):
            continue
        if w.ndim < 3:
            # we don't shard scales for the nvfp4 weights
            continue
        axis = w.ndim - 2 if sharding == "all-to-sharded" else w.ndim - 1
        n = w.shape[axis] // N
        lo, hi = r * n, (r + 1) * n
        per_expert = w.size // w.shape[0]
        step = max(1, _MAX_COPY_ELEMENTS // (2 * per_expert))
        parts = []
        for i in range(0, w.shape[0], step):
            chunk = w[i : i + step]
            chunk = chunk[:, lo:hi] if axis == 1 else chunk[..., lo:hi]
            chunk = mx.contiguous(chunk)
            mx.eval(chunk)
            parts.append(chunk)
        updates[name] = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    module.update(updates)


class Attention(nn.Module):
    def __init__(self, args: TextConfig, *, use_rope: bool):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5
        # Queries carry a tuned constant scale of their own, on top of the
        # usual 1/sqrt(head_dim) applied inside the attention call
        self.query_scale = args.qk_scale_factor
        self.eps = args.rms_norm_eps

        self.qkvg_proj = nn.Linear(
            dim, 2 * (n_heads + n_kv_heads) * head_dim, bias=False
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.use_rope = use_rope
        if use_rope:
            self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        qkvg = self.qkvg_proj(x)
        norm_args = {
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "eps": self.eps,
            "query_scale": self.query_scale,
        }
        if self.use_rope:
            # Also rotates q and k, at the positions after the cache.
            offset = cache.offset if cache is not None else 0
            queries, keys = qk_norm_rope(
                qkvg, **norm_args, rope_base=self.rope.base, offset=offset
            )
        else:
            queries, keys = qk_norm(qkvg, **norm_args)
        values = qkvg[..., q_dim + kv_dim : q_dim + 2 * kv_dim]
        gate = qkvg[..., q_dim + 2 * kv_dim :]

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        return self.o_proj(rms_norm_sigmoid_gate(output, gate, eps=self.eps))


class PreciseSwiGLU(nn.Module):
    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return precise_swiglu(x, gate)


class MoEGate(nn.Module):
    """The router, (num_experts, dim) for a gemv, and its score bias."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, dim), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)


class MoE(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()

        dim = args.hidden_size
        compressed = args.moe_hidden_size
        num_experts = args.num_local_experts

        self.top_k = args.num_experts_per_tok
        self.route_scale = args.routed_scaling_factor
        self.score_eps = _ROUTER_SCORE_EPS
        self.router_multiplier = args.output_multiplier

        self.gate = MoEGate(dim, num_experts)

        # The experts run at a compressed width, between the two projections.
        self.pre_expert_proj = nn.Linear(dim, compressed, bias=False)
        self.pre_expert_norm = nn.RMSNorm(compressed, eps=args.rms_norm_eps)
        self.experts = SwitchGLU(
            compressed,
            args.moe_intermediate_size,
            num_experts,
            activation=PreciseSwiGLU(),
        )
        # The checkpoint stores this norm under the experts.
        self.experts.post_expert_norm = InputScaleRMSNorm(
            compressed, eps=args.post_norm_eps
        )
        self.post_expert_proj = nn.Linear(compressed, dim, bias=False)

        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            x = sum_gradients(group)(x)
        # Quantized expert projections stay whole.
        split = (
            group is not None
            and group.size() > 1
            and x.size // x.shape[-1] <= POOLED_TP_MAX_TOKENS
            and isinstance(self.pre_expert_proj, nn.Linear)
            and isinstance(self.post_expert_proj, nn.Linear)
        )

        logits = x.astype(mx.float32) @ self.gate.weight.T
        inds, weights = expert_routing(
            logits,
            self.gate.e_score_correction_bias,
            top_k=self.top_k,
            multiplier=self.router_multiplier,
            score_eps=self.score_eps,
            route_scale=self.route_scale,
        )

        h = (
            row_parallel_linear(self.pre_expert_proj, x, group)
            if split
            else self.pre_expert_proj(x)
        )
        h = self.experts(self.pre_expert_norm(h), inds)
        if group is not None:
            # Sum the expert outputs before the post-expert norm.
            h = mx.distributed.all_sum(h, group=group)
        norm = self.experts.post_expert_norm
        h = post_expert_norm_combine(h, weights, norm_weight=norm.weight, eps=norm.eps)
        return (
            row_parallel_linear(self.post_expert_proj, h, group)
            if split
            else self.post_expert_proj(h)
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: TextConfig, *, use_rope: bool, use_sliding: bool):
        super().__init__()

        dim = args.hidden_size
        self.use_sliding = use_sliding
        self.post_norm_eps = args.post_norm_eps

        self.input_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.self_attn = Attention(args, use_rope=use_rope)
        self.pre_feedforward_layernorm = nn.RMSNorm(dim, eps=args.rms_norm_eps)
        self.mlp = MoE(args)
        self.post_feedforward_layernorm = InputScaleRMSNorm(dim, eps=args.post_norm_eps)
        # Sanitize computes these from the fixed gates. The defaults are the
        # values for a zero gate.
        zero_gate = mx.zeros((dim,))
        self.add_attn_alpha, self.add_attn_beta = residual_gate_coefficients(zero_gate)
        self.add_ffn_alpha, self.add_ffn_beta = residual_gate_coefficients(zero_gate)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Any | None = None,
        attn_input: mx.array | None = None,
        next_norm: nn.Module | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if attn_input is None:
            attn_input = self.input_layernorm(x).astype(mx.bfloat16)
        r = self.self_attn(attn_input, mask, cache)
        h, ffn_input = norm_add_norm(
            r,
            x,
            alpha=self.add_attn_alpha,
            beta=self.add_attn_beta,
            post_weight=None,
            post_eps=self.post_norm_eps,
            next_norm=self.pre_feedforward_layernorm,
        )
        r = self.mlp(ffn_input)
        return norm_add_norm(
            r,
            h,
            alpha=self.add_ffn_alpha,
            beta=self.add_ffn_beta,
            post_weight=self.post_feedforward_layernorm.weight,
            post_eps=self.post_feedforward_layernorm.eps,
            next_norm=next_norm,
        )


class MuseSparkTextModel(PipelineMixin, nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(
                args,
                use_rope=bool(rope_theta),
                use_sliding=layer_type == "sliding_attention",
            )
            for rope_theta, layer_type in zip(args.layer_rope_theta, args.layer_types)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @property
    def fa_idx(self) -> int:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if not l.use_sliding), 0)

    @property
    def swa_idx(self) -> int | None:
        layers = self.pipeline_layers
        return next((i for i, l in enumerate(layers) if l.use_sliding), None)

    def embed(self, inputs: mx.array) -> mx.array:
        """The normalized token embeddings."""
        return mx.fast.rms_norm(self.embed_tokens(inputs), None, self.args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        *,
        cache: list[Any] | None = None,
        inputs_embeds: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        # inputs_embeds are already normalized.
        h = self.embed(inputs) if inputs_embeds is None else inputs_embeds
        h = h.astype(mx.float32)

        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)

        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported attention mask: {mask}")
            mask = None
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        if mask is not None:
            if mask.ndim == 2:
                mask = (mask != 0)[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None]
            additive = mx.issubdtype(mask.dtype, mx.floating)
            if additive:
                # Fully padded query rows must stay finite so their NaNs do
                # not enter the next layer's keys, even when those keys are masked.
                mask = mx.maximum(mask, mx.finfo(mask.dtype).min)

            def combine(causal):
                if causal is None:
                    return mask
                if mask.dtype == mx.bool_ or mx.issubdtype(mask.dtype, mx.integer):
                    return causal & mask.astype(mx.bool_)
                return mx.where(causal, mask, mx.finfo(mask.dtype).min)

            fa_mask = combine(
                create_attention_mask(h, cache[self.fa_idx], return_array=True)
            )
            if self.swa_idx is not None:
                swa_mask = combine(
                    create_attention_mask(
                        h,
                        cache[self.swa_idx],
                        window_size=self.sliding_window,
                        return_array=True,
                    )
                )

        rank, size = self.pipeline_rank, self.pipeline_size
        attn_input = None
        if rank < size - 1:
            h = mx.distributed.recv_like(h, rank + 1)
            # Equal to the fused norm of the previous stage, not to input_layernorm.
            norm = layers[0].input_layernorm
            attn_input = pre_norm(h, norm, dtype=norm.weight.dtype)

        # Each block also makes the next block's attention input, so that the
        # norm is fused with its add.
        for i, (layer, c) in enumerate(zip(layers, cache)):
            mask = swa_mask if layer.use_sliding else fa_mask
            nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else None
            h, attn_input = layer(
                h, mask=mask, cache=c, attn_input=attn_input, next_norm=nxt
            )

        # Send to the next process in the pipeline
        if rank != 0:
            h = mx.distributed.send(h, (rank - 1) % size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph
        if size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        # The norm reads the fp32 stream.
        return self.norm(h).astype(mx.bfloat16)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = self.args = config
        self.model = MuseSparkTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        mask=None,
        logits_to_keep=0,
        **kwargs,
    ):
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds, mask=mask)
        if logits_to_keep:
            out = out[:, -logits_to_keep:]
        out = self.lm_head(out.astype(self.lm_head.weight.dtype))
        if self.sharding_group is not None:
            out = gather_last_axis(out, self.sharding_group)
        out = out.astype(mx.float32) * self.args.output_multiplier
        cap = self.args.final_logit_softcapping
        return LanguageModelOutput(logits=mx.tanh(out / cap) * cap if cap else out)

    @property
    def layers(self):
        return self.model.pipeline_layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    def shard(self, group: mx.distributed.Group | None = None) -> None:
        group = group or mx.distributed.init()
        N = group.size()
        n_kv_heads = self.args.num_key_value_heads
        assert (
            n_kv_heads % N == 0
        ), f"group size {N} must divide num_key_value_heads {n_kv_heads}"
        vocab = self.args.vocab_size
        assert vocab % N == 0, f"group size {N} must divide vocab_size {vocab}"
        self.lm_head = shard_linear(self.lm_head, "all-to-sharded", group=group)
        self.sharding_group = group
        for layer in self.layers:
            attn = layer.self_attn
            q_dim = attn.n_heads * attn.head_dim
            kv_dim = attn.n_kv_heads * attn.head_dim
            attn.qkvg_proj = shard_linear(
                attn.qkvg_proj,
                "all-to-sharded",
                segments=[q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
                group=group,
            )
            attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=group)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            layer.mlp.sharding_group = group
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_proj, "all-to-sharded", group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", group)
            shard_experts_inplace(experts.up_proj, "all-to-sharded", group)
