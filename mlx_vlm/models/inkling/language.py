from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, create_attention_mask
from ..cache import ArraysCache, CacheList, KVCache
from ..switch_layers import SwitchGLU
from .config import TextConfig


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    if n_rep == 1:
        return x
    B, H, L, D = x.shape
    x = mx.expand_dims(x, 2)
    x = mx.broadcast_to(x, (B, H, n_rep, L, D))
    return x.reshape(B, H * n_rep, L, D)


def log_sigmoid(x: mx.array) -> mx.array:
    return mx.minimum(x, 0.0) - mx.log1p(mx.exp(-mx.abs(x)))


class InklingShortConvolution(nn.Module):
    """Causal depthwise conv over the sequence axis with a residual add.

    Mirrors HF's `InklingShortConvolution`: `y = x + causal_conv1d(x)`. HF keeps
    this in fp32 for stability (`_keep_in_fp32_modules_strict`); we do the same.
    """

    def __init__(self, hidden_size: int, conv_kernel_size: int):
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.conv1d = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            bias=False,
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[ArraysCache] = None,
        cache_idx: int = 0,
    ) -> mx.array:
        input_dtype = x.dtype
        residual = x
        x = x.astype(mx.float32)

        if cache is not None:
            state = cache[cache_idx]
            if state is None:
                state = mx.zeros(
                    (x.shape[0], self.conv_kernel_size - 1, x.shape[-1]),
                    dtype=mx.float32,
                )
            x = mx.concatenate([state, x], axis=1)
            cache[cache_idx] = x[:, -(self.conv_kernel_size - 1) :, :]
        else:
            x = mx.pad(x, [(0, 0), (self.conv_kernel_size - 1, 0), (0, 0)])

        out = self.conv1d(x.astype(self.conv1d.weight.dtype))
        return (residual.astype(mx.float32) + out.astype(mx.float32)).astype(
            input_dtype
        )


class InklingRelativeLogits(nn.Module):
    """Hidden-state-conditioned relative position bias.

    `proj` is a bank of distance-vs-bias profiles; each token's relative
    projection mixes them into one bias value per backward distance. Zero
    outside `0 <= distance < rel_extent` (causality/padding are handled by the
    separate attention mask).
    """

    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(
        self,
        relative_states: mx.array,
        q_positions: mx.array,
        k_positions: mx.array,
    ) -> mx.array:
        # relative_states: (B, L, num_heads, d_rel) -> rel_logits: (B, num_heads, L, rel_extent)
        rel_logits = (relative_states @ self.proj).transpose(0, 2, 1, 3)
        distance = q_positions[:, None] - k_positions[None, :]
        gather_index = mx.clip(distance, 0, self.rel_extent - 1)[None, None]
        gather_index = mx.broadcast_to(
            gather_index, rel_logits.shape[:-1] + (gather_index.shape[-1],)
        )
        position_bias = mx.take_along_axis(rel_logits, gather_index, axis=-1)
        out_of_range = (distance < 0) | (distance >= self.rel_extent)
        return mx.where(out_of_range[None, None], mx.array(0.0, position_bias.dtype), position_bias)


class InklingAttention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        # q/k are per-head RMS-normalized, hence 1/d rather than 1/sqrt(d).
        self.scale = 1.0 / self.head_dim

        hidden_size = config.hidden_size
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.r_proj = nn.Linear(hidden_size, self.num_heads * config.d_rel, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        self.k_sconv = InklingShortConvolution(
            self.num_kv_heads * self.head_dim, config.conv_kernel_size
        )
        self.v_sconv = InklingShortConvolution(
            self.num_kv_heads * self.head_dim, config.conv_kernel_size
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_logits_proj = InklingRelativeLogits(config.d_rel, self.rel_extent)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[CacheList] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        kv_cache = cache[0] if cache is not None else None
        conv_cache = cache[1] if cache is not None else None

        q = self.q_proj(x)
        k = self.k_sconv(self.k_proj(x), cache=conv_cache, cache_idx=0)
        v = self.v_sconv(self.v_proj(x), cache=conv_cache, cache_idx=1)
        r = self.r_proj(x)

        q = self.q_norm(q.reshape(B, L, self.num_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(k.reshape(B, L, self.num_kv_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        r = r.reshape(B, L, self.num_heads, -1)

        # Mask must be built from the pre-update offset: it needs to cover
        # `[0, offset + L)` keys, which is exactly what `update_and_fetch`
        # is about to produce, but `cache.offset` is mutated in place by
        # `update_and_fetch`, so this has to happen first.
        mask = create_attention_mask(
            x, kv_cache, window_size=self.sliding_window, return_array=True
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(k, v)
        S = k.shape[2]

        q_positions = mx.arange(offset, offset + L)
        k_positions = mx.arange(S)
        position_bias = self.rel_logits_proj(r, q_positions, k_positions)

        if not self.is_sliding and self.config.log_scaling_n_floor is not None:
            effective_n = (q_positions + 1).astype(mx.float32)
            tau = 1.0 + self.config.log_scaling_alpha * mx.log(
                mx.maximum(effective_n / self.config.log_scaling_n_floor, 1.0)
            )
            tau = tau.reshape(1, 1, -1, 1)
            q = (q.astype(mx.float32) * tau).astype(q.dtype)
            position_bias = (position_bias.astype(mx.float32) * tau).astype(
                position_bias.dtype
            )

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = (q @ k.swapaxes(-1, -2)) * self.scale
        scores = scores + position_bias.astype(scores.dtype)
        if mask is not None:
            scores = mx.where(
                mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype)
            )
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class InklingMLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.global_scale = mx.ones((1,))

    def __call__(self, x: mx.array) -> mx.array:
        h = self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
        return h * self.global_scale


class InklingTopkRouter(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = self.num_experts + self.n_shared_experts
        self.route_scale = config.route_scale
        self.top_k = config.num_experts_per_tok

        self.weight = mx.zeros((self.n_total_experts, config.hidden_size))
        self.global_scale = mx.ones((1,))
        self.e_score_correction_bias = mx.zeros((self.num_experts,))

    def __call__(self, x: mx.array):
        router_logits = x @ self.weight.T

        scores = mx.sigmoid(router_logits.astype(mx.float32))
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice = routed_scores + self.e_score_correction_bias

        top_k = self.top_k
        topk_indices = mx.argpartition(-scores_for_choice, kth=top_k - 1, axis=-1)[
            ..., :top_k
        ]

        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]
        gathered = mx.take_along_axis(routed_logits, topk_indices, axis=-1)
        topk_logits = mx.concatenate([gathered, shared_logits], axis=-1)

        topk_log_probs = log_sigmoid(topk_logits.astype(mx.float32))
        denom = mx.logsumexp(topk_log_probs, axis=-1, keepdims=True)
        topk_weights = mx.exp(topk_log_probs - denom)
        topk_weights = topk_weights * self.route_scale * self.global_scale

        shared_gammas = topk_weights[..., -self.n_shared_experts :]
        topk_weights = topk_weights[..., :top_k]

        return topk_weights, topk_indices, shared_gammas


class InklingSharedExperts(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.n_shared_experts = config.n_shared_experts
        inter = config.moe_intermediate_size
        self.gate_proj = mx.zeros((self.n_shared_experts, inter, config.hidden_size))
        self.up_proj = mx.zeros((self.n_shared_experts, inter, config.hidden_size))
        self.down_proj = mx.zeros((self.n_shared_experts, config.hidden_size, inter))

    def __call__(self, x: mx.array, gammas: mx.array) -> mx.array:
        out = None
        for i in range(self.n_shared_experts):
            gate = x @ self.gate_proj[i].T
            up = x @ self.up_proj[i].T
            down = (nn.silu(gate) * up) @ self.down_proj[i].T
            down = down * gammas[..., i : i + 1]
            out = down if out is None else out + down
        return out


class InklingMoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate = InklingTopkRouter(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )
        self.shared_experts = InklingSharedExperts(config)

    def __call__(self, x: mx.array) -> mx.array:
        topk_weights, topk_indices, shared_gammas = self.gate(x)
        y = self.switch_mlp(x, topk_indices)
        y = (y * topk_weights[..., None].astype(y.dtype)).sum(axis=-2)
        y = y + self.shared_experts(x, shared_gammas)
        return y


class InklingDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = InklingAttention(config, layer_idx)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = InklingMoE(config)
        else:
            self.mlp = InklingMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.conv_kernel_size
        )
        self.mlp_sconv = InklingShortConvolution(
            config.hidden_size, config.conv_kernel_size
        )

    def __call__(
        self,
        x: mx.array,
        cache: Optional[CacheList] = None,
    ) -> mx.array:
        conv_cache = cache[1] if cache is not None else None

        residual = x
        h = self.self_attn(self.input_layernorm(x), cache=cache)
        h = self.attn_sconv(h, cache=conv_cache, cache_idx=2)
        h = residual + h

        residual = h
        out = self.mlp(self.post_attention_layernorm(h))
        out = self.mlp_sconv(out, cache=conv_cache, cache_idx=3)
        return residual + out


class InklingTextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            InklingDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        if inputs_embeds is None:
            h = self.embed_norm(self.embed_tokens(inputs))
        else:
            h = inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.args = config
        self.model_type = config.model_type
        self.model = InklingTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[list] = None,
        **kwargs: Any,
    ) -> LanguageModelOutput:
        h = self.model(inputs, inputs_embeds=inputs_embeds, cache=cache)
        h = h / self.config.logits_mup_width_multiplier
        logits = self.lm_head(h)
        unpadded_vocab_size = self.config.unpadded_vocab_size
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., :unpadded_vocab_size]
        return LanguageModelOutput(logits=logits)

    def make_cache(self):
        return [
            CacheList(KVCache(), ArraysCache(4)) for _ in range(len(self.model.layers))
        ]

    def sanitize(self, weights):
        # Prefix-agnostic (works whether this dict's keys are rooted at
        # "model.layers..." or "language_model.model.layers...").
        new_weights = {}
        for k, v in weights.items():
            if k.endswith(".mlp.experts.gate_up_proj"):
                base = k[: -len(".experts.gate_up_proj")]
                inter = v.shape[1] // 2
                new_weights[f"{base}.switch_mlp.gate_proj.weight"] = v[:, :inter, :]
                new_weights[f"{base}.switch_mlp.up_proj.weight"] = v[:, inter:, :]
            elif k.endswith(".mlp.experts.down_proj"):
                base = k[: -len(".experts.down_proj")]
                new_weights[f"{base}.switch_mlp.down_proj.weight"] = v
            elif k.endswith("sconv.conv1d.weight"):
                # torch Conv1d weight is (C, 1, K); MLX Conv1d expects (C, K, 1).
                new_weights[k] = v.transpose(0, 2, 1)
            else:
                new_weights[k] = v
        return new_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.config.head_dim

    @property
    def n_kv_heads(self):
        return self.config.num_key_value_heads

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate
