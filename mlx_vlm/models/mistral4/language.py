from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import create_attention_mask, scaled_dot_product_attention


def _get_llama_4_attn_scale(
    start: int, stop: int, beta: float, max_position_embeddings: int
):
    scaling = 1 + beta * mx.log(
        1 + mx.floor(mx.arange(start, stop) / max_position_embeddings)
    )
    return scaling[:, None]


class Mistral4Attention(nn.Module):
    """Multi-Latent Attention (MLA) with compressed KV projections."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.scale = self.qk_head_dim**-0.5

        # Query: optional LoRA compression
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(
                config.hidden_size,
                config.q_lora_rank,
                bias=config.attention_bias,
            )
            self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )

        # KV: LoRA compression + rope projection
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # RoPE for the rope portion only
        self.rope = initialize_rope(
            self.qk_rope_head_dim,
            config.rope_parameters["rope_theta"],
            config.rope_traditional,
            config.rope_parameters,
            config.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Query projection (optionally through LoRA)
        if self.q_lora_rank is None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

        q = q.reshape(B, L, self.num_heads, self.qk_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

        # KV projection: compressed representation + rope component
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)

        # k_pe is single-head (MQA for rope), will be expanded later
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)

        # Decompress KV through second projection
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

        # Apply RoPE to positional components only
        if cache is not None:
            q_pe = self.rope(q_pe, offset=cache.offset)
            k_pe = self.rope(k_pe, offset=cache.offset)
            k_pe = mx.broadcast_to(k_pe, k_nope.shape[:-1] + (self.qk_rope_head_dim,))
            keys = mx.concatenate([k_nope, k_pe], axis=-1)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            q_pe = self.rope(q_pe)
            k_pe = self.rope(k_pe)
            k_pe = mx.broadcast_to(k_pe, k_nope.shape[:-1] + (self.qk_rope_head_dim,))
            keys = mx.concatenate([k_nope, k_pe], axis=-1)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)

        # Apply Llama-4 position-dependent attention scaling
        queries = queries * attn_scale

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Mistral4MLP(nn.Module):
    def __init__(self, config, intermediate_size: int = None):
        super().__init__()
        dim = config.hidden_size
        hidden_dim = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Mistral4MoE(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        # Router
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)

        # Routed experts (SwitchGLU with 3D weight tensors)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )

        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Mistral4MLP(
                config, intermediate_size=shared_intermediate
            )

    def __call__(self, x: mx.array) -> mx.array:
        residuals = x

        # Route tokens to experts
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        # Top-k expert selection
        k = self.top_k
        inds = mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(gates, inds, axis=-1)

        if self.norm_topk_prob:
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)

        scores = scores * self.routed_scaling_factor

        # Dispatch to selected experts
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Add shared expert output
        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(residuals)

        return y


class Mistral4TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(config)

        if layer_idx >= config.first_k_dense_replace and config.n_routed_experts:
            self.mlp = Mistral4MoE(config)
        else:
            self.mlp = Mistral4MLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), attn_scale, mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Mistral4Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Mistral4TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Compute cache offset for attention scaling
        cache_offset = 0
        if cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        mask = create_attention_mask(h, cache[0])

        # Llama-4 position-dependent attention scaling
        attn_scale = _get_llama_4_attn_scale(
            cache_offset,
            cache_offset + h.shape[1],
            self.config.rope_parameters["llama_4_scaling_beta"],
            self.config.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        for layer, c in zip(self.layers, cache):
            h = layer(h, attn_scale, mask, cache=c)

        return self.norm(h)
