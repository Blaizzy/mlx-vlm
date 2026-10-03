"""Aleph Alpha Kolibri 1 (model_type "kolibri1").

Ported from the reference vLLM implementation in
https://github.com/Aleph-Alpha/aleph-alpha-inference (kolibri1.py).
Structurally close to afmoe; the differences that matter numerically:

- Routing selects the top-k on `logits + expert_bias` (not on
  `sigmoid(logits) + bias`), weights are `sigmoid(logits)` of the selected
  experts, no renormalisation (`norm_topk_prob: false`), no route scale.
  Router logits are computed in fp32.
- One ungated shared expert, added to the routed output.
- Sandwich norms: post_attn_norm after attention, post_ffn_norm after the
  MoE; post_attention_layernorm is the pre-MoE norm.
- Sliding-window layers use RoPE, full-attention layers use none.
- No attention output gate, no embedding scaling, every layer is MoE.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from ..mlp import SwiGLUMLP
from ..rope_utils import initialize_rope
from ..switch_layers import SwitchGLU
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, use_sliding: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # vLLM get_rope defaults to neox style, i.e. traditional=False.
        self.rope = (
            initialize_rope(
                self.head_dim,
                args.rope_theta,
                False,
                args.rope_scaling,
                args.max_position_embeddings,
            )
            if use_sliding
            else None
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Kolibri1MoE(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        self.experts = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = SwiGLUMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        if isinstance(self.gate, nn.QuantizedLinear):
            logits = self.gate(x).astype(mx.float32)
        else:
            logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T

        k = self.num_experts_per_tok
        selection = logits + self.expert_bias
        inds = mx.stop_gradient(
            mx.argpartition(-selection, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.sigmoid(mx.take_along_axis(logits, inds, axis=-1))
        if self.norm_topk_prob:
            scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)

        y = self.experts(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, use_sliding: bool):
        super().__init__()
        self.use_sliding = use_sliding
        self.self_attn = Attention(args, use_sliding)
        self.mlp = Kolibri1MoE(args)

        eps = args.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + self.post_attn_norm(r)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + self.post_ffn_norm(r)


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, use_sliding=t == "sliding_attention")
            for t in args.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.fa_idx = args.layer_types.index("full_attention")
        self.swa_idx = (
            args.layer_types.index("sliding_attention")
            if "sliding_attention" in args.layer_types
            else None
        )

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs: mx.array, cache=None, inputs_embeds=None, mask=None, **kwargs
    ):
        out = self.model(inputs, cache, inputs_embeds=inputs_embeds)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            bias_key = f"{prefix}.moe.router.expert_bias"
            if bias_key in weights:
                weights[f"{prefix}.mlp.expert_bias"] = weights.pop(bias_key)
            for n in ["gate_proj", "up_proj", "down_proj"]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{n}.{k}" in weights:
                        weights[f"{prefix}.mlp.experts.{n}.{k}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                        )
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        # The release keeps the router in bf16 (modules_to_not_convert).
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
