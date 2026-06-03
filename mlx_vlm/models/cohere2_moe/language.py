from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from .config import ModelConfig


def is_prefix_dense_layer(args: ModelConfig, layer_idx: int):
    return layer_idx < args.first_k_dense_replace


def is_sliding_layer(args: ModelConfig, layer_idx: int):
    if is_prefix_dense_layer(args, layer_idx):
        return False
    if args.layer_types is not None:
        return args.layer_types[layer_idx] == "sliding_attention"
    return (layer_idx + 1) % args.sliding_window_pattern != 0


def norm_layer(args: ModelConfig):
    if args.rms_norm_eps is not None:
        return nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    return nn.LayerNorm(
        args.hidden_size, eps=args.layer_norm_eps, bias=args.layer_norm_bias
    )


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        attention_bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = nn.RoPE(head_dim, traditional=True, base=args.rope_theta)

        self.use_sliding_window = is_sliding_layer(args, layer_idx)
        self.force_rope = (
            is_prefix_dense_layer(args, layer_idx)
            and args.prefix_dense_sliding_window_pattern == 1
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.use_sliding_window or self.force_rope:
            if cache is None:
                queries = self.rope(queries)
                keys = self.rope(keys)
            else:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        sdpa_type = mx.float32 if queries.dtype == mx.float16 else queries.dtype
        output = scaled_dot_product_attention(
            queries.astype(sdpa_type),
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        ).astype(queries.dtype)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CohereMoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.intermediate_size

        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

        if getattr(args, "moe_num_shared_experts", 0) > 0:
            shared_intermediate_size = (
                args.intermediate_size * args.moe_num_shared_experts
            )
            self.shared_experts = MLP(args.hidden_size, shared_intermediate_size)
            self.shared_expert_combination_strategy = (
                args.shared_expert_combination_strategy
            )
            assert self.shared_expert_combination_strategy in [
                "average",
                "sum",
            ], "shared_expert_combination_strategy must be one of ['average', 'sum']"
        else:
            self.shared_experts = None
            self.shared_expert_combination_strategy = None

        if args.moe_gate_act == "softmax":
            self.gate_act = nn.Softmax()
        elif args.moe_gate_act == "sigmoid":
            self.gate_act = nn.Sigmoid()
        else:
            raise ValueError(f"{args.moe_gate_act} is not supported.")

    def __call__(self, x: mx.array):
        gates = self.gate(x)
        gates = self.gate_act(gates.astype(mx.float32))

        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)

        if self.shared_experts is not None:
            if self.shared_expert_combination_strategy == "average":
                y = (y + self.shared_experts(x)) / 2
            else:
                y = y + self.shared_experts(x)

        return y


class CohereMoEDecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads

        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            MLP(args.hidden_size, args.prefix_dense_intermediate_size)
            if is_prefix_dense_layer(args, layer_idx)
            else CohereMoeSparseMoeBlock(args)
        )
        self.input_layernorm = norm_layer(args)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)
        attn_h = self.self_attn(h, mask, cache)
        ff_h = self.mlp(h)
        return attn_h + ff_h + x


class CohereModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.window_size = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            CohereMoEDecoderLayer(args=args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = norm_layer(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            mask = create_attention_mask(
                h,
                c,
                window_size=(
                    self.window_size if layer.self_attn.use_sliding_window else None
                ),
            )
            h = layer(h, mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.model_type = args.model_type
        self.model = CohereModel(args)
        self.args = args
        self.config = args

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        out = self.model(inputs, cache, inputs_embeds)
        out = self.model.embed_tokens.as_linear(out)
        out = out * self.model.args.logit_scale
        return LanguageModelOutput(logits=out)

    def make_cache(self):
        caches = []
        for i in range(self.args.num_hidden_layers):
            if is_sliding_layer(self.args, i):
                caches.append(
                    RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights):
        for layer_idx in range(self.args.num_hidden_layers):
            if is_prefix_dense_layer(self.args, layer_idx):
                continue
            prefix = f"model.layers.{layer_idx}"
            for name in ["up_proj", "down_proj", "gate_proj"]:
                for suffix in ["weight", "scales", "biases"]:
                    first_key = f"{prefix}.mlp.experts.0.{name}.{suffix}"
                    if first_key not in weights:
                        continue
                    weights[f"{prefix}.mlp.switch_mlp.{name}.{suffix}"] = mx.stack(
                        [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{name}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                    )

        for key in list(weights.keys()):
            if "rotary_emb.inv_freq" in key:
                weights.pop(key)
            elif key.endswith(".bias"):
                if ".mlp." in key:
                    weights.pop(key)
                elif ".self_attn." in key and not self.args.attention_bias:
                    weights.pop(key)
                elif "layernorm" in key.lower() and not self.args.layer_norm_bias:
                    weights.pop(key)

        return weights

    @property
    def quant_predicate(self):
        def predicate(path, module):
            if ".self_attn." in path:
                return False
            if ".mlp.gate" in path and "gate_proj" not in path:
                return False
            return True

        return predicate

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
