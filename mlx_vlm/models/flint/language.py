import mlx.core as mx
import mlx.nn as nn

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import KVCache, RotatingKVCache
from ..mlp import SwiGLUMLP
from ..rope_utils import initialize_rope
from ..switch_layers import SwiGLU, SwitchGLU, stack_expert_projection
from .attention_mask import create_attention_mask
from .compilation import CompiledStage, linear_signature
from .config import ModelConfig
from .kernels import down_reduce, gate_up_silu, metal_sum, route_experts
from .normalization import norm_rope, residual_norm


class Attention(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.window_size = (
            args.sliding_window
            if args.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        self.q_proj = nn.Linear(
            args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((self.n_heads,))
        self.rope = None
        if layer_idx not in args.no_rope_layer_indices:
            self.rope = initialize_rope(
                dims=self.head_dim,
                base=args.rope_theta,
                traditional=False,
                scaling_config=args.rope_scaling,
                max_position_embeddings=args.max_position_embeddings,
            )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        offset = cache.offset if cache is not None else 0
        normalized = norm_rope(self, q, k, offset)
        if normalized is None:
            q = self.q_norm(
                q.reshape(batch, length, self.n_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            k = self.k_norm(
                k.reshape(batch, length, self.n_kv_heads, self.head_dim)
            ).transpose(0, 2, 1, 3)
            if self.rope is not None:
                q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        else:
            q, k = normalized
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.softmax_offset.astype(q.dtype),
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,))
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = (
            SwiGLUMLP(args.hidden_size, args.shared_expert_intermediate_size)
            if args.shared_expert_intermediate_size > 0
            else None
        )
        # Keep weights as inputs and reuse the graph across batch/prompt sizes.
        # Index selection and expert dispatch stay outside this shape-independent
        # graph; both can choose different algorithms as the workload changes.
        self._compiled = mx.compile(self._prepare, inputs=self, shapeless=True)
        self._compiled_inference = CompiledStage(self._inference, self)

    def route(self, x):
        # The reference computes the projection in the activation dtype, then
        # sigmoid and normalization in FP32. Bias affects selection only.
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        choice = scores + self.expert_bias.astype(mx.float32)
        indices = mx.stop_gradient(
            mx.argpartition(choice, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        )
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return indices, weights

    def _forward(self, x):
        indices, weights = self.route(x)
        routed = self.switch_mlp(x, indices)
        out = (
            (routed.astype(mx.float32) * weights[..., None])
            .sum(axis=-2)
            .astype(x.dtype)
        )
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
        return out

    def _prepare(self, x):
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        choice = scores + self.expert_bias.astype(mx.float32)
        shared = (
            self.shared_expert(x)
            if self.shared_expert is not None
            else mx.zeros_like(x)
        )
        return scores, choice, shared

    def _inference(self, x):
        scores, choice, shared = self._compiled(x)
        indices, weights = route_experts(scores, choice, self.top_k)
        switch = self.switch_mlp
        activated = gate_up_silu(switch, x, indices)
        if activated is not None:
            output = down_reduce(switch.down_proj, activated, indices, weights, shared)
            if output is not None:
                return output
            routed = switch.down_proj(activated[..., None, :], indices).squeeze(-2)
        else:
            routed = switch(x, indices)
        return metal_sum(
            routed, weights, shared if self.shared_expert is not None else None
        )

    def __call__(self, x):
        # Paged experts perform data-dependent I/O and cannot be traced.
        if self.training or type(self.switch_mlp) is not SwitchGLU:
            return self._forward(x)
        switch = self.switch_mlp
        shared = self.shared_expert
        if type(switch.activation) is not SwiGLU or (
            shared is not None and type(shared) is not SwiGLUMLP
        ):
            return self._forward(x)
        projections = [self.gate, switch.gate_proj, switch.up_proj, switch.down_proj]
        if shared is not None:
            projections += [shared.gate_proj, shared.up_proj, shared.down_proj]
        signature = tuple(linear_signature(p) for p in projections)
        if any(p is None for p in signature):
            return self._forward(x)
        if mx.default_device() == mx.gpu:
            return self._compiled_inference(
                x, signature=(self.top_k, switch.training, signature)
            )
        return self._inference(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = (
            SwiGLUMLP(args.hidden_size, args.intermediate_size)
            if layer_idx in args.mlp_only_layers
            else SparseMoeBlock(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        x, normalized = residual_norm(
            x,
            self.self_attn(self.input_layernorm(x), mask, cache),
            self.post_attention_layernorm,
        )
        return x + self.mlp(normalized)


class FlintModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, inputs_embeds=None):
        x = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        cache = [None] * len(self.layers) if cache is None else cache
        mask_parameters = {}
        # Carry the normalization into the next layer so its residual addition
        # and RMS reduction share a kernel. Training and custom layers retain
        # their normal module calls.
        if (
            not self.training
            and self.layers
            and len(cache) == len(self.layers)
            and all(type(layer) is DecoderLayer for layer in self.layers)
        ):
            normalized = self.layers[0].input_layernorm(x)
            for i, (layer, c) in enumerate(zip(self.layers, cache)):
                mask = create_attention_mask(
                    x,
                    c,
                    window_size=layer.self_attn.window_size,
                    parameter_cache=mask_parameters,
                )
                x, normalized = residual_norm(
                    x,
                    layer.self_attn(normalized, mask, c),
                    layer.post_attention_layernorm,
                )
                next_norm = (
                    self.layers[i + 1].input_layernorm
                    if i + 1 < len(self.layers)
                    else self.norm
                )
                x, normalized = residual_norm(x, layer.mlp(normalized), next_norm)
            return normalized
        for layer, c in zip(self.layers, cache):
            mask = create_attention_mask(
                x,
                c,
                window_size=layer.self_attn.window_size,
                parameter_cache=mask_parameters,
            )
            x = layer(x, mask, c)
        return self.norm(x)


class LanguageModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self, inputs, cache=None, inputs_embeds=None, input_embeddings=None, **kwargs
    ):
        if input_embeddings is not None:
            inputs_embeds = input_embeddings
        hidden = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        # Match the reference serving configuration's FP32 output projection.
        hidden = hidden.astype(mx.float32)
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.args.tie_word_embeddings
            else self.lm_head(hidden)
        )
        return LanguageModelOutput(logits=logits)

    def sanitize(self, weights):
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        for i, layer in enumerate(self.layers):
            if not isinstance(layer.mlp, SparseMoeBlock):
                continue
            prefix = f"model.layers.{i}.mlp."
            for name in ("gate_proj", "up_proj", "down_proj"):
                stack_expert_projection(
                    weights,
                    f"{prefix}experts",
                    name,
                    f"{prefix}switch_mlp.{name}",
                    self.args.num_experts,
                )
            for alias in ("e_score_correction_bias", "expert_bias"):
                key = f"{prefix}gate.{alias}"
                if key in weights:
                    weights[f"{prefix}expert_bias"] = weights.pop(key)
            # Router correction bias is optional in the source checkpoint.
            if f"{prefix}gate.weight" in weights:
                weights.setdefault(
                    f"{prefix}expert_bias", mx.zeros((self.args.num_experts,))
                )
        return weights

    @property
    def quant_predicate(self):
        return lambda path, _: (
            {"group_size": 64, "bits": 8} if path.endswith("mlp.gate") else True
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=layer.self_attn.window_size, keep=0)
                if layer.self_attn.window_size is not None
                else KVCache()
            )
            for layer in self.layers
        ]
