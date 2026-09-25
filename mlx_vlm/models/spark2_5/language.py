import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache, RotatingKVCache
from .config import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Spark fuses q/k/v into one projection and gates the attention output
        # per head with a sigmoid produced from a separate small projection.
        self.q_k_v_proj = nn.Linear(
            dim, self.q_dim + 2 * self.kv_dim, bias=config.attention_bias
        )
        self.out_proj = nn.Linear(self.q_dim, dim, bias=config.attention_bias)
        self.headwise_gate = config.headwise_attn_output_gate
        self.gate_mode = config.gate_attn_act_mode
        if self.headwise_gate:
            self.g_proj = nn.Linear(dim, self.n_heads, bias=config.attention_bias)

        layer_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )
        rope_dims, rope_base = config.rope_for(layer_type)
        self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_base)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape

        qkv = self.q_k_v_proj(x)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim :]

        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )

        if self.headwise_gate:
            gate = self.g_proj(x).reshape(B, L, self.n_heads, 1).transpose(0, 2, 1, 3)
            gate = gate.astype(mx.float32)
            gate = mx.sigmoid(gate) if self.gate_mode == "sigmoid" else nn.silu(gate)
            out = out * gate.astype(out.dtype)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Spark2_5Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.window_size = config.sliding_window
        # Masks are built from a representative cache of each kind; sliding and
        # full layers hold different cache types with different offsets.
        lt = config.layer_types
        self._swa_idx = (
            lt.index("sliding_attention") if "sliding_attention" in lt else 0
        )
        self._full_idx = lt.index("full_attention") if "full_attention" in lt else 0

    def __call__(self, inputs=None, cache=None, input_embeddings=None):
        h = input_embeddings if input_embeddings is not None else self.embedding(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(h, cache[self._full_idx])
        swa_mask = create_attention_mask(
            h, cache[self._swa_idx], window_size=self.window_size
        )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_sliding = self.config.layer_types[i] == "sliding_attention"
            h = layer(h, swa_mask if is_sliding else full_mask, c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.args = config
        self.config = config
        self.model_type = config.model_type
        self.model = Spark2_5Model(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings

        out = self.model(inputs, cache, inputs_embeds)
        if self.config.tie_word_embeddings:
            out = self.model.embedding.as_linear(out)
        else:
            out = self.lm_head(out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        if self.config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def make_cache(self):
        # Sliding layers only ever attend to the last `sliding_window` tokens,
        # so bound their cache with a rotating buffer instead of growing it to
        # full context. Full-attention layers keep an unbounded cache.
        caches = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention":
                caches.append(
                    RotatingKVCache(max_size=self.config.sliding_window, keep=0)
                )
            else:
                caches.append(KVCache())
        return caches

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
