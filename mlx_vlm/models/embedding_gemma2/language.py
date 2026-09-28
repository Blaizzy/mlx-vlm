import mlx.core as mx
import mlx.nn as nn

from ..gemma4.language import RMSNormNoScale
from .config import TextConfig


class Attention(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        overrides = config.per_layer_config.get(f"{layer_idx:02d}", {})
        self.head_dim = overrides.get("head_dim", config.head_dim)
        self.num_heads = overrides.get(
            "num_attention_heads", config.num_attention_heads
        )
        self.num_kv_heads = overrides.get(
            "num_key_value_heads", config.num_key_value_heads
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)
        layer_type = config.layer_types[layer_idx]
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=config.rope_parameters[layer_type]["rope_theta"],
        )

    def __call__(self, x, mask):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.num_heads, -1))
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.num_kv_heads, -1))
        v = self.v_norm(self.v_proj(x).reshape(batch, length, self.num_kv_heads, -1))
        q = self.rope(q.transpose(0, 2, 1, 3))
        k = self.rope(k.transpose(0, 2, 1, 3))
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x):
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class PerLayerInputs(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.width = config.hidden_size_per_layer_input
        self.scale = config.hidden_size**-0.5
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, self.num_layers * self.width, bias=False
        )
        self.per_layer_projection_norm = nn.RMSNorm(self.width, eps=config.rms_norm_eps)

    def __call__(self, x):
        projected = self.per_layer_model_projection(x) * self.scale
        projected = projected.reshape(*x.shape[:-1], self.num_layers, self.width)
        return self.per_layer_projection_norm(projected)


class PerLayerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.per_layer_input_gate = nn.Linear(
            config.hidden_size, config.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            config.hidden_size_per_layer_input, config.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, per_layer_input):
        gate = nn.gelu_approx(self.per_layer_input_gate(x)) * per_layer_input
        return x + self.post_per_layer_input_norm(self.per_layer_projection(gate))


class EncoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ple_block = (
            PerLayerBlock(config) if config.hidden_size_per_layer_input else None
        )
        self.layer_scalar = mx.ones((1,))

    def __call__(self, x, mask, per_layer_input=None):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), mask)
        )
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x))
        )
        if self.ple_block is not None:
            x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.ple = (
            PerLayerInputs(config) if config.hidden_size_per_layer_input else None
        )
        self.layers = [EncoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_projection = nn.Linear(
            config.hidden_size, config.embedding_dim, bias=False
        )

    def __call__(self, inputs_embeds, attention_mask):
        length = inputs_embeds.shape[1]
        full_mask = attention_mask[:, None, None, :].astype(mx.bool_)
        positions = mx.arange(length)
        sliding_mask = full_mask & (
            mx.abs(positions[:, None] - positions[None, :]) < self.config.sliding_window
        )
        per_layer_inputs = self.ple(inputs_embeds) if self.ple is not None else None
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            mask = (
                sliding_mask
                if self.config.layer_types[i] == "sliding_attention"
                else full_mask
            )
            h = layer(
                h, mask, None if per_layer_inputs is None else per_layer_inputs[:, :, i]
            )
        return self.embedding_projection(self.norm(h))
